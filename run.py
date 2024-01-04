from time import time_ns, sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers import AutoTokenizer, AutoModel, GenerationConfig
from tqdm import tqdm

import gradio as gr

from hakuphi.model import PhiForCausalLM
from hakuphi.attn_patcher import apply_attn_algo
from hakuphi.inference import generate

from lycoris.wrapper import create_lycoris_from_weights
from modules.contrastive import ContrastiveScorer
from dataset import final
from data.translation import translate


def load_final_dataset(split="train"):
    dataset = final.load(split)["train"]
    return dataset


@torch.no_grad()
@torch.autocast("cuda")
def get_result(
    request,
    model,
    tokenizer,
    embed_model,
    scorer_model,
    course_num,
    course_name,
    course_name_en,
    pre_calc_pref_embed,
    pre_calc_sum_embed,
):
    print("=" * 50, "\n")
    prompt = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{request}

### Response:
"""
    prev = prompt
    t0 = time_ns()
    for i in tqdm(
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=0.9,
            top_p=0.8,
            top_k=45,
            repetition_penalty=1.05,
            max_new_tokens=64,
            stream_output=True,
        ),
        disable=True,
    ):
        if len(i) > len(prev):
            new_len = len(i) - len(prev)
            print(i[len(prev) :], end="", flush=True)
            prev = i
            yield i[len(prompt) :], {}
        pass
    t1 = time_ns()
    result = i[len(prompt) :]
    result_tokens = len(tokenizer.tokenize(result))
    print()
    print("=" * 50)
    print(f"Total generated tokens: {result_tokens}")
    print(f"Total cost time: {(t1-t0)/1e9:.2f}s")
    print(f"Average Speed: {(result_tokens/((t1-t0)/1e9)):.2f} tokens/sec")
    print()
    request_en = translate(request, "eng_Latn")
    torch.cuda.empty_cache()
    embed_model.cuda()
    result_embed = torch.tensor(embed_model.encode([result]))
    request_embed = torch.tensor(embed_model.encode([request_en]))
    embed_model.cpu()
    torch.cuda.empty_cache()
    preference_sim = torch.cosine_similarity(
        pre_calc_pref_embed, request_embed
    )
    contrastive_score = scorer_model(request_embed, pre_calc_sum_embed)[0]
    summary_sim = torch.cosine_similarity(
        pre_calc_sum_embed, result_embed
    )
    sim_topk = torch.topk(
        (
            preference_sim * 0.3
            + contrastive_score * 0.5
            + summary_sim * 0.7
        ),
        50
    ) # Take top 50 since we may have 5 repeat result for same course
    print("Top 10 similar courses: ")
    choosed = {}
    for idx in sim_topk.indices:
        if len(choosed) >= 10:
            break
        if f"{course_num[idx]} | {course_name[idx]} | {course_name_en[idx]}" in choosed:
            continue
        choosed[f"{course_num[idx]} | {course_name[idx]} | {course_name_en[idx]}"] = (
            summary_sim[idx].item() + 0.3 * preference_sim[idx].item()
        ) / 1.3
        print(
            f"Summary Similarity: {summary_sim[idx].item():.3f}, "
            f"Contrastive Score: {contrastive_score[idx].item():.3f}, "
            f"Preference Similarity: {preference_sim[idx].item():.3f}"
        )
        print(course_num[idx], course_name[idx], course_name_en[idx])
        print()
    yield result, choosed


if __name__ == "__main__":
    model: PhiForCausalLM = PhiForCausalLM.from_pretrained("microsoft/phi-2")
    model = model.half()
    lycoris_sd = torch.load("./models/lycoris-weights/epoch=4.pt", map_location="cpu")
    lycoris_net, _ = create_lycoris_from_weights(1.0, "", model, lycoris_sd)
    lycoris_net.half()
    lycoris_net.merge_to(1.0)
    model.transformer.h.to(torch.float8_e4m3fn)
    model.lm_head.to(torch.float8_e4m3fn)
    model.cuda()

    apply_attn_algo(model, algo="xformers")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    generate(model=model, tokenizer=tokenizer, prompt="Hello", max_new_tokens=16)

    dataset = load_final_dataset(split="summary")
    course_num = [data["course_number"] for data in dataset]
    course_name = [data["course_title"] for data in dataset]
    course_name_en = [data["course_title_en"] for data in dataset]
    pre_calc_sum_embed = torch.tensor(torch.load("./models/summary-embeddings.pt"))
    pre_calc_pref_embed = torch.tensor(torch.load("./models/preference-embeddings.pt"))
    embed_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    ).half()

    scorer_model = ContrastiveScorer.load_from_checkpoint(
        r"models\SigLIP\ver1.ckpt"
    ).cpu()

    def wrapper(request):
        yield from get_result(
            request,
            model,
            tokenizer,
            embed_model,
            scorer_model,
            course_num,
            course_name,
            course_name_en,
            pre_calc_pref_embed,
            pre_calc_sum_embed,
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                request = gr.TextArea(label="Input your request")
                submit = gr.Button("Submit")
            with gr.Column(scale=2):
                label = gr.Label(label="Possible Courses")
                result = gr.TextArea(label="Predicted Summary")
        submit.click(
            wrapper,
            inputs=[request],
            outputs=[result, label],
        )

    demo.launch(server_name="192.168.1.1", server_port=17415, max_threads=2)
