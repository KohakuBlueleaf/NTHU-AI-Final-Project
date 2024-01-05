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
from utils import slicepart


def load_final_dataset(split="train"):
    dataset = final.load(split)["train"]
    return dataset


def calc_score(summary_sim, contrastive_score, preference_sim):
    weight = torch.tensor([1.0, 1.0, 0.5])
    return (
        weight[0] * summary_sim
        + weight[1] * contrastive_score
        + weight[2] * preference_sim
    )


@torch.no_grad()
@torch.autocast("cuda")
def get_result(
    request,
    text_model,
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
    # Use LLM to predict possible summary
    # This prompt allow model itself to make request longer based on what it learned
    # Which will be better for preference sim and pref-sum contrastive scorer
    prompt = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{request}"""
    prev = prompt
    t0 = time_ns()
    for i in tqdm(
        generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=0.9,
            top_p=0.8,
            top_k=45,
            repetition_penalty=1.05,
            max_new_tokens=144,
            stream_output=True,
        ),
        disable=True,
    ):
        if len(i) > len(prev):
            new_len = len(i) - len(prev)
            print(i[len(prev) :], end="", flush=True)
            prev = i
            yield i.split("Response:\n", 1)[-1], {}
        pass
    t1 = time_ns()
    request = (
        i.split("Input:\n", 1)[-1].rsplit("Response:\n", 1)[0].rsplit("\n\n", 1)[0]
    )
    result = i.split("Response:\n", 1)[-1]
    result_tokens = len(tokenizer.tokenize(result))
    print()
    print("=" * 50)
    print(f"Total generated tokens: {result_tokens}")
    print(f"Total cost time: {(t1-t0)/1e9:.2f}s")
    print(f"Average Speed: {(result_tokens/((t1-t0)/1e9)):.2f} tokens/sec")
    print()

    # Calc Embedding for preference and generated summary
    request_en = translate(request, "eng_Latn")
    result_embed = torch.tensor(embed_model.encode([result]))
    request_embed = torch.tensor(embed_model.encode([request_en]))
    preference_sim = torch.cosine_similarity(pre_calc_pref_embed, request_embed)
    summary_sim = torch.cosine_similarity(pre_calc_sum_embed, result_embed)

    # Calc contrastive score based on preference and pre calc summary embedding
    contrastive_score = scorer_model(request_embed, pre_calc_sum_embed)[0]
    torch.cuda.empty_cache()

    # Get final score
    sim_score = calc_score(
        preference_sim,
        contrastive_score,
        summary_sim,
    )
    # Normalize
    sim_score = sim_score - sim_score.min()
    sim_score = sim_score / sim_score.max()

    # Select best fit
    print("Top 10 similar courses: ")
    choosed = {}
    for idx in torch.topk(sim_score, 100).indices:
        if len(choosed) >= 10:
            break
        name = f"{course_num[idx]} | {course_name[idx]} | {course_name_en[idx]}"
        if name in choosed:
            continue
        choosed[name] = float(sim_score[idx])
        print(
            f"Summary Similarity: {summary_sim[idx].item():.3f}, "
            f"Contrastive Score: {contrastive_score[idx].item():.3f}, "
            f"Preference Similarity: {preference_sim[idx].item():.3f}"
        )
        print(course_num[idx], course_name[idx], course_name_en[idx])
        print()
    yield result, choosed


if __name__ == "__main__":
    # Load metadata information
    dataset = load_final_dataset(split="summary")
    course_num = [data["course_number"] for data in dataset]
    course_name = [data["course_title"] for data in dataset]
    course_name_en = [data["course_title_en"] for data in dataset]

    # Load pre calc embeddings
    pre_calc_sum_embed = torch.tensor(torch.load("./models/summary-embeddings.pt"))
    pre_calc_pref_embed = torch.tensor(torch.load("./models/preference-embeddings.pt"))

    # Load LLM
    text_model: PhiForCausalLM = PhiForCausalLM.from_pretrained("microsoft/phi-2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    # Apply xformers optimization
    apply_attn_algo(text_model, algo="xformers")

    # Load LyCORIS model for LLM and apply it
    lycoris_sd = torch.load("./models/lycoris-weights/epoch=4.pt", map_location="cpu")
    lycoris_net, _ = create_lycoris_from_weights(1.0, "", text_model, lycoris_sd)
    lycoris_net.to(next(text_model.parameters()).dtype)
    lycoris_net.merge_to(1.0)

    # Cast LLM to FP8 for efficiency
    text_model.transformer.h.to(torch.float8_e4m3fn)
    text_model.lm_head.to(torch.float8_e4m3fn)
    text_model.cuda()

    # Load Jina-Emb model and contrastive scorer model
    embed_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    ).cpu()
    scorer_model = ContrastiveScorer.load_from_checkpoint(
        r"models\SigLIP\ver1.ckpt"
    ).cpu()

    def wrapper(request):
        yield from get_result(
            request,
            text_model,
            tokenizer,
            embed_model,
            scorer_model,
            course_num,
            course_name,
            course_name_en,
            pre_calc_pref_embed,
            pre_calc_sum_embed,
        )

    with gr.Blocks(theme=gr.themes.Soft) as demo:
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
