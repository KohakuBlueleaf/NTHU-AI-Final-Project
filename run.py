import sys
sys.setrecursionlimit(10000000)
from time import time_ns, sleep

import torch

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import gradio as gr

from hakuphi.model import PhiForCausalLM
from hakuphi.attn_patcher import apply_attn_algo
from hakuphi.inference import generate

from lycoris.wrapper import create_lycoris_from_weights
from modules.contrastive import ContrastiveScorer
from dataset import final
from data.translation import translate
from utils import normalize, manual_cast
from config import *


def load_final_dataset(split="train"):
    dataset = final.load(split)["train"]
    return dataset


def make_autocast():
    if not torch.cuda.is_available():
        return manual_cast(device, dtype)
    else:
        return torch.autocast("cuda")


@torch.no_grad()
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
    weight_sum_sim,
    weight_con_sim,
    weight_pref_sim,
    expand_input,
):
    print("=" * 50, "\n")
    # Use LLM to predict possible summary
    # This prompt allow model itself to make request longer based on what it learned
    # Which will be better for preference sim and pref-sum contrastive scorer
    prompt = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{request}"""
    prev = ""
    if not expand_input:
        prompt += "\n\n### Response:\n"

    t0 = time_ns()
    for llm_gen in tqdm(
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
            autocast_gen=make_autocast,
        ),
        disable=True,
    ):
        if len(llm_gen) > len(prev):
            print(llm_gen[len(prev) :], end="", flush=True)
            prev = llm_gen
            yield {}, llm_gen, {}, {}, {}
        pass
    t1 = time_ns()
    request = (
        llm_gen.split("Input:\n", 1)[-1]
        .rsplit("Response:\n", 1)[0]
        .rsplit("\n\n", 1)[0]
    )
    result = llm_gen.split("Response:\n", 1)[-1]
    result_tokens = len(tokenizer.tokenize(result))
    print()
    print("=" * 50)
    print(f"Request: {request}")
    print(f"Result: {result}")
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
    torch.mps.empty_cache()

    # Normalize all the score
    summary_sim = normalize(summary_sim)
    contrastive_score = normalize(contrastive_score)
    preference_sim = normalize(preference_sim)

    # Get final score
    sim_score = (
        preference_sim * weight_pref_sim
        + summary_sim * weight_sum_sim
        + contrastive_score * weight_con_sim
    )
    # normalize
    sim_score = normalize(sim_score)

    # Select best fit
    print("Top 10 similar courses: ")
    choosed = {}
    choosed_sum_sim = {}
    choosed_con_sim = {}
    choosed_pref_sim = {}
    for idx in torch.topk(sim_score, 1000).indices:
        if len(choosed) >= 10:
            break
        # don't take course number as part of name, so we will filter out same name
        # Since the result is sorted by topk, only the top score of same name will be count
        name = f"{course_name[idx]} | {course_name_en[idx]}"
        if name in choosed:
            continue
        choosed[name] = float(sim_score[idx])
        choosed_sum_sim[name] = float(summary_sim[idx])
        choosed_con_sim[name] = float(contrastive_score[idx])
        choosed_pref_sim[name] = float(preference_sim[idx])
        print(
            f"Summary Similarity: {summary_sim[idx].item():.3f}, "
            f"Contrastive Score: {contrastive_score[idx].item():.3f}, "
            f"Preference Similarity: {preference_sim[idx].item():.3f}"
        )
        print(course_num[idx], course_name[idx], course_name_en[idx])
        print()
    
    for idx, (k, v) in enumerate(list(choosed.items())):
        del choosed[k]
        choosed[f'{idx+1:02}, {k}'] = v
    yield choosed, llm_gen, choosed_sum_sim, choosed_con_sim, choosed_pref_sim


def apply_lycoris(text_model, lycoris_path, weight=1.0):
    lycoris_sd = torch.load(lycoris_path, map_location="cpu")
    lycoris_net, _ = create_lycoris_from_weights(1.0, "", text_model, lycoris_sd)
    lycoris_net.to(next(text_model.parameters()).dtype)
    lycoris_net.merge_to(weight)
    return text_model


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
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Apply xformers optimization
    try:
        apply_attn_algo(text_model, algo="xformers")
    except Exception as e:
        print(e)

    # Load LyCORIS model for LLM and apply it
    apply_lycoris(text_model, "./models/lycoris-weights/epoch=4.pt", 0.7)
    apply_lycoris(text_model, "./models/lycoris-weights/train_on_input/epoch=2.pt", 0.3)

    # Cast LLM to FP8 for efficiency
    text_model.half()
    # text_model.transformer.h.to(torch.float8_e4m3fn)
    # text_model.lm_head.to(torch.float8_e4m3fn)
    text_model.to(device)

    # Load Jina-Emb model and contrastive scorer model
    embed_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    ).cpu()
    scorer_model = ContrastiveScorer.load_from_checkpoint(
        "./models/SigLIP/ver1.ckpt"
    ).cpu()

    def wrapper(request, weight_sum_sim, weight_con_sim, weight_pref_sim, expand_input):
        with make_autocast():
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
                weight_sum_sim,
                weight_con_sim,
                weight_pref_sim,
                expand_input,
            )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                request = gr.TextArea(label="Input your request")
                weight_sum_sim = gr.Slider(
                    label="Weight of Summary Similarity",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                )
                weight_con_sim = gr.Slider(
                    label="Weight of Contrastive Score",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                )
                weight_pref_sim = gr.Slider(
                    label="Weight of Preference Similarity",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                )
                expand_input = gr.Checkbox(label="Expand Input")
                submit = gr.Button("Submit")
            with gr.Column(scale=2):
                label = gr.Label(label="Possible Courses")
        with gr.Accordion("addtional infos", open=False):
            with gr.Row():
                with gr.Column():
                    result = gr.TextArea(label="LLM output", lines=20)
                with gr.Column():
                    summary_sim = gr.Label(label="Summary Similarity")
            with gr.Row():
                with gr.Column():
                    contrastive_score = gr.Label(label="Contrastive Score")
                with gr.Column():
                    preference_sim = gr.Label(label="Preference Similarity")
        submit.click(
            wrapper,
            inputs=[
                request,
                weight_sum_sim,
                weight_con_sim,
                weight_pref_sim,
                expand_input,
            ],
            outputs=[label, result, summary_sim, contrastive_score, preference_sim],
        )

    demo.launch(server_name="127.0.0.1", server_port=17415, max_threads=2)
