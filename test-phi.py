from time import time_ns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers import AutoTokenizer, AutoModel, GenerationConfig
from tqdm import tqdm

from hakuphi.model import PhiForCausalLM
from hakuphi.attn_patcher import apply_attn_algo
from hakuphi.inference import generate

from lycoris.wrapper import create_lycoris_from_weights
from dataset import final
from data.translation import translate


def load_final_dataset(split="train"):
    dataset = final.load(split)["train"]
    return dataset


if __name__ == "__main__":
    model: PhiForCausalLM = PhiForCausalLM.from_pretrained("microsoft/phi-2")
    model = model.half()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        lycoris_sd = torch.load(
            "./models/lycoris-weights/epoch=4.pt", map_location="cpu"
        )
        model_sd = {}
        for k in model.state_dict():
            if k in lycoris_sd:
                model_sd[k] = lycoris_sd.pop(k)
        model.load_state_dict(model_sd, strict=False)
        lycoris_net, _ = create_lycoris_from_weights(1.0, "", model, lycoris_sd)
        lycoris_net.half()
        lycoris_net.merge_to(0.9)
        model.cuda()
        # model = torch.compile(model, mode='reduce-overhead')
        # lycoris_net.cuda()

        apply_attn_algo(model, algo="xformers")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer.pad_token = tokenizer.eos_token
        generate(model=model, tokenizer=tokenizer, prompt="Hello", max_new_tokens=16)

        dataset = load_final_dataset(split="summary")
        course_num = [data["course_number"] for data in dataset]
        course_name = [data["course_title"] for data in dataset]
        course_name_en = [data["course_title_en"] for data in dataset]
        pre_calc_sum_embed = torch.tensor(torch.load("./models/summary-embeddings.pt"))
        pre_calc_pref_embed = torch.tensor(
            torch.load("./models/preference-embeddings.pt")
        )
        embed_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        embed_model = embed_model.half().cuda()

        print()
        while (request := input("Input your request: ")) != "q":
            print("=" * 50, "\n")
            prev = ""
            prompt = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{request}

### Response:
"""
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
                    max_new_tokens=1024,
                    stream_output=True,
                ),
                disable=True,
            ):
                if len(i) > len(prev):
                    new_len = len(i) - len(prev)
                    print(i[len(prev) :], end="", flush=True)
                    prev = i
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
            result_embed = embed_model.encode([result])
            request_embed = embed_model.encode([translate(request, "eng_Latn")])
            preference_sim = torch.cosine_similarity(
                pre_calc_pref_embed, torch.tensor(request_embed)
            )
            summary_sim = torch.cosine_similarity(
                pre_calc_sum_embed, torch.tensor(result_embed)
            )
            sim_topk = torch.topk(preference_sim * 0.3 + summary_sim, 50)
            print("Top 10 similar courses: ")
            choosed = {}
            for idx in sim_topk.indices:
                if len(choosed) >= 10:
                    break
                if course_num[idx] in choosed:
                    continue
                choosed[course_num[idx]] = True
                print(
                    f"Summary Similarity: {summary_sim[idx].item():.3f}, "
                    f"Preference Similarity: {preference_sim[idx].item():.3f}"
                )
                print(course_num[idx], course_name[idx], course_name_en[idx])
                print()
