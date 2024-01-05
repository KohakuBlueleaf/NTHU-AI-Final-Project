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
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        # Load LLM
        text_model: PhiForCausalLM = PhiForCausalLM.from_pretrained("microsoft/phi-2")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer.pad_token = tokenizer.eos_token
        # Apply xformers optimization
        apply_attn_algo(text_model, algo="xformers")

        # Load LyCORIS model for LLM and apply it
        lycoris_sd = torch.load(
            "./models/lycoris-weights/epoch=4.pt", map_location="cpu"
        )
        lycoris_net, _ = create_lycoris_from_weights(1.0, "", text_model, lycoris_sd)
        lycoris_net.to(next(text_model.parameters()).dtype)
        lycoris_net.merge_to(1.0)

        # Cast LLM to FP8 for efficiency
        text_model.transformer.h.to(torch.float8_e4m3fn)
        text_model.lm_head.to(torch.float8_e4m3fn)
        text_model.cuda()

        print()
        while (request := input("Input your request: ")) != "q":
            print("=" * 50, "\n")
            prev = ""
            prompt = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{request}"""
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
