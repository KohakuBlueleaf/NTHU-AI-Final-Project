import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.nn.functional as F

from dataset import final
from transformers import AutoModel


def load_final_dataset(tokenizer=None, split="train"):
    dataset = final.load(split)["train"]
    return dataset


@torch.no_grad()
@torch.autocast("cuda")
def main():
    dataset = load_final_dataset(split="summary")
    summary = [data["summary"] for data in dataset]
    dataset = load_final_dataset(split="preference")
    preference = [data["preference"] for data in dataset]

    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    model = model.half().cuda()
    summary_embeddings = model.encode(summary)
    preference_embeddings = model.encode(preference)

    torch.save(summary_embeddings, "./models/summary-embeddings.pt")
    torch.save(preference_embeddings, "./models/preference-embeddings.pt")


if __name__ == "__main__":
    main()
