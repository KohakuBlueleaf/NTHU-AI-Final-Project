import os
from itertools import chain

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

torch.set_float32_matmul_precision("medium")
from transformers import AutoModel

from dataset import final
from utils import slicepart


def load_final_dataset(tokenizer=None, split="train"):
    dataset = final.load(split)["train"]
    return dataset


@torch.no_grad()
@torch.autocast("cuda")
def main():
    # We only precompute the train set.
    # to ensure the test set be isolated
    dataset = load_final_dataset(split="summary")
    summary = [data["summary"] for data in dataset]
    summary = slicepart(summary, 5, slice(None, 4))
    dataset = load_final_dataset(split="preference")
    preference = [data["preference"] for data in dataset]
    preference = slicepart(preference, 5, slice(None, 4))
    print(len(summary), len(preference))

    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    model = model.half().cuda()
    summary_embeddings = model.encode(summary)
    preference_embeddings = model.encode(preference)

    torch.save(summary_embeddings, "./models/summary-embeddings-train-only.pt")
    torch.save(preference_embeddings, "./models/preference-embeddings-train-only.pt")


if __name__ == "__main__":
    main()
