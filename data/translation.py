import json

import torch
import tqdm

from transformers import M2M100ForConditionalGeneration, AutoTokenizer


model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)
model: M2M100ForConditionalGeneration = model.half()
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "zho_Hant"
target_lengths = ["zho_Hant", "zho_Hans", "jpn_Jpan"]


@torch.no_grad()
@torch.autocast("cuda")
def translate(text, target_lang="zho_Hant"):
    inputs = tokenizer(text, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].cuda()
    inputs["attention_mask"] = inputs["attention_mask"].cuda()
    outputs = model.cuda().generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=1024,
    )
    model.cpu()
    torch.cuda.empty_cache()
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def main():
    preferences = json.load(open("./info-with-preference.json", "r", encoding="utf-8"))
    for target_lang in tqdm.tqdm(target_lengths, smoothing=0.9, desc="language"):
        for data in tqdm.tqdm(preferences, smoothing=0.9, desc="preference"):
            data[f"nllb-200-distilled-600M translated {target_lang} preference"] = [
                translate(preference, target_lang)
                for preference in data["gpt3.5 preference"]
            ]

            with open(
                f"./info-with-preference-translated.json", "w", encoding="utf-8"
            ) as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
