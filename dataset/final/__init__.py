import os
from datasets import load_dataset


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    "all": ["summary-preference-set.json", "summary-preference-set-test.json"],
    "train": ["summary-preference-set.json"],
    "test": ["summary-preference-set-test.json"],
    "train-en": ["summary-preference-set-train-en.json"],
    "test-en": ["summary-preference-set-test-en.json"],
    "summary": ["summary-set-full-en.json"],
    "preference": ["preference-set-full-en.json"],
}


def load(split="all"):
    assert split in SPLITS
    return load_dataset(
        "json", data_files=[os.path.join(SELF_FOLDER, i) for i in SPLITS[split]]
    )


def generate_prompt(data_point):
    """Guanaco-alpaca chat format"""
    user_part = f"""### Instruct:
Use student's preference to predict the summary of final choosed course.

### Input:
{data_point['preference']}

"""

    output_part = f"""### Response:
{data_point["summary"]}"""

    return user_part, output_part


def tokenize(tokenizer, prompt, cutoff_len=2048, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        # return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def processor(tokenizer, cutoff_len=2048, train_on_inputs=False, padding=True):
    import torch

    def generate_and_tokenize_prompt(data_point):
        user_part, output_part = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(
            tokenizer, user_part + output_part, cutoff_len, add_eos_token=True
        )
        tokenized_user_prompt = tokenize(
            tokenizer, user_part, cutoff_len, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        full_prompt_len = len(tokenized_full_prompt["input_ids"])

        if not train_on_inputs:
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        pad_len = cutoff_len - full_prompt_len
        if padding:
            tokenized_full_prompt["input_ids"] = (
                tokenized_full_prompt["input_ids"] + [0] * pad_len
            )
            tokenized_full_prompt["labels"] = (
                tokenized_full_prompt["labels"] + [-100] * pad_len
            )
            tokenized_full_prompt["attention_mask"] = (
                tokenized_full_prompt["attention_mask"] + [0] * pad_len
            )

        for k in tokenized_full_prompt.keys():
            tokenized_full_prompt[k] = torch.LongTensor(tokenized_full_prompt[k])
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def contrastive_processor(emb_model):
    def pre_calc_emb(data_point):
        data_point["summary"] = emb_model.encode(data_point["summary"])
        data_point["preference"] = emb_model.encode(data_point["preference"])
        return data_point

    return pre_calc_emb
