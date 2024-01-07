import json
import tqdm


with open("./info-with-preference-translated.json", "r", encoding="utf-8") as f:
    data = json.load(f)


new_data = []
for idx, info in tqdm.tqdm(enumerate(data)):
    for summary in info["gpt3.5 summary"][-1:]:
        for preference in (
            info["gpt3.5 preference"][:]
            # + info['nllb-200-distilled-600M translated zho_Hant preference'][-1:]
            # + info['nllb-200-distilled-600M translated zho_Hans preference'][-1:]
            # + info['nllb-200-distilled-600M translated jpn_Jpan preference'][-1:]
        ):
            new_entry = {
                "course_number": info["course_number"],
                "course_title": info["course_title"],
                "course_title_en": info["course_title_en"],
                "summary": summary,
                "preference": preference,
            }
            new_data.append(new_entry)

print(len(new_data))
with open("./preference-set-full-en.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
