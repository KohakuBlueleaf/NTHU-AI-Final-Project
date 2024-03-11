import asyncio
import json
import random

import openai
import tqdm


TEMPLATE = """Finish this json based on the information it have, this is a entry of a survey for students. 
All the content should be first person from student without the name of final course selection:

{{
    "final course selection": "{}",
    "studen preference on subject/content/direction/teaching style":"""
BEGIN_BLACK_LIST = [" ", ":", ",", ".", "Sorry", "Error"]
CONTENT_SHOULD_APPEAR = ["I", "my", "mine"]

retries = 5
client = openai.AsyncOpenAI(api_key="API_KEY")
info_list = json.load(open("info-with-preference.json", "r", encoding="utf-8"))
semaphore = asyncio.Semaphore(16)
pbar: tqdm.tqdm = None


def with_semaphore(semaphore=None):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def preference_check(preference):
    if preference is None:
        return False
    if any(preference.startswith(x) for x in BEGIN_BLACK_LIST):
        return False
    if not any(x in preference for x in CONTENT_SHOULD_APPEAR):
        return False
    return True


@with_semaphore(semaphore)
async def course_preference(course_info):
    result = []
    for idx, summary in enumerate(course_info["gpt3.5 summary"]):
        preference = course_info.get(
            "gpt3.5 preference", [None] * len(course_info["gpt3.5 summary"])
        )[idx]
        if preference_check(preference):
            result.append(
                preference.replace(
                    "Student preference on subject/content/direction/teaching style: ",
                    "",
                ).strip()
            )
            continue

        for i in range(retries):
            try:
                await asyncio.sleep(random.random() * 2)
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",  # Specify the model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": TEMPLATE.format(summary)},
                    ],
                    max_tokens=1024,
                    timeout=30,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                elif content.count('"') >= 1:
                    content = content.split('"')[-2]
                if preference_check(content):
                    result += [content]
                    break
            except Exception as e:
                # print(e)
                continue
        else:
            result += ["Error"]
    return result


async def gen_preference(course_info):
    course_info["gpt3.5 preference"] = await course_preference(course_info)

    with open("info-with-preference.json", "w", encoding="utf-8") as f:
        json.dump(info_list, f, ensure_ascii=False, indent=2)
    if pbar is not None:
        pbar.update(1)


async def main():
    global pbar
    tasks = [gen_preference(info) for info in info_list]
    pbar = tqdm.tqdm(total=len(tasks))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
