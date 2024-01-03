import asyncio
import json
import random

import openai
import tqdm


TEMPLATE = """Summarize the content of this course in English based on the informations provided.
You should list the informations with explanations or explain the content of the course with conformed descriptions.

------ Course information ------
Course Title: {}
Instructor: {}
Time: {}
Room: {}

Core capability to be cultivated by this course: {}

Brief description: {}

Syllabus: {}"""

retries = 5
client = openai.AsyncOpenAI(
    api_key="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"
)
info_list = json.load(open("info-with-preference.json", "r", encoding="utf-8"))
semaphore = asyncio.Semaphore(6)
pbar: tqdm.tqdm = None


async def summarize_course(course_info):
    for i in range(retries):
        try:
            await semaphore.acquire()
            await asyncio.sleep(random.random() * 5 * (i + 1))
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # Specify the model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for summarize existing data and generated synthetic data based on them.",
                    },
                    {
                        "role": "user",
                        "content": TEMPLATE.format(
                            course_info["course_title_en"],
                            course_info["instructor"],
                            course_info["time"],
                            course_info["room"],
                            course_info["core_cap"],
                            course_info["brief_desc"],
                            course_info["syllabus"],
                        ),
                    },
                ],
                n=5,
                max_tokens=1024,
                timeout=30,
            )
            semaphore.release()
            return [c.message.content for c in response.choices]
        except:
            return ["Error"]


async def gen_summary(course_info: dict):
    if course_info.get("gpt3.5 summary", ["Error"])[0] == "Error":
        course_info["gpt3.5 summary"] = await summarize_course(course_info)
    course_info.pop("gpt3.5 preference", None)
    with open("info-with-summary.json", "w", encoding="utf-8") as f:
        json.dump(info_list, f, ensure_ascii=False, indent=2)
    if pbar is not None:
        pbar.update(1)


async def main():
    global pbar
    tasks = [gen_summary(info) for info in info_list]
    pbar = tqdm.tqdm(total=len(info_list))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
