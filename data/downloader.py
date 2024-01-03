import asyncio
import csv
import json


import requests
import httpx
from bs4 import BeautifulSoup as Soup
from tqdm import tqdm, trange


URL = "https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/JH/common/Syllabus/1.php?ACIXSTORE={}&c_key={}"
ACIXSTORE = "8ce10p8aad3comsjq7bpqj8a26"

client = httpx.AsyncClient(timeout=600)
pbar = tqdm(total=0)


def parse_information(soup: Soup):
    tds = soup.find_all("td")
    if len(tds) < 24:
        return
    else:
        return {
            "course_number": tds[2].text.strip(),
            "credit": tds[4].text.strip(),
            "class_size": tds[6].text.strip(),
            "course_title": tds[8].text.strip(),
            "course_title_en": tds[10].text.strip(),
            "instructor": tds[12].text.split("\n")[0].strip(),
            "time": tds[14].text.strip(),
            "room": tds[16].text.strip(),
            "core_cap": tds[19].text.strip(),
            "brief_desc": tds[21].text.strip(),
            "syllabus": tds[23].text.split("觀看上傳之檔案")[0].strip(),
        }


async def download_info(course_number):
    url = URL.format(ACIXSTORE, course_number)
    content = await client.get(url)
    pbar.update(1)
    return parse_information(Soup(content.content, "html.parser"))


info_keys = [
    "course_number",
    "credit",
    "class_size",
    "course_title",
    "course_title_en",
    "instructor",
    "time",
    "room",
    "core_cap",
    "brief_desc",
    "syllabus",
]


async def main():
    # open 'course_numbers.csv'
    with (
        open("course_numbers.csv", "r", encoding="utf-8") as f,
        open("info.json", "w", encoding="utf-8") as g,
    ):
        print(len(f.readlines()))
        f.seek(0)
        reader = csv.reader(f)
        next(reader)
        g.write("[\n")
        try:
            tasks = [download_info(course_number[0]) for course_number in reader]
            infos = await asyncio.gather(*tasks)
            for info in infos:
                if info is None:
                    continue
                g.write(json.dumps(info, ensure_ascii=False, indent=2) + ",\n")
        finally:
            g.seek(g.tell() - 3, 0)
            g.write("\n]")


if __name__ == "__main__":
    asyncio.run(main())
