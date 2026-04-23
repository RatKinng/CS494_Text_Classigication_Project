import requests
import json
import time
import re
from bs4 import BeautifulSoup

def fetch_posts(blog_name, num_posts=50):
    url = f"https://{blog_name}.tumblr.com/api/read/json?num={num_posts}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed: {blog_name}")
        return []

    text = response.text
    json_str = text.replace('var tumblr_api_read = ', '')[:-2]
    data = json.loads(json_str)

    posts_data = []

    for post in data.get("posts", []):
        content = ""

        if "regular-body" in post:
            content = post["regular-body"]
        elif "photo-caption" in post:
            content = post["photo-caption"]
        elif "quote-text" in post:
            content = post["quote-text"]
        elif "link-description" in post:
            content = post["link-description"]

        if content:
            posts_data.append(content)

    return posts_data


def clean_text(html_text):
    if not html_text:
        return None

    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\b(via|source)\b.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 30:
        return None

    return text


# ------------------------
# MAIN SCRIPT
# ------------------------

blogs = ["staff",
    # "i-am-a-fish",
    # "just-shower-thoughts",
    # "topherchris",
    # "givethispromptatry",
    # "dailystoryprompts",
    # "here-haveaprompt",
    # "dark-fiction-and-angst",
    # "youneedsomeprompts",
    # "poetryisnotdead",
    ]

raw_count = 0
clean_count = 0
dataset = []

for blog in blogs:
    print(f"Scraping {blog}...")
    posts = fetch_posts(blog, num_posts=50)

    raw_count += len(posts)

    for post in posts:
        cleaned = clean_text(post)
        if cleaned:
            dataset.append([cleaned, "tumblr"])
            clean_count += 1

    time.sleep(2)


# Save dataset
import csv
with open("tumblr_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(dataset)


# ✅ FINAL COUNTS
print("\n--- SUMMARY ---")
print(f"Raw posts collected: {raw_count}")
print(f"Clean posts kept: {clean_count}")
print(f"Removed during cleaning: {raw_count - clean_count}")