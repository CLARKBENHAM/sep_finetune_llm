from bs4 import BeautifulSoup
import os
import re
import requests

url = "https://archiveofourown.org/works?commit=Sort+and+Filter&work_search%5Bsort_column%5D=kudos_count&include_work_search%5Brating_ids%5D%5B%5D=13&include_work_search%5Bfreeform_ids%5D%5B%5D=123409&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=10000&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=&tag_id=Sexual+Content"
# Generic explicit, any length https://archiveofourown.org/works/search?work_search%5Bquery%5D=&work_search%5Btitle%5D=&work_search%5Bcreators%5D=&work_search%5Brevised_at%5D=&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bword_count%5D=&work_search%5Blanguage_id%5D=&work_search%5Bfandom_names%5D=&work_search%5Brating_ids%5D=13&work_search%5Bcharacter_names%5D=&work_search%5Brelationship_names%5D=&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=&work_search%5Bcomments_count%5D=&work_search%5Bbookmarks_count%5D=&work_search%5Bsort_column%5D=hits&work_search%5Bsort_direction%5D=desc&commit=Search
# Sorting by hits may be better than kudos


# %%
def get_existing_pages(d="bad_text/top5_explicit_oao_10k_words/"):
    """
    Manually copied files in for first test
    Get Top 5 sexual fan fics from biggest fan fic site, less than 10k words: https://archiveofourown.org/works?commit=Sort+and+Filter&work_search%5Bsort_column%5D=kudos_count&include_work_search%5Brating_ids%5D%5B%5D=13&include_work_search%5Bfreeform_ids%5D%5B%5D=123409&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=10000&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=&tag_id=Sexual+Content
    """
    fic_pages = []
    for f in os.listdir(d):
        with open(f"{d}/{f}", "r") as f:
            fic_pages += [f.read()]
    return fic_pages


# %%


def scrape_ao3(url=url, num_stories_scrape=100, d="bad_text/top95_explicit_ao3_10k_words"):
    """Download the pages from ao3 into

    Args:
                    url (_type_): the url to start from. Make it match the desired search numers
                    num_stories_scrape (int, optional): Each page has 20 stories each. Defaults to 100
                    d2 (str, optional): _description_. Defaults to "bad_text/top95_explicit_ao3_10k_words".
    """
    fic_pages2 = []
    num_pages_scrape = num_stories_scrape // 20
    if not os.path.exists(d):
        print(f"{d} doesnt exist. Scraping data from {url}")
        os.mkdir(d)
        with requests.Session() as session:
            for pg_ix in range(1, 1 + num_pages_scrape):
                response = session.get(f"{url}&page={pg_ix}")
                response.raise_for_status()

                soup_search_page = BeautifulSoup(response.content, "html.parser")
                links = soup_search_page.find_all(
                    "a",
                    href=lambda href: href
                    and re.match("^/works/\d+$", href)
                    and href != "/works/2080878",  # "I am Groot" repeated 400 times
                )

                for s_ix, link in enumerate(links):
                    title = (
                        " ".join(re.findall("[a-zA-Z0-9\ \-\_]+", link.text))
                        .lower()
                        .replace(" ", "_")
                    )
                    assert len(title) > 2, title

                    story_r = session.get(
                        "https://archiveofourown.org"
                        + link.get("href")
                        + "?view_adult=true&view_full_work=true"
                    )
                    soup_story_page = BeautifulSoup(story_r.content, "html.parser")
                    soup_story_page.find_all("div.userstuff")
                    text_chunks = [
                        re.sub("(\n{2,}\s*|\s*\n{2,})", "\n\n", p.text).strip()
                        for i in soup_story_page.select("div.userstuff")
                        for p in i.select("p")
                    ]
                    n_story = "\n\n".join([t for t in text_chunks if t])
                    assert 100 <= n_story.count(" ") and n_story.count(" ") <= 10000, n_story.count(
                        " "
                    )
                    fic_pages2 += [n_story]
                    with open(f"{d}/pg{pg_ix}_ix{s_ix}_{title}.txt", "w") as f:
                        f.write(n_story)
    else:
        fic_pages2 = get_existing_pages(d=d)
    return fic_pages2
