########################################################################################################################
########################################################################################################################
#### SCRAPE GLASSDOOR.COM TO RECEIVE THE
########################################################################################################################
########################################################################################################################
#### AUTHOR: Peter Schaefer
#### THIS VERSION: 2024, July 22nd
#### OVERVIEW OF STEPS:
#### This code scrapes Glassdoor.com for employee reviews and saves them to json files
#### It tracks the average time it takes per Glassdoor page


########################################################################################################################
### STEP 0: IMPORT PACKAGES AND DEFINE PARAMETERS AND FILE DIRECTORIES
########################################################################################################################
import asyncio
import json
import re
import os
import tempfile
import time
from typing import Dict, Optional
from scrapfly import ScrapeApiResponse, ScrapeConfig, ScrapflyClient

client = ScrapflyClient(key="scp-live-799547289b9247f39e40ba4b6b9b3417")

# Setting up relative file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
link_file_path = os.path.join(base_dir, '01_raw_data', 'list_of_links_to_do.txt')
name_file_path = os.path.join(base_dir, '01_raw_data', 'list_of_companies_to_do.txt')
checkpoint_file_path = os.path.join(base_dir, '02_temp_files', 'checkpoint.json')
error_log_path = os.path.join(base_dir, '02_temp_files', 'error_log.txt')
#output_dir = os.path.join(base_dir, '03_scraped_gd_reviews')
output_dir = os.path.join(base_dir)
os.makedirs(output_dir, exist_ok=True)

MAX_PAGES_TO_SCRAPE = 50

BASE_CONFIG = {
    "country": "US",
    "asp": True,
    "cookies": {"tldp": "1"},
    #"proxy_pool": "PUBLIC_RESIDENTIAL_POOL"
}

def find_hidden_data(result: ScrapeApiResponse) -> dict:
    """
    Extract hidden web cache (Apollo Graphql framework) from Glassdoor page HTML.

    Args:
        result (ScrapeApiResponse): The response from the Scrapfly API.

    Returns:
        dict: The unpacked Apollo GraphQL data.
    """
    data = result.selector.css("script#__NEXT_DATA__::text").get()
    if data:
        data = json.loads(data)["props"]["pageProps"]["apolloCache"]
    else:
        matches = re.findall(r'apolloState":\s*({.+})};', result.content)
        if not matches:
            log_error(f"No apolloState data found in the page content for URL: {result.context['url']}")
            return {}
        data = json.loads(matches[0])

    def _unpack_apollo_data(apollo_data):
        """
        Unpack __ref references to actual values.

        Args:
            apollo_data (dict): The Apollo GraphQL data.

        Returns:
            dict: The resolved Apollo GraphQL data.
        """
        def resolve_refs(data, root):
            if isinstance(data, dict):
                if "__ref" in data:
                    return resolve_refs(root[data["__ref"]], root)
                else:
                    return {k: resolve_refs(v, root) for k, v in data.items()}
            elif isinstance(data, list):
                return [resolve_refs(i, root) for i in data]
            else:
                return data

        return resolve_refs(apollo_data.get("ROOT_QUERY") or apollo_data, apollo_data)

    return _unpack_apollo_data(data)

def log_error(message: str):
    """
    Log an error message to the error log file.

    Args:
        message (str): The error message to log.
    """
    with open(error_log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def change_page(url: str, page: int) -> str:
    """
    Update the page number in a Glassdoor URL.

    Args:
        url (str): The original URL.
        page (int): The new page number.

    Returns:
        str: The updated URL with the new page number.
    """
    if re.search(r"_P\d+\.htm", url):
        new = re.sub(r"(?:_P\d+)*.htm", f"_P{page}.htm", url)
    else:
        new = re.sub(".htm", f"_P{page}.htm", url)
    assert new != url
    return new

def parse_reviews(result: ScrapeApiResponse) -> Dict:
    """
    Parse Glassdoor reviews page for review data.

    Args:
        result (ScrapeApiResponse): The response from the Scrapfly API.

    Returns:
        Dict: The parsed reviews.
    """
    cache = find_hidden_data(result)
    if not cache:
        return {"reviews": []}
    reviews = next((v for k, v in cache.items() if k.startswith("employerReviews") and v.get("reviews")), None)
    if reviews is None:
        log_error(f"No reviews found in the apolloState data for URL: {result.context['url']}")
        return {"reviews": []}
    return reviews

def save_checkpoint(company: str, page: int, reviews: Dict):
    """
    Save the progress checkpoint using atomic write.

    Args:
        company (str): The company name.
        page (int): The current page number.
        reviews (Dict): The accumulated reviews.
    """
    checkpoint = {
        "company": company,
        "page": page,
        "reviews": reviews
    }
    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
        json.dump(checkpoint, temp_file, ensure_ascii=False, indent=4)
    os.replace(temp_file.name, checkpoint_file_path)

def load_checkpoint() -> Optional[Dict]:
    """
    Load the progress checkpoint if it exists, with error handling.

    Returns:
        Optional[Dict]: The checkpoint data if it exists, otherwise None.
    """
    if os.path.exists(checkpoint_file_path):
        try:
            with open(checkpoint_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log_error(f"Error loading checkpoint: {e}")
            return None
    return None

async def scrape_reviews(url: str, company: str, start_page: int = 1, max_pages: Optional[int] = None) -> Dict:
    """
    Scrape Glassdoor reviews listings from the reviews page (with pagination).

    Args:
        url (str): The URL of the Glassdoor reviews page.
        company (str): The company name.
        start_page (int): The starting page number.
        max_pages (Optional[int]): The maximum number of pages to scrape.

    Returns:
        Dict: The scraped reviews.
    """
    reviews = {"reviews": []}
    if start_page == 1:
        first_page = await client.async_scrape(ScrapeConfig(url=url, **BASE_CONFIG))
        page_reviews = parse_reviews(first_page)
        reviews["reviews"].extend(page_reviews["reviews"])
        total_pages = page_reviews.get("numberOfPages", 1)
    else:
        first_page = await client.async_scrape(ScrapeConfig(url=change_page(url, start_page),  **BASE_CONFIG))
        page_reviews = parse_reviews(first_page)
        reviews["reviews"].extend(page_reviews["reviews"])
        total_pages = page_reviews.get("numberOfPages", 1)

    if max_pages and max_pages < total_pages:
        total_pages = max_pages

    start_time = time.time()
    page_times = []

    other_pages = [
        ScrapeConfig(url=change_page(url, page=page), **BASE_CONFIG)
        for page in range(start_page + 1, total_pages + 1)
    ]

    for page in range(start_page + 1, total_pages + 1):
        page_start_time = time.time()
        async for result in client.concurrent_scrape([ScrapeConfig(url=change_page(url, page=page), **BASE_CONFIG)]):
            page_reviews = parse_reviews(result)
            reviews["reviews"].extend(page_reviews["reviews"])
            save_checkpoint(company, page, reviews)
            print(f"Successfully scraped page {page} for {company}")

            # Save the intermediate results to JSON with UTF-8 encoding and atomic write
            new_file_name = os.path.join(output_dir, f"{company}.json")
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
                json.dump(reviews, temp_file, ensure_ascii=False, indent=4)
            os.replace(temp_file.name, new_file_name)

            page_end_time = time.time()
            page_times.append(page_end_time - page_start_time)
            average_time = sum(page_times) / len(page_times)
            print(f"Average time per page so far: {average_time:.2f} seconds")

    return reviews

async def run():
    """
    Main function to run the scraper. Loads checkpoint if exists, iterates through companies, and scrapes reviews.
    """
    # Load the checkpoint if it exists
    checkpoint = load_checkpoint()

    # Open both files in read mode
    with open(link_file_path, 'r', encoding='utf-8', errors='ignore') as link_file, open(name_file_path, 'r', encoding='utf-8', errors='ignore') as name_file:
        for link_line, name_line in zip(link_file, name_file):
            stripped_name = name_line.strip()
            gd_url = link_line.strip()
            start_page = 1
            reviews = {"reviews": []}

            # If there is a checkpoint, resume from there
            if checkpoint and checkpoint["company"] == stripped_name:
                start_page = checkpoint["page"] + 1
                reviews = checkpoint["reviews"]

            glassdoor_reviews = await scrape_reviews(
                url=gd_url,
                company=stripped_name,
                start_page=start_page,
                max_pages=MAX_PAGES_TO_SCRAPE
            )

            # Save the final results to JSON with UTF-8 encoding and atomic write
            new_file_name = os.path.join(output_dir, f"{stripped_name}.json")
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
                json.dump(glassdoor_reviews, temp_file, ensure_ascii=False, indent=4)
            os.replace(temp_file.name, new_file_name)

            # Clear the checkpoint after successfully scraping the company
            if os.path.exists(checkpoint_file_path):
                os.remove(checkpoint_file_path)

if __name__ == "__main__":
    asyncio.run(run())
