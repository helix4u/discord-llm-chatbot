import re
import logging
import asyncio
import json
import subprocess # For curl fallback

import aiohttp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

# Attempt to import SEARX_URL from config. If not found, use a default or log a warning.
try:
    from config import SEARX_URL
except ImportError:
    SEARX_URL = None # Or a default like "http://localhost:8080"
    logging.warning("SEARX_URL not found in config.py. query_searx might not work as expected.")

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Text Cleaning and URL Detection ---
def detect_urls(message_text: str) -> list:
    # (Identical to the one in lmcordx.py)
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(message_text)

def clean_text(text: str) -> str:
    # (Identical to the one in lmcordx.py)
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\n+', '\n', text).strip() # Normalize newlines
    # Define patterns for removal or replacement
    patterns_to_replace = [
        (r'\s+', ' '),              # Normalize whitespace to single space
        (r'\[.*?\]', ''),          # Remove content in square brackets (e.g., [Music], [Applause])
        (r'\[\s*__\s*\]', ''),    # Remove placeholders like [ __ ]
        # Add specific known ad texts or boilerplate if necessary, e.g.:
        (r'NFL Sunday Ticket', ''), 
        (r'© \d{4} Google LLC', '') # Remove Google copyright notices
    ]
    for pattern, repl in patterns_to_replace:
        text = re.sub(pattern, repl, text)
    return text.strip() # Ensure leading/trailing whitespace is removed finally

def clean_youtube_transcript(transcript: str) -> str:
    # (Identical to the one in lmcordx.py, which just calls clean_text)
    return clean_text(transcript)

# --- Web Scraping Functions ---

async def scrape_website(url: str) -> str:
    """
    Scrapes a website using Playwright to get dynamic content.
    (This version is from lmcordx.py)
    """
    logger.info(f"Scraping website with Playwright: {url}")
    user_agent = ( # A common user agent
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    
    async with async_playwright() as p:
        # Consider making headless mode configurable via config.py if needed for debugging
        browser = await p.chromium.launch(headless=True) 
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        try:
            # Increased timeout, and wait until 'domcontentloaded'
            await page.goto(url, wait_until='domcontentloaded', timeout=20000) # Increased timeout
            # A short delay to allow for any JavaScript rendering after DOM load
            await asyncio.sleep(2) # Small delay for JS if any
            content = await page.evaluate('document.body.innerText')
            if not content or not content.strip():
                # Fallback to HTML if innerText is empty (e.g. for SPA that don't populate innerText well)
                logger.warning(f"No innerText found for {url}, trying HTML content.")
                content = await page.content() # Get full HTML
                # If using HTML, it's good to clean it with BeautifulSoup here or pass to clean_text
                # For now, clean_text will handle basic HTML tag removal
            
            await browser.close()
            cleaned_content = clean_text(content) # Clean the extracted content
            return cleaned_content if cleaned_content else "Failed to scrape the website (no content found after cleaning)."
        except PlaywrightTimeoutError:
            logger.error(f"Playwright timed out while scraping {url}")
            await browser.close()
            return "Failed to scrape the website (timeout)."
        except Exception as e:
            logger.error(f"Playwright failed for {url}: {e}")
            await browser.close()
            return "Failed to scrape the website (general error)."

def scrape_with_beautifulsoup(url: str) -> str:
    """
    Scrapes a website using requests and BeautifulSoup (from llmcord_search.py).
    This is a synchronous function.
    """
    logger.info(f"Attempting to scrape with BeautifulSoup: {url}")
    try:
        # Using aiohttp for async context if this were async, but it's sync
        # For sync, we'd use 'requests' library.
        # Since this file is primarily async, making this also async:
        # async def scrape_with_beautifulsoup_async(url: str) -> str:
        # However, to keep it as moved from llmcord_search.py (sync), we'd need requests.
        # Let's assume for now it should be async to fit with web_utils.py predominantly async nature.
        # If it MUST be sync, it might be better in a different utils file or clearly marked.
        # For this exercise, I'll make it async and use aiohttp.
        # --> Correction: The original was synchronous. For minimal change, it should remain sync.
        # This means it will block if called from an async event loop directly.
        # Consider using asyncio.to_thread if calling from async code.
        import requests # Synchronous, will block if called directly in async code.
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return clean_text(text)
    except requests.RequestException as e:
        logger.error(f"BeautifulSoup (requests) failed for {url}: {e}")
        return f"Failed to scrape with BeautifulSoup: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred with BeautifulSoup for {url}: {e}")
        return f"An unexpected error occurred during BeautifulSoup scraping: {e}"

def scrape_with_curl(url: str) -> str:
    """
    Scrapes a website using a curl command (from llmcord_search.py).
    This is a synchronous function.
    """
    logger.info(f"Attempting to scrape with curl: {url}")
    try:
        # Added -L to follow redirects, and a common user agent
        result = subprocess.run(
            ['curl', '-L', '-A', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", url],
            capture_output=True, text=True, check=True, timeout=15 # Added timeout
        )
        # Basic cleaning, assuming curl output might be HTML
        soup = BeautifulSoup(result.stdout, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return clean_text(text)
    except subprocess.CalledProcessError as e:
        logger.error(f"Curl command failed for {url}: {e.stderr}")
        return f"Failed to scrape with curl (process error): {e.stderr}"
    except subprocess.TimeoutExpired:
        logger.error(f"Curl command timed out for {url}")
        return "Failed to scrape with curl (timeout)."
    except Exception as e:
        logger.error(f"An unexpected error occurred with curl for {url}: {e}")
        return f"An unexpected error occurred during curl scraping: {e}"


# --- Searx/YouTube Transcript Functions ---

async def query_searx(query: str) -> list:
    # (Moved from lmcordx.py, uses config.SEARX_URL)
    logger.info(f"Querying Searx for: {query}")
    if not SEARX_URL:
        logger.error("SEARX_URL is not configured. Cannot query Searx.")
        return []

    # The very long 'preferences_str' was part of the original function.
    # It's generally better to configure preferences on the SearxNG instance itself,
    # or make this string configurable via config.py if it must be client-side.
    # For now, keeping it as it was in lmcordx.py for direct porting.
    # Consider moving to config.py if static: config.SEARX_PREFERENCES
    preferences_str = (
        "eJx1V0GT6yYM_jXNxfMyfX2HTg85dabXdqa9MzIottaAWMBJvL--IrZjvN53WGKEEOKT9KHVkLHjSJguHXqMYH_57U-P9yQ_lGVImtBrLF-"
        "sCWzj0BCcLPhuhA4vMGY-WdZg8YL-VKaaXbCY8XIiJyoqRH5Ml7_AJjw5zD2byz9___vfKcEVE0LU_eXXU-7R4SVR2XqKmEabk2KvxBeVoV22"
        "GyYli2xvGC8MMj1z7E7zNpXyZBePNPqMUYGlzjv5XvaDuYHcxqjl3Fn6PmKcFHmVKYuBWUj-Sp6yWNWRrV0doAStFQPoO_KC2h8ddErV2AhU4"
        "D2kppxAN1TqShYLoBCGxlGMHGuZeN7I2KTMsVaedzeUlNpC05Lv9vPcjnrArNQzXFrrb_mm1I0McjEjzqcU8So-viIpsru5kSBbWZow1FM9Rk"
        "tYSwzih0Cq3JhIl7npGoNPkIj9zkuUA8goxRLVWFTFxfLXcfNMCVGef_drd4SyYds3g-MgyAYZi2XHbxQKzJtWlkBMO1e_P6oDriZycWYF9h"
        "oRm8TXfIeIjaGIWpCfFgCvkfxAoCsDnWQFtCvAbLDF2C3TjrkTF4OFqQQxbcfUK44F7VRFpQtsTA1mD22EMix2yZl2uyF5qDaTl0_iMX0tW0"
        "2-Uep5szGQHiBVDlqSI-PUFNgS1QvcpoznmBZf4MNOkXRlWqoYdAC_KPjg1q8JYDPEAX3EwJXtAOJFR2nN1zC2Z4O3ZTZXZYG4KcMrJOCgeI"
        "svNQc-k26S7tlCrHM7ZYg5FNqp8iHzMHFmwWMoTq_O5DvlwhKfy3eCnrnef6d2qudCdg_wJpYsrLUGapmH9FlY-FSphVWL4H3kjJ-1Eo9RH6"
        "RCdInydBDz9OmO33_8-P2x3c2MBv0W_YQfHlytL4WEONQSD7cCxiaIYzt16NY8CIgxjy1WafdETI4YCpvcsa2W4uiEKCvBnR80sJdwN2ny7"
        "CeHlX9v4RzulcPlkk64ZUd2EW67OwfJQzsnzU_ER76RRNpSe8a8H_NWzi9DJ_R7qn_as-THR1O8W6sjSiZW9iE-6FZnZCuZosGFrXzK7oXD9"
        "sx-9PYp3eXPUzKDWoHbZkNdty-9kjpY2RJKj-LTtCf1G0ktySVqSmZ5UWLTjyvpFYcN5J2_G2_vmD9Pjr2AVMX2aoV6Yk2o6KlmKpnnAvwc"
        "AsrbwTOJ1gcstHoAapHvoFpkB7AW-Rfs0S_nPQ__-LZQ5JEz-SoE5TtpRfap5bvbDiRHD215NBV1gjwvuNJ8CRPIC1TATSgdT-XLuiYUKQD"
        "C8sju11OWxyzL07i-j8GU-tyUQi-851-rFKXfa6G6kViXIt5tmcKa2-93yY36Pk_BHuRZdIjHLD5AL1c1lA-MmzAQrIX8UpbqlNLZ4yf8rg"
        "eW1L5avq8lmIaxHX0e16oeA8YxvVCWfpGMvPGS_bl6i0af5HVOfeX1k_n395t4_ER6L8mrEwKykvYlQpXajRxyzX80FI1nPr1Yju38uoW-"
        "amgM6fzBfpf3TujSSZvS5Ag-WQmjqc1E42moBDnHM610unWtwY7CZelSoH6cl9m555TldUBprQWQ-RndKTCoufO-R-mID8sCrdI96uGwIhg"
        "oycABp_Si05_4UAIxTw5WlsQQdPX878ok_buVtu0LTXuVRv7KRx8jFFJT0tbKPVzJoJO0BVJCl_8BGa3pmQ==&q=%s"
    )

    params = {
        "q": query,
        "preferences": preferences_str % query, # Original had %s in preferences_str, assuming query was intended
        "format": "json",
        "language": "en-US", # Could be from config
    }
    try:
        async with aiohttp.ClientSession() as session:
            # Ensure SEARX_URL does not end with /search if it's part of the base URL
            search_endpoint = SEARX_URL.rstrip('/') + "/search" if not SEARX_URL.endswith("/search") else SEARX_URL
            async with session.get(search_endpoint, params=params, timeout=15) as response: # Timeout could be config
                if response.status == 200:
                    results_json = await response.json()
                    return results_json.get("results", [])[:5] # Return top 5 results
                else:
                    logger.error(f"Failed to fetch data from Searx ({search_endpoint}). Status: {response.status}, Response: {await response.text()}")
                    return []
    except aiohttp.ClientError as e:
        logger.error(f"An error occurred while fetching data from Searx: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in query_searx: {e}")
        return []


async def fetch_youtube_transcript(url: str) -> str:
    # (Moved from lmcordx.py)
    logger.info(f"Fetching YouTube transcript for: {url}")
    try:
        video_id_match = re.search(r'v=([^&]+)', url)
        if not video_id_match:
            video_id_match = re.search(r'youtu\.be/([^?]+)', url)
        
        if not video_id_match:
            logger.warning(f"Could not extract video ID from URL: {url}")
            return ""
            
        video_id = video_id_match.group(1)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        return clean_youtube_transcript(transcript_text) # Use the cleaned version
    except NoTranscriptFound:
        logger.warning(f"No transcript found for YouTube video: {url}")
        return "" # Return empty string if no transcript is found
    except Exception as e:
        logger.error(f"Failed to fetch YouTube transcript for {url}: {e}")
        return "" # Return empty string on other errors
