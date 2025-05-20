import logging
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Configure logger for this module
logger = logging.getLogger(__name__)

async def scrape_latest_tweets(username: str, limit: int = 10) -> list:
    """
    Scrapes the latest tweets from a given X/Twitter user handle.
    (Moved from lmcordx.py)
    """
    logger.info(f"[SCRAPE] Attempting to fetch last {limit} tweets from @{username}")
    tweets_collected = []
    seen_timestamps = set() # To avoid duplicates if tweets are pinned or UI loads them multiple times
    
    # Configuration for scrolling and delays (could be made configurable if needed)
    scroll_delay = 2.0  # Time in seconds to wait after each scroll
    # More scrolls for more tweets, but also diminishing returns and higher chance of rate limits/blocks
    total_scrolls = max(5, limit // 2 + 3) # Heuristic for number of scrolls

    # Playwright browser launch options
    # user_data_dir allows for persistent login session, reducing need to log in every time.
    # headless=False is useful for initial setup/debugging to see what the browser is doing.
    # For production/unattended runs, headless=True is generally preferred.
    # args are to help avoid bot detection.
    browser_args = ["--disable-blink-features=AutomationControlled"] #, "--start-maximized"] # Maximize can help some sites
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=".pw-profile-twitter", # Specific profile for Twitter
                headless=True,  # Set to False for debugging login/captcha issues
                args=browser_args,
                # slow_mo=50, # Slow down operations by 50ms for debugging if needed
                # viewport={'width': 1920, 'height': 1080} # Set viewport
            )
        except Exception as e:
            logger.error(f"Failed to launch Playwright browser context: {e}")
            return [] # Cannot proceed if browser doesn't launch

        page = await browser.new_page()
        url = f"https://x.com/{username}/with_replies" # Or just /username for tweets without replies in main feed
        logger.debug(f"Navigating to Twitter URL: {url}")

        try:
            await page.goto(url, timeout=60000, wait_until="domcontentloaded") # wait_until can be 'load' or 'networkidle' too
            # Wait for a selector that indicates tweets are loaded, e.g., 'article' or a specific tweet structure
            # This helps ensure the page is ready before we start scrolling.
            await page.wait_for_selector("article", timeout=30000) 
            logger.debug(f"Successfully navigated to {url} and found tweet articles.")
        except PlaywrightTimeoutError:
            logger.error(f"Timeout navigating to {url} or finding initial tweets.")
            await browser.close()
            return []
        except Exception as e:
            logger.error(f"Error during initial navigation to {url}: {e}")
            await browser.close()
            return []

        scrolls_done = 0
        consecutive_no_new_tweets = 0

        while scrolls_done < total_scrolls and len(tweets_collected) < limit:
            previous_tweet_count = len(tweets_collected)
            
            await page.mouse.wheel(0, 3000) # Scroll down
            await asyncio.sleep(scroll_delay) # Wait for content to load
            scrolls_done += 1
            logger.debug(f"Scroll {scrolls_done}/{total_scrolls}, {len(tweets_collected)} tweets collected so far.")

            # Locate tweet articles. This selector might need adjustment if Twitter changes its structure.
            articles = await page.locator("article:has(div[lang])").all() 
            
            if not articles and scrolls_done == 1: # No articles on first scroll could mean profile error or immediate block
                logger.warning(f"No tweet articles found on the page for @{username} after first scroll.")
                # Could try a page reload here, or just give up.
                # await page.reload(wait_until="domcontentloaded")
                # await asyncio.sleep(scroll_delay)
                # articles = await page.locator("article:has(div[lang])").all()
                # if not articles:
                #     logger.error(f"Still no articles after reload for @{username}.")
                #     break # Exit scroll loop

            for art_idx, art in enumerate(articles):
                try:
                    timestamp_elem = art.locator("time").nth(0)
                    ts_attr = await timestamp_elem.get_attribute("datetime")
                    if not ts_attr or ts_attr in seen_timestamps:
                        continue # Skip already processed or invalid timestamp tweets

                    # Extract user handle (useful for quote tweets or if not on main profile page)
                    user_handle = username # Default to the target username
                    try:
                        # This selector attempts to find the user handle within the tweet structure
                        # It might need updates if Twitter's HTML changes.
                        anchor = art.locator("div[data-testid='User-Name'] a[href*='/status/']").first
                        href_parts = (await anchor.get_attribute("href") or "").strip("/").split("/")
                        if len(href_parts) > 1 and href_parts[1] == "status": # Format like /ActualUser/status/tweet_id
                            user_handle = href_parts[0]
                    except Exception: # Broad except as this is supplementary info
                        # logger.debug(f"Could not extract specific user handle for a tweet, defaulting to @{username}")
                        pass
                    
                    text_locator = art.locator("div[lang]").nth(0) # Main tweet text
                    content = (await text_locator.inner_text(timeout=5000)).strip() # 5s timeout for text extraction

                    if not content: # Skip tweets with no text content (e.g., just images/videos without text)
                        # logger.debug(f"Skipping tweet with no text content (ts: {ts_attr}).")
                        continue
                    
                    tweets_collected.append({
                        "timestamp": ts_attr,
                        "from_user": user_handle, # This will be the user whose tweet it is
                        "content": content
                    })
                    seen_timestamps.add(ts_attr)
                    
                    if len(tweets_collected) >= limit:
                        break # Reached desired number of tweets
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout extracting text from an article for @{username}. Skipping this article.")
                except Exception as e:
                    # Log specific error for one article but continue with others
                    logger.error(f"Error processing one tweet article for @{username} (idx {art_idx}): {e}")
                    continue 
            
            if len(tweets_collected) >= limit:
                logger.info(f"Collected {limit} tweets for @{username}. Stopping scroll.")
                break
            
            if len(tweets_collected) == previous_tweet_count:
                consecutive_no_new_tweets += 1
            else:
                consecutive_no_new_tweets = 0
            
            if consecutive_no_new_tweets >= 3: # If 3 consecutive scrolls yield no new tweets, likely end of feed or issue
                logger.info(f"No new tweets found after {consecutive_no_new_tweets} consecutive scrolls for @{username}. Assuming end of feed or issue.")
                break
        
        await browser.close()
        logger.debug("Playwright browser closed.")

    # Sort tweets by timestamp descending (newest first) before returning the limited list
    tweets_collected.sort(key=lambda x: x["timestamp"], reverse=True)
    final_tweets = tweets_collected[:limit]
    logger.info(f"Successfully scraped {len(final_tweets)} tweets for @{username}.")
    return final_tweets

# Example usage (for testing this module directly)
# if __name__ == '__main__':
#     async def main_test():
#         # Configure logging for testing
#         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         
#         test_username = "elonmusk"  # Replace with a public Twitter handle for testing
#         number_of_tweets = 5
#         logger.info(f"Testing scrape_latest_tweets with username: {test_username}, limit: {number_of_tweets}")
#         
#         # You might need to log in manually the first time if headless=False and no persistent context is saved yet.
#         # For persistent context to work, ensure the directory ".pw-profile-twitter" is writable.
#         
#         tweets = await scrape_latest_tweets(test_username, limit=number_of_tweets)
#         if tweets:
#             logger.info(f"Collected {len(tweets)} tweets:")
#             for i, tweet in enumerate(tweets):
#                 logger.info(f"Tweet {i+1}: [{tweet['timestamp']}] @{tweet['from_user']}: {tweet['content'][:100]}...")
#         else:
#             logger.warning("No tweets collected from test run.")
#
#     asyncio.run(main_test())
