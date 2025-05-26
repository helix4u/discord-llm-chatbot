import asyncio
import os
from playwright.async_api import async_playwright

async def open_twitter_login():
    """
    Launches a non-headless browser using a persistent context,
    navigates to Twitter/X login page, and waits for the user to close it.
    """
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir)
            print(f"Created profile directory: {user_data_dir}")
        except OSError as e:
            print(f"Could not create .pw-profile directory: {e}. Will use a temporary profile.")
            user_data_dir = None # Fallback to a non-persistent context

    print(f"Attempting to use profile directory: {user_data_dir or 'Temporary Profile'}")

    async with async_playwright() as p:
        try:
            if user_data_dir:
                # Using launch_persistent_context to save login state
                context = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False, # Makes the browser visible
                    args=["--disable-blink-features=AutomationControlled"],
                    slow_mo=50 # Slows down Playwright operations by 50ms to make it easier to see
                )
            else: # Fallback if profile directory creation failed
                browser = await p.chromium.launch(
                    headless=False,
                    args=["--disable-blink-features=AutomationControlled"],
                    slow_mo=50
                )
                context = await browser.new_context() # Standard non-persistent context

            page = await context.new_page()
            
            print("Navigating to Twitter/X login page...")
            await page.goto("https://twitter.com/login", wait_until="domcontentloaded", timeout=60000)
            
            print("\n-------------------------------------------------------------------------")
            print("Browser is open. Please log in to Twitter/X in this window.")
            print("Once you have logged in and the page has loaded, you can close the browser.")
            print("Your login session should be saved in the '.pw-profile' directory.")
            print("-------------------------------------------------------------------------")
            
            # Keep the script running until the browser is closed by the user
            # This is detected by waiting for the 'close' event on the context.
            await context.wait_for_event("close")
            print("Browser closed by user.")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # If browser was launched non-persistently, ensure it's closed
            if 'browser' in locals() and not user_data_dir:
                await browser.close()

if __name__ == "__main__":
    print("Starting script to open Twitter/X login...")
    asyncio.run(open_twitter_login())
    print("Script finished.")
