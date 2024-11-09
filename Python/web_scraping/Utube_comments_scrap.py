import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_youtube_comments(video_url, output_csv, headless=True):
    # Set up Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Set up the Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the YouTube video
        driver.get(video_url)

        # Wait for the page to load
        time.sleep(5)

        # Scroll to load comments
        scroll_pause_time = 4  # Time to wait between scrolls
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        max_scroll_attempts = 60  # Increase the number of scrolls for better results
        comment_count = 0

        while True:
            # Scroll down and wait
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause_time)

            # Check for new height
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                # Break if no new comments are loaded
                break
            last_height = new_height

            # Find comment elements
            comments_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
            authors_elements = driver.find_elements(By.XPATH, '//*[@id="author-text"]')
            comment_count = len(comments_elements)

            # If a sufficient number of comments have been loaded, break
            if comment_count >= 100:  # Adjust this number based on your requirement
                break

        # Wait until comments are visible
        try:
            WebDriverWait(driver, 30).until(
                EC.visibility_of_element_located((By.XPATH, '//*[@id="content-text"]'))
            )
        except Exception as e:
            print(f"Error waiting for comments to become visible: {e}")

        # Re-fetch comments after scrolling
        comments_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
        authors_elements = driver.find_elements(By.XPATH, '//*[@id="author-text"]')

        if not comments_elements:
            print("No comments found.")
            return

        # Extract comment data
        comments_data = []
        for author, comment in zip(authors_elements, comments_elements):
            comments_data.append({
                "Author": author.text,
                "Comment": comment.text
            })

        # Save comments to a CSV file
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Author", "Comment"])
            writer.writeheader()
            writer.writerows(comments_data)

        print(f"Successfully scraped {len(comments_data)} comments and saved to {output_csv}.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    video_id = "YOUR_VIDEO_ID"  # Replace with the actual video ID
    video_url = "https://www.youtube.com/watch?v=90VvjgDXwdY"
    output_csv = "youtube_comments.csv"  # Specify the output CSV file name
    scrape_youtube_comments(video_url, output_csv, headless=False)  # Set headless=False for debugging
