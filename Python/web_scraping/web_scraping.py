import requests
from bs4 import BeautifulSoup
import csv
import os
import webbrowser
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def scrape_dunkin_donuts():
    url = "https://www.dunkindonuts.com/en/menu"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all menu categories
    categories = soup.find_all('h2', class_='category-title')

    menu_items = []

    for category in categories:
        category_name = category.text.strip()
        items = category.find_next('ul', class_='item-list').find_all('li')

        for item in items:
            item_name = item.find('h3', class_='item-title').text.strip()
            item_description = item.find('p', class_='item-description')
            item_description = item_description.text.strip() if item_description else "No description available"

            menu_items.append({
                'Category': category_name,
                'Item': item_name,
                'Description': item_description
            })

    # Save to CSV
    with open('dunkin_donuts_menu.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Category', 'Item', 'Description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in menu_items:
            writer.writerow(item)

    print(f"Scraped {len(menu_items)} menu items and saved to dunkin_donuts_menu.csv")

def get_dunkin_donuts_html():
    url = "https://www.dunkindonuts.com/en/menu"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the HTML to a file
        html_file_path = os.path.abspath("dunkin_donuts.html")
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"HTML content has been saved to '{html_file_path}'")
        
        # Open the HTML file in Chrome
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
        webbrowser.get(chrome_path).open('file://' + html_file_path)
        
    except requests.RequestException as e:
        print(f"An error occurred while fetching the webpage: {e}")
    except IOError as e:
        print(f"An error occurred while saving or opening the file: {e}")
    except webbrowser.Error as e:
        print(f"An error occurred while trying to open Chrome: {e}")

def open_dunkin_donuts_in_chrome():
    url = "https://www.dunkindonuts.com/en/menu"
    
    try:
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--new-window")  # This opens in a new window, which effectively opens a new tab
        chrome_options.add_experimental_option("detach", True)  # This keeps the browser open
        
        # Set up the Chrome driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Open the URL
        driver.get(url)
        
        print(f"Opened {url} in Chrome.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    open_dunkin_donuts_in_chrome()
