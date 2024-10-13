import requests
from bs4 import BeautifulSoup

def scrape_and_save(url, filename="output.txt"):
    try:
        # Send an HTTP GET request to the given URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the main text content (all paragraphs as an example)
            paragraphs = soup.find_all('p')
            text_content = "\n".join([para.get_text(strip=True) for para in paragraphs])

            # Save the text content to a file
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text_content)

            print(f"Content from {url} has been saved to {filename}.")
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Provide the URL as a string
url = "https://www.dunkindonuts.com/en/sign-in.html"  # The URL should be in quotes
scrape_and_save(url)
