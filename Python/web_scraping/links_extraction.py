import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def extract_links_and_save(url, filename="links.txt"):
    try:
        # Send an HTTP GET request to the given URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all anchor tags with an href attribute
            links = set()
            for a_tag in soup.find_all('a', href=True):
                # Get the full URL (handling relative links)
                full_url = urljoin(url, a_tag['href'])
                links.add(full_url)

            # Save the links to a file
            with open(filename, "w", encoding="utf-8") as file:
                for link in links:
                    file.write(link + "\n")

            print(f"All links from {url} have been saved to {filename}.")
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Use a hardcoded URL instead of input()
url = "https://www.dunkindonuts.com"
extract_links_and_save(url)
