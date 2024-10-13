import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import re

def extract_links(filename):
    with open(filename, "r", encoding="utf-8") as file:
        links = [line.strip() for line in file.readlines()]
    return links

def scrape_data_from_link(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract data
            title = soup.title.string if soup.title else "No Title"
            paragraphs = soup.find_all('p')
            content = title + "\n\n" + "\n".join([p.get_text() for p in paragraphs[:2]])  # First two paragraphs

            return title, content
        else:
            print(f"Failed to retrieve {url}. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None, None

def sanitize_filename(title):
    # Remove invalid characters and strip leading/trailing whitespace
    valid_title = re.sub(r'[<>:"/\\|?*\n\r]', '', title).strip()
    return valid_title if valid_title else "scraped_page"

def save_data_to_pdf(content, title):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Encode to utf-8 and handle special characters
    pdf.multi_cell(0, 10, content.encode('latin-1', 'ignore').decode('latin-1'))
    pdf.ln()  # Line break

    # Sanitize the filename
    sanitized_title = sanitize_filename(title)
    pdf_filename = f"{sanitized_title}.pdf"
    pdf.output(pdf_filename)
    print(f"Data from {title} saved to {pdf_filename}")

def main():
    # Step 1: Extract links from the text file
    links = extract_links("links.txt")
    
    # Step 2: Scrape data from each link and save each one in a separate PDF file
    for link in links:
        print(f"Scraping data from: {link}")
        title, content = scrape_data_from_link(link)
        if content:
            save_data_to_pdf(content, title)

if __name__ == "__main__":
    main()
