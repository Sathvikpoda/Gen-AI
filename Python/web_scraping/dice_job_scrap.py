from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time

def scrape_dice_jobs(role, csv_filename):
    # Set up Selenium Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=options)

    # Construct the URL for the specified role
    url = f"https://www.dice.com/jobs?q={role.replace(' ', '+')}&countryCode=US"
    driver.get(url)

    # Open a CSV file for writing
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header row if the file is empty
        if file.tell() == 0:
            writer.writerow(['Job Title', 'Company', 'Location', 'Date Posted', 'Job Link'])

        page = 1
        while True:
            print(f"Scraping page {page} for role: {role}...")
            try:
                # Wait for job listings to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'card'))  # Adjust if necessary
                )

                # Get the page source and parse it with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                job_listings = soup.find_all('div', class_='card')  # Adjust if necessary

                if not job_listings:
                    print("No more job listings found.")
                    break

                # Loop through the job listings and extract data
                for index, job in enumerate(job_listings):
                    try:
                        job_title_elem = job.find('a', class_='card-title-link')  # Update if necessary
                        job_title = job_title_elem.text.strip() if job_title_elem else "N/A"
                        job_link = job_title_elem['href'] if job_title_elem and 'href' in job_title_elem.attrs else "N/A"

                        company_elem = job.find('a', class_='card-company-link')  # Update if necessary
                        company = company_elem.text.strip() if company_elem else "N/A"

                        location_elem = job.find('span', class_='card-location')  # Update if necessary
                        location = location_elem.text.strip() if location_elem else "N/A"

                        date_posted_elem = job.find('span', class_='posted-date')  # Update if necessary
                        date_posted = date_posted_elem.text.strip() if date_posted_elem else "N/A"

                        # Debug output for checking what was found
                        print(f"Job {index + 1 + (page - 1) * 20}: {job_title} | {company} | {location} | {date_posted} | {job_link}")

                        # Write the job details to the CSV file
                        writer.writerow([job_title, company, location, date_posted, job_link])

                    except Exception as e:
                        print(f"Error processing job {index + 1}: {e}")

                # Move to the next page
                try:
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Next" or contains(text(),"Next")]'))
                    )
                    
                    # Check if the next button is enabled
                    if "disabled" not in next_button.get_attribute("class"):
                        next_button.click()
                        time.sleep(3)  # Wait for the new page to load
                        page += 1
                    else:
                        print("No more pages to scrape.")
                        break
                except Exception as e:
                    print("Next button not found or pagination error:", e)
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                break

    # Close the browser
    driver.quit()
    print(f"Job data for {role} has been saved to {csv_filename}.")

def main():
    # List of roles to scrape
    roles = [
        "Full Stack Developer",
        "Frontend Developer",
        "Backend Developer",
        "Software Engineer",
        "Web Developer",
        "Java Developer"
    ]

    # Name of the CSV file where the data will be saved
    csv_filename = "dice_full_stack_development_jobs.csv"

    # Scrape the jobs for each role and save to CSV
    for role in roles:
        scrape_dice_jobs(role, csv_filename)

if __name__ == "__main__":
    main()
