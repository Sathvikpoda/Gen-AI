from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os

# Set up the path to your ChromeDriver
driver_path = r"C:\Users\sathv\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# Check if the driver path exists
if not os.path.isfile(driver_path):
    raise FileNotFoundError(f"ChromeDriver not found at: {driver_path}")

# Start the ChromeDriver service
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

def scrape_f18_graduate_data(start_id, end_id):
    url = 'https://ceoaperolls.ap.gov.in/AP_MLC_2024/ERO/Status_Update_2024/knowYourApplicationStatus.aspx'
    driver.get(url)

    # Click the "Graduate" tab once
    try:
        graduate_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@href="#Graduate" and @data-toggle="tab"]'))
        )
        graduate_tab.click()
    except Exception as e:
        print(f"Error clicking Graduate tab: {e}")
        driver.quit()
        return

    all_data = []
    current_id = start_id

    while True:
        try:
            # Search input field and search button
            search_input = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.ID, 'txtSearch'))
            )
            search_button = driver.find_element(By.ID, 'btnTeachersEW')

            # Clear the input field using JavaScript and set new value
            driver.execute_script("arguments[0].value = '';", search_input)
            search_input.send_keys(current_id)

            # Click the search button
            search_button.click()

            # Wait for data to appear in the "Graduate" tab or timeout if not found
            tab_content = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.ID, 'Graduate'))
            )

            # Scrape all data within the "Graduate" tab
            data_rows = tab_content.find_elements(By.TAG_NAME, 'tr')
            row_data = {}
            for row in data_rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) >= 2:
                    key = cells[0].text.strip()
                    value = cells[1].text.strip()
                    row_data[key] = value

            # If data is found, add to the list
            if row_data:
                row_data['Application ID'] = current_id
                all_data.append(row_data)

        except Exception as e:
            print(f"An error occurred while fetching data for {current_id}: {e}")

        # Increment the application number
        current_number = int(current_id.split('-')[1])
        current_number += 1
        current_id = f"F18-{current_number:07d}"

        # Check if we have reached the end ID
        if current_id > end_id:
            break

    return all_data

# Initial and end application IDs
start_app_id = 'F18-0004321'
end_app_id = 'F18-0004325'

# Scrape data for the specified range of application IDs
data = scrape_f18_graduate_data(start_app_id, end_app_id)

# Save the data to a CSV file
if data:
    df = pd.DataFrame(data)
    df.to_csv('f18_graduate_data_range.csv', index=False)
    print("Data has been saved to f18_graduate_data_range.csv")
else:
    print("No data found.")

# Clean up
driver.quit()
