import requests
from bs4 import BeautifulSoup
import os
import argparse

# Use argparse to get command-line input
parser = argparse.ArgumentParser(description="Fetch and parse website content")
parser.add_argument("--company", required=True, help="company name")
parser.add_argument("--url", required=True, help="The URL of the website to scrape")

args = parser.parse_args()

# Target website URL from external input
url = args.url

folder = f'./utils/data/{args.company}'
os.makedirs(folder, exist_ok=True)

# Extract the last part of the URL to use as the filename
filename = folder + os.path.basename(url) + ".txt"

# Send a GET request to retrieve the website content
response = requests.get(url)

# Ensure the request was successful
if response.status_code == 200:
    # Use BeautifulSoup with the externally provided parser
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all text content
    text = soup.get_text()

    # Remove excess whitespace
    clean_text = " ".join(text.split())

    # Save the extracted text to a .txt file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(clean_text)

    print(f"Website content has been saved to {filename}")
else:
    print(f"Unable to access the website. Status code: {response.status_code}")