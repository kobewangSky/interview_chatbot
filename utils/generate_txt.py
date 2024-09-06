import requests
from bs4 import BeautifulSoup
import os
import argparse
from urllib.parse import urlparse

# Use argparse to get command-line input
parser = argparse.ArgumentParser(description="Fetch and parse website content")
parser.add_argument("-c", "--company", required=True, help="Company name")
parser.add_argument("-u", "--url", required=True, help="Website URL to scrape")

args = parser.parse_args()

# Target website URL from external input
url = args.url

# Create a folder to save the file
folder = f'./utils/data/{args.company}'
os.makedirs(folder, exist_ok=True)

# Parse the URL to extract the domain and path
parsed_url = urlparse(url)
domain = parsed_url.netloc.split('.')[0]  # Extract the base domain (e.g., "hipagesgroup")
path = parsed_url.path.strip("/").replace("/", "_")  # Replace slashes with underscores

# Determine the filename based on whether the path is empty or not
if path == "":
    filename = os.path.join(folder, f"{domain}.txt")  # Use the domain for the root URL
else:
    filename = os.path.join(folder, f"{path}.txt")    # Use the modified path for non-root URLs

# Ensure that the directory for the file exists
os.makedirs(os.path.dirname(filename), exist_ok=True)

# Set the request headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Send a GET request to retrieve the website content
response = requests.get(url, headers=headers)

# Ensure the request was successful
if response.status_code == 200:
    # Use BeautifulSoup to parse the website content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all text content
    text = soup.get_text()

    # Remove excess whitespace
    clean_text = " ".join(text.split())

    # Save the extracted text to a .txt file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(clean_text)

    print(f"Website content has been saved to {filename}")
else:
    print(f"Unable to access the website. Status code: {response.status_code}")
