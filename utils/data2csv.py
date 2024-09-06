import os
import csv
from openai import OpenAI
import tiktoken  # Use tiktoken for token counting
import re
import pandas as pd
import argparse

# Initialize OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')


client = OpenAI(api_key=api_key)


# Set the token limit for each chunk
TOKEN_LIMIT = 1600
EMBEDDING_MODEL = "text-embedding-3-small"


def clean_press_release(text):
    """
    Clean the press release by removing unwanted sections such as press release label, 
    contact information, and company description.
    """
    
    # Remove the "PRESS RELEASE" label if present
    text = re.sub(r'PRESS RELEASE\n', '', text)
    
    # Remove sections like "Contact", "About Alternaleaf", "About Montu", etc.
    text = re.sub(r'\nAbout .*', '', text)  # Removes from "About" sections onward
    text = re.sub(r'\nContact\n.*', '', text, flags=re.DOTALL)  # Removes everything after "Contact"

    # Optional: Remove any extra newlines or spaces
    text = re.sub(r'\n{2,}', '\n\n', text)  # Replace multiple newlines with two newlines
    text = text.strip()  # Remove leading/trailing whitespaces
    
    return text

def read_txt_files_from_folders(folder_paths):
    """Read all .txt files from multiple folders and return a list of their contents."""
    file_contents = []
    
    for folder_path in folder_paths:
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        # Read the content of each .txt file in the folder
        for txt_file in txt_files:
            with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as f:
                cleaned_text = clean_press_release(f.read())
                file_contents.append(cleaned_text)
    
    return file_contents

def num_tokens_from_string(string, model="gpt-3.5-turbo"):
    """Returns the number of tokens in a string based on the specified model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))

def split_text_by_tokens(text, token_limit=TOKEN_LIMIT, model="gpt-3.5-turbo"):
    """Split text into chunks, each containing no more than token_limit tokens."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    # Split into chunks of token_limit
    chunks = [tokens[i:i + token_limit] for i in range(0, len(tokens), token_limit)]
    
    # Decode the tokens back into text
    decoded_chunks = [encoding.decode(chunk) for chunk in chunks]
    return decoded_chunks

def generate_summary_and_sub_summary(text, model="gpt-3.5-turbo"):
    """Use OpenAI's API to generate a summary and sub-summary from the text."""
    prompt = (
        "Please generate a summary and sub-summary for the following text.\n\n"
        "Text: " + text + "\n\n"
        "Provide the output in the following format:\n"
        "Summary: ...\n"
        "Sub-summary: ..."
    )

    # Make the API call to OpenAI's GPT model
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that generates summaries."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the API response
    summary_response = completion.choices[0].message.content
    
    # Split the response into summary and sub-summary
    summary, sub_summary = "", ""
    for line in summary_response.splitlines():
        if line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
        elif line.startswith("Sub-summary:"):
            sub_summary = line.replace("Sub-summary:", "").strip()
    
    return summary, sub_summary

def generate_embedding(text, model=EMBEDDING_MODEL):
    """Use OpenAI's API to generate embeddings for the given text."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    # Extract the embeddings (first item in the 'data' list)
    embedding = response.data[0].embedding
    return embedding

def save_to_csv(data, output_csv_path):
    """
    Save the data (summary, sub-summary, body, and embedding) to a CSV file 
    by first converting it to a DataFrame.
    """
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data, columns=["Summary", "Sub-summary", "Body", "embedding"])
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

def process_txt_files_to_csv(company_name):
    """Read all .txt files from a folder, process them, and save to a CSV with the company name."""
    folder_path = f'./utils/data/{company_name}'
    output_csv_path = f'./utils/data/{company_name}/{company_name}.csv'
    os.makedirs(folder_path, exist_ok=True)
    file_contents = read_txt_files_from_folders([folder_path, './utils/data/Kobe'])

    processed_data = []
    for text in file_contents:
        chunks = split_text_by_tokens(text)
        for chunk in chunks:
            summary, sub_summary = generate_summary_and_sub_summary(chunk)
            embedding = generate_embedding(summary)
            processed_data.append((summary, sub_summary, chunk, embedding))
    save_to_csv(processed_data, output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process text files and generate CSV for a company.')
    parser.add_argument('--company', type=str, required=True, help='The name of the company')
    args = parser.parse_args()

    company_name = args.company

    process_txt_files_to_csv(company_name)

    print(f"CSV file for {company_name} has been generated!")