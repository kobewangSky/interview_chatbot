# OpenAI-Based Chatbot with Flask

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [API](#api)
  - [Login](#login)
- [Database Management](#database-management)
- [Text Processing and Scraping](#text-processing-and-scraping)
- [License](#license)

## Overview

This project is a Flask-based web application integrated with OpenAI’s GPT models. It allows for a multi-tenant chatbot system where different users access different databases and chat with GPT-powered assistants relevant to their company or dataset.

## Features

- **Multiple Assistant Profiles**: Different companies can log in and interact with a custom assistant, backed by their own dataset.
- **Customizable Databases**: Each company can have its own CSV database with pre-processed embeddings to improve chatbot performance.
- **Real-time Chat**: Users can ask questions and receive responses in real-time.
- **Website Scraper**: Ability to scrape website content and save it into a file for processing.
- **Text Summarization and Embeddings**: The system supports breaking large documents into chunks and generating embeddings and summaries, which are used to answer queries.

## Requirements

- Python 3.8+
- OpenAI API Key
- Flask
- Pandas
- SciPy
- Requests
- BeautifulSoup
- tiktoken

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repository/chatbot-flask-openai.git
    cd chatbot-flask-openai
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

1. **API Key**: Add your OpenAI API key to the environment:
    ```bash
    export OPENAI_API_KEY=your_openai_api_key_here
    ```

2. **File Structure**: Ensure the following file structure is maintained for company-specific datasets:
    ```
    /utils/data/{company_name}/{company_name}.csv
    ```
    

## Usage

### Running the App

To run the Flask application, use:
```bash
python app.py
