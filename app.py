from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from openai import OpenAI
import pandas as pd
import ast
import tiktoken  # For token counting
from scipy import spatial 
import os
import secrets
from pinecone import Pinecone

secret_key = secrets.token_hex(32)

app = Flask(__name__)
app.secret_key = secret_key


DATABASE_CACHE = {}

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

Pinecone_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=Pinecone_key)
Pinecone_index = pc.Index("interview-chatbot")

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def strings_ranked_by_relatedness(
    query: str,
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    
    query_embedding = query_embedding_response.data[0].embedding
    
    query_results = Pinecone_index.query(
        namespace=session['password'],
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )
    
    strings_and_relatednesses = []
    for match in query_results['matches']:
        if 'metadata' in match and 'body' in match['metadata']:
            strings_and_relatednesses.append((match['metadata']['body'], match['score']))
        else:
            # If metadata or body is not available, use the id as a fallback
            strings_and_relatednesses.append((f"Content for ID: {match['id']}", match['score']))
    
    if not strings_and_relatednesses:
        return [], []
    
    strings, relatednesses = zip(*strings_and_relatednesses)
    return list(strings), list(relatednesses)

def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, top_n=3)
    introduction = 'Use the below articles to answer the subsequent question."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nArticle section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": f"You are a {session['assistant_name']} helpful assistant."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


def namespace_exists(index, namespace):
    namespaces = index.describe_index_stats()['namespaces']
    return namespace in namespaces

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    company_name = request.form['company_name'].lower().strip()
    
    if namespace_exists(Pinecone_index, company_name):
        session['password'] = company_name
        session['assistant_name'] = company_name  # Using the folder name as the assistant name
        return redirect(url_for('index'))
    else:
        flash("Invalid company name. Please try again.")
        return redirect(url_for('login'))
    

@app.route('/index')
def index():
    if 'password' not in session:
        return redirect(url_for('login'))

    assistant_name = session['assistant_name']
    
    return render_template('index.html', assistant_name=assistant_name)

@app.route('/chat', methods=['POST'])
def chat():
    if 'password' not in session:
        return redirect(url_for('login'))

    user_input = request.form['user_input']
    password = session['password']
    
    # Use the ask function to get relevant text and generate a response
    chatbot_response = ask(user_input, model=GPT_MODEL)

    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
