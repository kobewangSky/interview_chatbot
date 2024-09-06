from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from openai import OpenAI
import pandas as pd
import ast
import tiktoken  # For token counting
from scipy import spatial 
import os

import secrets
secret_key = secrets.token_hex(32)

app = Flask(__name__)
app.secret_key = secret_key

PASSWORDS = {
    "montu": {"db_path": "./utils/data/Montu/Montu.csv", "assistant_name": "Montu"},
    "hipagesgroup": {"db_path": "./utils/data/hipagesgroup/hipagesgroup.csv", "assistant_name": "hipagesgroup"},
    "cbhs": {"db_path": "./utils/data/cbhs/cbhs.csv", "assistant_name": "cbhs"},
    "woolworthsgroup": {"db_path": "./utils/data/woolworthsgroup/woolworthsgroup.csv", "assistant_name": "woolworthsgroup"}
    # Add more as needed
}

DATABASE_CACHE = {}

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key )
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

# df = pd.read_csv(embeddings_path)
# df['embedding'] = df['embedding'].apply(ast.literal_eval)

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["Body"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=3)
    introduction = 'Use the below articles answer the subsequent question."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nArticle section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": f"You are a {PASSWORDS[session['password']]['assistant_name']} helpful assistant."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

def load_database(password: str) -> pd.DataFrame:
    """Loads the database and caches it. Returns the cached database if already loaded."""
    if password in DATABASE_CACHE:
        return DATABASE_CACHE[password]
    
    # If the database is not in cache, load from file and store in cache
    db_path = PASSWORDS[password]['db_path'] 
    df = pd.read_csv(db_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    DATABASE_CACHE[password] = df
    return df

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    company_name = request.form['company_name'].lower().strip()
    
    # Check if the password is valid
    if company_name in PASSWORDS:
        session['password'] = company_name
        return redirect(url_for('index'))
    else:
        flash("Invalid password. Please try again.")
        return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'password' not in session:
        return redirect(url_for('login'))

    # Get the assistant name based on the password
    assistant_name = PASSWORDS[session['password']]['assistant_name']
    
    return render_template('index.html', assistant_name=assistant_name)

@app.route('/chat', methods=['POST'])
def chat():
    if 'password' not in session:
        return redirect(url_for('login'))

    user_input = request.form['user_input']
    password = session['password']
    
    # Load (or retrieve from cache) the corresponding database
    df = load_database(password)
    
    # Use the ask function to get relevant text and generate a response
    chatbot_response = ask(user_input, df, model=GPT_MODEL)

    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)



