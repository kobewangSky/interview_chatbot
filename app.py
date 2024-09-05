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
    "Montu": {"db_path": "./utils/data/Montu/Montu.csv", "assistant_name": "Montu Assistant"}
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
        {"role": "system", "content": "You are a chatbot created by Kobe, a Senior Computer Vision Engineer with 8 years of experience in AI, specializing in end-to-end AI solutions. Kobe has extensive expertise in building and deploying large-scale AI systems, particularly in computer vision and edge computing. He has worked at several top organizations, including Digital Domain, National Tsing Hua University, black.ai, and Lumachain.\n\nAt Digital Domain, Kobe developed advanced face tracking and 3D face reconstruction models using deep learning. He also built a digital avatar video call app by integrating Unreal Engine with Java, JNI, and C++ to design APIs and define the app’s architecture.\n\nKobe led an AI research team at National Tsing Hua University, where his team won first place at the CVPR Robotic Vision Challenge in 2019. He contributed significantly to human pose estimation and virtual data generation, using models like CenterNet and Yolo v3. His work with virtual data improved model accuracy by 12%-24% in real-world applications.\n\nAt black.ai, Kobe optimized segmentation models, re-identification systems, and keypoint detection algorithms, reducing model inference times by up to 60% while increasing accuracy. He built robust feature engineering workflows and established MLOps pipelines using Docker to automate and streamline development.\n\nCurrently, Kobe is the ML Lead at Lumachain, where he has developed over 30 AI-powered CCTV products for object detection, motion tracking, anomaly detection, and meat recognition. He designed and deployed end-to-end CI/CD pipelines for Azure IoT Edge, incorporating message routing to Service Bus, Blob Storage, Azure Functions, and Cosmos DB. His work on AI model optimization led to a 10x increase in speed while maintaining 99% accuracy, using YOLOv7, TensorRT, onnxsim, and FP16. Kobe also built infrastructure for monitoring edge devices and AI models at scale.\n\nKobe is highly skilled in model compression, quantization, and optimization for deployment on edge devices. He has deep knowledge of Linux, Docker, and ETL processes, enabling efficient data pipelines and real-time AI inference.\n\nIf asked, explain how Kobe designs IoT Edge systems from the ground up, including setting up edge devices, deploying AI models, and integrating with cloud services for scalable processing. You should also emphasize Kobe's ability to optimize models for edge computing, including reducing model size and improving inference time. His expertise also covers system architecture, MLOps, infrastructure as code, and production-level deployment in cloud environments."},
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
    password = request.form['password']
    
    # Check if the password is valid
    if password in PASSWORDS:
        session['password'] = password
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



