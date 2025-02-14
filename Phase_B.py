from flask import Flask, request, jsonify
import os
import subprocess
import json
import sqlite3
from datetime import datetime
from PIL import Image
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
import git
import duckdb
import markdown
import csv

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_task():
    task_description = request.args.get('task')
    if not task_description:
        return jsonify({"error": "Task description is required"}), 400

    try:
        result = execute_task(task_description)
        return jsonify({"result": result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"error": "File path is required"}), 400

    if not file_path.startswith('/data/'):
        return jsonify({"error": "Access to files outside /data is not allowed"}), 403

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content, 200
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

def execute_task(task_description):
    # Use an LLM to parse the task description and determine the task
    task = parse_task_with_llm(task_description)
    
    if task == "A1":
        return handle_task_A1()
    elif task == "A2":
        return handle_task_A2()
    elif task == "A3":
        return handle_task_A3()
    elif task == "A4":
        return handle_task_A4()
    elif task == "A5":
        return handle_task_A5()
    elif task == "A6":
        return handle_task_A6()
    elif task == "A7":
        return handle_task_A7()
    elif task == "A8":
        return handle_task_A8()
    elif task == "A9":
        return handle_task_A9()
    elif task == "A10":
        return handle_task_A10()
    elif task == "B3":
        return handle_task_B3()
    elif task == "B4":
        return handle_task_B4()
    elif task == "B5":
        return handle_task_B5()
    elif task == "B6":
        return handle_task_B6()
    elif task == "B7":
        return handle_task_B7()
    elif task == "B8":
        return handle_task_B8()
    elif task == "B9":
        return handle_task_B9()
    elif task == "B10":
        return handle_task_B10()
    else:
        raise ValueError("Unknown task")

def parse_task_with_llm(task_description):
    # Placeholder function to simulate LLM parsing
    # Replace this with actual LLM API call
    # For example, using OpenAI's GPT-3:
    openai.api_key = 'your_openai_api_key'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Parse the following task description and return the task ID: {task_description}",
        max_tokens=10
    )
    task_id = response.choices[0].text.strip()
    return task_id

def handle_task_A1():
    # Install uv and run datagen.py
    subprocess.run(["pip", "install", "uv"])
    subprocess.run(["python", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", "user.email"])
    return "Task A1 completed"

def handle_task_A2():
    # Extract the Prettier version from the task description
    prettier_version = "3.4.2"  # Default version, can be updated based on task description
    
    # Install the specified version of Prettier
    subprocess.run(["npm", "install", f"prettier@{prettier_version}"])
    
    # Format the contents of /data/format.md using Prettier
    subprocess.run(["npx", f"prettier@{prettier_version}", "--write", "/data/format.md"])
    
    return "Task A2 completed"

def handle_task_A3():
    # Count the number of Wednesdays in /data/dates.txt
    with open('/data/dates.txt', 'r') as file:
        dates = file.readlines()
    wednesday_count = sum(1 for date in dates if datetime.strptime(date.strip(), '%Y-%m-%d').weekday() == 2)
    with open('/data/dates-wednesdays.txt', 'w') as file:
        file.write(str(wednesday_count))
    return "Task A3 completed"

def handle_task_A4():
    # Sort contacts in /data/contacts.json
    with open('/data/contacts.json', 'r') as file:
        contacts = json.load(file)
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
    with open('/data/contacts-sorted.json', 'w') as file:
        json.dump(sorted_contacts, file, indent=4)
    return "Task A4 completed"

def handle_task_A5():
    # Write the first line of the 10 most recent .log files in /data/logs/
    log_files = sorted([f for f in os.listdir('/data/logs/') if f.endswith('.log')], key=lambda x: os.path.getmtime(os.path.join('/data/logs/', x)), reverse=True)[:10]
    with open('/data/logs-recent.txt', 'w') as outfile:
        for log_file in log_files:
            with open(os.path.join('/data/logs/', log_file), 'r') as infile:
                outfile.write(infile.readline())
    return "Task A5 completed"

def handle_task_A6():
    # Create an index of Markdown files in /data/docs/
    index = {}
    for root, _, files in os.walk('/data/docs/'):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    title = soup.find('h1').text.strip()
                    index[file] = title
    with open('/data/docs/index.json', 'w') as file:
        json.dump(index, file, indent=4)
    return "Task A6 completed"

def handle_task_A7():
    # Extract sender's email address from /data/email.txt using LLM
    with open('/data/email.txt', 'r') as file:
        email_content = file.read()
    sender_email = extract_email_with_llm(email_content)
    with open('/data/email-sender.txt', 'w') as file:
        file.write(sender_email)
    return "Task A7 completed"

def extract_email_with_llm(email_content):
    # Placeholder function to simulate LLM email extraction
    # Replace this with actual LLM API call
    openai.api_key = 'your_openai_api_key'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Extract the sender's email address from the following email content: {email_content}",
        max_tokens=50
    )
    sender_email = response.choices[0].text.strip()
    return sender_email

def handle_task_A8():
    # Extract credit card number from /data/credit-card.png using LLM
    image = Image.open('/data/credit-card.png')
    card_number = extract_card_number_with_llm(image)
    with open('/data/credit-card.txt', 'w') as file:
        file.write(card_number)
    return "Task A8 completed"

def extract_card_number_with_llm(image):
    # Placeholder function to simulate LLM card number extraction
    # Replace this with actual LLM API call
    openai.api_key = 'your_openai_api_key'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Extract the credit card number from the following image: {image}",
        max_tokens=50
    )
    card_number = response.choices[0].text.strip().replace(" ", "")
    return card_number

def handle_task_A9():
    # Find the most similar pair of comments in /data/comments.txt using embeddings
    with open('/data/comments.txt', 'r') as file:
        comments = file.readlines()
    embeddings = [get_embedding(comment) for comment in comments]
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -1)  # Ignore self-similarity
    most_similar_pair = np.unravel_index(np.argmax(similarities), similarities.shape)
    with open('/data/comments-similar.txt', 'w') as file:
        file.write(comments[most_similar_pair[0]])
        file.write(comments[most_similar_pair[1]])
    return "Task A9 completed"

def get_embedding(text):
    # Placeholder function to simulate getting embeddings
    # Replace this with actual embedding API call
    openai.api_key = 'your_openai_api_key'
    response = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=text
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def handle_task_A10():
    # Calculate total sales of "Gold" ticket type in /data/ticket-sales.db
    conn = sqlite3.connect('/data/ticket-sales.db')
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]
    conn.close()
    with open('/data/ticket-sales-gold.txt', 'w') as file:
        file.write(str(total_sales))
    return "Task A10 completed"

def handle_task_B3():
    # Fetch data from an API and save it
    response = requests.get("https://api.example.com/data")
    data = response.json()
    with open('/data/api_data.json', 'w') as file:
        json.dump(data, file, indent=4)
    return "Task B3 completed"

def handle_task_B4():
    # Clone a git repo and make a commit
    repo = git.Repo.clone_from("https://github.com/example/repo.git", "/data/repo")
    repo.git.add(A=True)
    repo.index.commit("Automated commit")
    repo.remote().push()
    return "Task B4 completed"

def handle_task_B5():
    # Run a SQL query on a SQLite or DuckDB database
    conn = duckdb.connect('/data/database.duckdb')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM table_name")
    result = cursor.fetchall()
    with open('/data/query_result.json', 'w') as file:
        json.dump(result, file, indent=4)
    conn.close()
    return "Task B5 completed"

def handle_task_B6():
    # Extract data from a website
    response = requests.get("https://example.com")
    soup = BeautifulSoup(response.content, 'html.parser')
    data = soup.find_all('p')
    with open('/data/web_data.json', 'w') as file:
        json.dump([p.text for p in data], file, indent=4)
    return "Task B6 completed"

def handle_task_B7():
    # Compress or resize an image
    image = Image.open('/data/image.png')
    image = image.resize((800, 600))
    image.save('/data/image_resized.png')
    return "Task B7 completed"

def handle_task_B8():
    # Transcribe audio from an MP3 file
    audio_file = open('/data/audio.mp3', 'rb')
    response = openai.Audio.transcribe("whisper-1", audio_file)
    transcription = response['text']
    with open('/data/audio_transcription.txt', 'w') as file:
        file.write(transcription)
    return "Task B8 completed"

def handle_task_B9():
    # Convert Markdown to HTML
    with open('/data/document.md', 'r') as file:
        markdown_content = file.read()
    html_content = markdown.markdown(markdown_content)
    with open('/data/document.html', 'w') as file:
        file.write(html_content)
    return "Task B9 completed"

def handle_task_B10():
    # Write an API endpoint that filters a CSV file and returns JSON data
    @app.route('/filter_csv', methods=['GET'])
    def filter_csv():
        column = request.args.get('column')
        value = request.args.get('value')
        with open('/data/data.csv', 'r') as file:
            reader = csv.DictReader(file)
            filtered_data = [row for row in reader if row[column] == value]
        return jsonify(filtered_data)
    return "Task B10 completed"

if __name__ == '__main__':
    app.run(debug=True)