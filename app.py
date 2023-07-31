from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Load the CSV file and initialize the model and embeddings
def init():
    df = pd.read_csv("test.txt", encoding="windows-1251", delimiter=';', quotechar='"', quoting=1, engine='python', header=0)
    df['text'] = df['title'] + " [SEP] " + df['description']
    data_texts = df['text'].to_list()
    dev = torch.device("cpu")  # Using CPU here
    model = SentenceTransformer('LaBSE').to(dev)
    corpus_embeddings1 = torch.load('corpus_embeddings.pt', map_location=torch.device('cpu'))
    return model, data_texts, corpus_embeddings1

model, data_texts, corpus_embeddings1 = init()

# Rest of the code that uses the model and embeddings...
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        inp_question = request.form.get('question')
        df_results = search3(inp_question)
        results = df_results.to_dict('records')
        return render_template('results.html', results=results)
    else:
        return render_template('index.html')

def search3(inp_question):
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings1)
    hits = hits[0]  # Get the hits for the first query
    results = []
    for hit in hits[0:10]:
        split_text = data_texts[hit['corpus_id']].split("[SEP]")
        title = split_text[0]  # Text before [SEP]
        description = split_text[1] if len(split_text) > 1 else ""  # Text after [SEP] if it exists, else empty string
        
        results.append({
            "id": hit['corpus_id'],  # Using corpus_id as the id
            "Relevance": hit['score'],
            "Title": title.strip(),  # Remove any leading/trailing spaces
            "Description": description.strip()  # Remove any leading/trailing spaces
        })

    # Create a DataFrame and return it
    return pd.DataFrame(results)

@app.route('/result/<int:result_id>')
def result_page(result_id):
    # Assuming you have a function to retrieve the description based on the result_id
    result = get_result_by_id(result_id)
    print(result.title)
    return render_template('result_page.html', result=result)

# Function to retrieve the result by id (you can modify this to match your data source)
def get_result_by_id(result_id):
    results = pd.read_csv("test.txt", encoding="windows-1251", delimiter=';', quotechar='"', quoting=1, engine='python', header=0)
    for result in results['id']:        
        print(result)
        if result == result_id:
            return results.iloc[result]
    # Return None if the result_id is not found
    return None



def verify_login(username, password):
    with open('accounts.csv', 'r') as file:
        for line in file:
            user, pwd = line.strip().split(',')
            if user == username and pwd == password:
                return True
    return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if verify_login(username, password):
            # Login successful, redirect to the homepage or any other page
            return render_template('index.html')
        else:
            # Login failed, show an error message or redirect back to the login page
            error_message = "Invalid username or password. Please try again."
            return render_template('login.html', error_message=error_message)
    else:
        return render_template('login.html', error_message="")


if __name__ == '__main__':
    app.run(debug=False)
