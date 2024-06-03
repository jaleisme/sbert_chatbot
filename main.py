from flask import Flask, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

def setup_app(app):
    global model, sentences, df
    model = SentenceTransformer('firqaaa/indo-sentence-bert-base')
    df = pd.read_csv('./data/intents.csv', names=["pattern", 'tag', 'response'], dtype=str)
    sentences = []
    for i in range(len(df["pattern"])):
        sentences.append(df["pattern"].values[i])
setup_app(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chatbot", methods=['POST'])
def chat():
    sentence = model.encode(request.form['sentence'])
    embeddings = model.encode(sentences)
    similarityResult = np.array(model.similarity(sentence, embeddings))
    highestIndex = np.argmax(similarityResult)
    highestVal = np.max(similarityResult)
    if(highestVal >= 0.5):
        print("\n[MODEL DEBUG LOG]")
        print("Similarity result: ", similarityResult)
        print("Highest index: ", highestIndex)
        print("Highest value: ", highestVal)
        print("Most similar sentence: ", sentences[highestIndex])
        print("Response: ", df["response"][highestIndex], "\n")
    else:
        return  "Gatau ah cape"
    return  df["response"][highestIndex]

if __name__ == '__main__':
    app.run(host='192.168.50.106', port=5555, debug=True)