import firebase_admin.db
from flask import Flask, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from flask_cors import CORS, cross_origin
import firebase_admin
import json

app = Flask(__name__)
CORS(app, support_credentials=True)

def setup_app(app):
    global model, db, intents, responses

    model = SentenceTransformer('firqaaa/indo-sentence-bert-base')
    # df = pd.read_csv('./data/intents-2.csv', names=["pattern", 'tag', 'response'], dtype=str)   
    cred = firebase_admin.credentials.Certificate('firebase-credentials.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://chat-app-14a2e-default-rtdb.asia-southeast1.firebasedatabase.app"
    })
    db = firebase_admin.db

    ref = db.reference("/intents").get()
    
    intents = []
    intentsCollections = []
    responseCollections = []
    responses = []
    
    for i in range(len(ref)):
        if(i>0):
            temp = ref[i]
            intentsCollections.append(temp['patterns'])
            responseCollections.append(temp['responses'])
    
    # Preparing intents list
    for j in range(len(intentsCollections)):
        for intent in intentsCollections[j]:
            intents.append(intent)
    
    # Preparing response list
    for j in range(len(responseCollections)):
        for response in responseCollections[j]:
            responses.append(response)
setup_app(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chatbot", methods=['POST'])
def chat():
    sentence = model.encode(request.form['sentence'])
    embeddings = model.encode(intents)
    similarityResult = np.array(model.similarity(sentence, embeddings))
    highestIndex = np.argmax(similarityResult)
    highestVal = np.max(similarityResult)
    if(highestVal >= 0.5):
        print("\n[MODEL DEBUG LOG]")
        print("Similarity result: ", similarityResult)
        print("Highest index: ", highestIndex)
        print("Highest value: ", highestVal)
        print("Most similar sentence: ", intents[highestIndex])
        print("Response: ", responses[highestIndex], "\n")
    else:
        ref = db.reference("/new-knowledge")
        delimiter = ("{", "}")
        ref.set(request.form['sentence'])
        return  "Maaf, kami belum mengerti tentang apa yang sedang anda bicarakan. Kami akan berusaha agar dapat memahami anda kedepannya."
    return  responses[highestIndex]

@app.route('/firebase-test')
def dbTestConnection():
    ref = db.reference("/intents").get()
    return ref

@app.route('/get-intents')
def getIntents():
    ref = db.reference("/intents").get()
    return ref

if __name__ == '__main__':
    app.run(host='192.168.50.106', port=5555, debug=True)