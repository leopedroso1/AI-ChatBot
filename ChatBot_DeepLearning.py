# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:25:03 2020

@author: Leonardo
"""

import nltk # Natural Languate Tool Kit
import numpy as np
import tensorflow as tf
import tflearn # WARNING! tflearn is only available at tensorflow 1.x check your installation first >> print(tensorflow.__version__)
import random
import json    
import pickle
from nltk.stem.lancaster import LancasterStemmer 

stemmer  = LancasterStemmer()
json_path = r"INSER THE PATH OF INTENTS.JSON FILE HERE"

# Open .json file with all intents separeted by classes
with open(json_path) as file: 
    
    data = json.load(file)

# Load previous trained file
try:

    with open("data.pickle", "rb") as f:
        
        words, labels, training, output = pickle.load(f)    

# Otherwise, prepare the data and train model 
except:

########## Data preparation ##########
        
    words = []
    labels = []
    docs_x = []
    docs_y = []
    training = []
    output = []

    for intent in data["intents"]: # Loop through the JSON file
    
        for pattern in intent["patterns"]: # Collect only the tag patterns from your dictionary JSON
        
            w_token_list = nltk.word_tokenize(pattern) # Transform our phrases in tokens
            words.extend(w_token_list) 
            docs_x.append(w_token_list)
            docs_y.append(intent["tag"])
        
        if intent["tag"] not in labels:
            
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    words = sorted(list(set(words))) # Set >> remove duplicates --> Return to a List and then sort
    labels = sorted(labels)

"""
Neural networks only understantd numbers and not strings, then we need to encode our word list in order to create a proper inpurt.
To do this we will create a list that each index will represent the frquency of the letters in our word.

For example for the word "coffe"
[0,0,1,0,1,2,0,0,0,0,0,0,0,0,1,...]
 a b c d e f g h i j k l m n o....

"""
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        
        bag = []    
        wrds = [stemmer.stem(w) for w in doc]
    
        for w in words: 

            if w in wrds:
            
                bag.append(1)
        
            else:
            
                bag.append(0)
        
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
    
        training.append(bag)
        output.append(output_row)
    

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
        
        picke.dump((words, labels, training, output), f)


########## Building our model ##########
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) # Input Layer
net = tflearn.fully_connected(net, 8) # Hidden layer
net = tflearn.fully_connected(net, 8) # Hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # Output layer
net = tflearn.regression(net)

model = tflearn.DNN(net) # neural network


try: 
    
    model.load("model.tflearn")


except:
    
    model.fit(training, output, n_epochs=1000, batch_size = 8, show_metric=True)
    model.save("chatbot_model.tflearn")



def bag_of_words(s, words):
    
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for sentence in s_words:

        for i, w in enumerate(words):
            
            if w == sentence:
                
                bag[i] = 1
                
    return np.array(bag)


def chat():
    
    print("Start talking with the Bot! Type 'quit' to stop")
    
    while True:
        
        input_stream = input("You: ")
        
        if input_stream.lower() == "quit":
            
            break
        
        result = model.predict([bag_of_words(input_stream, words)])
        results_index = np.argmax(results)  
        tag = labels[results_index]
        print("Tag: " + tag)
        print("Result: " + result)
        
        for tg in data["intents"]:

            if tg['tag'] == tag:

                responses = tg['responses']
                
        print(random.choice(responses))
    
chat()    