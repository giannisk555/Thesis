import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random 
import json
import pickle 
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lan= ['']
met = False

from keras.models import load_model # loads the model that we have created at the chatbot.py script
from tensorflow import keras
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
lang = pickle.load(open('language.pkl','rb'))



def clean_up_sentence( sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words =clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if  show_details:
                    print ("found in bag: %s" % word)

    return(np.array(bag))

def predict_class(sentence, model, userLang ):
    # filter out predictions below a threshold
    if userLang == 'en' or userLang == 'gr':
        p=bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.20
        results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent":classes[r[0]], "probability": str(r[1])})
        if not return_list:
            if userLang=='en':
                return "Sorry.I couldn't understand you.Could you please be more accurate?"
            else:
                return "Συγγνώμη.Δεν σας κατάλαβα.Θα μπορούσατε να είστε λίγο πιο ακριβής;"
        tag = return_list[0]['intent']
        list_of_intents =intents['intents']
        for i in list_of_intents :
            if float(str(r[1])) > 0.75 and i['context_set']==userLang:
                if i['tag'] == tag :
                    return random.choice(i['responses'])
    else:
        return 'First select language!For English type en.Για ελληνικά πληκτρολογήστε gr.'


def chatbot_response(msg, userLang):
    ints = predict_class(msg, model, userLang)
    return ints