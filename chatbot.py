import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from tensorflow import keras


lemmatizer = WordNetLemmatizer()

# initializing chatbot training
words = []
classes = []
documents = []
lang = []
ignore_letters = ['!', '?', ',', '.']
intentss = json.loads(open('intents.json', encoding='utf-8').read())



for intent in intentss['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # adding documents
        documents.append((word_list, intent['tag']))
        # adding classes into our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    if intent['tag'] not in lang:
        lang.append(intent['context_set'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #set eliminates duplicates 

#classes = sorted(list(set(classes)))
#print(len(lang), "lan",lang)
#print(len(documents), "patterns",documents)
#print(len(classes), "classes",classes)
#print(len(words), "unique lem words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
pickle.dump(documents, open('patterns.pkl', 'wb'))
pickle.dump(lang, open('language.pkl', 'wb'))
# initializing training data


#machine learning part
training = []
output_empty = [0] * len(classes)
D_ty = object

for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    word_patterns = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=D_ty)

# train and test lists X-patterns , Y- intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# neural network model
# sequential model
# create model-3 layers.First layer 128 neurons, second layer 64 neurons and third layer output  contains
# number of neurons equal to number of intents to predict output intent with softmax
model = Sequential() #a deep learnig model from keras
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5)) # prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model.Stochastic gradient descent with Nesterov accelerated gradient gives goo result for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
vari = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', vari)
