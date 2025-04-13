import json

import unicodedata
from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL

from chatty import chatbot_response

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'testt'
mysql = MySQL(app)

translationsJson = '{ \
    "en" : { \
        "askForName":"Hi there! What is your name?", \
        "welcomeText" : "Hi name_placeholder ! Do you need help with the registry services or with the municipal register services ?" \
    }, \
    "gr" : { \
        "askForName":"Γειά σας! Πως είναι το όνομα σας;", \
        "welcomeText" : "Γειά σου name_placeholder ! Χρειαζεσαι βοήθεια με το ληξιαρχείο η με το δημοτολόγιο;" \
    }, \
    "common" : { \
            "askForLang" : "For English type en.Για ελληνικά πληκτρολογήστε gr." \
    } \
}'
translations = json.loads(translationsJson);


@app.get("/")
def index_get():
    return render_template("base.html")


def strip_accents(s):  # ignore accents
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    lang = request.get_json().get("lang")
    cursor = mysql.connection.cursor()
    cursor.execute(''' INSERT INTO logs(phrase,operator) VALUES(%s,%s)''', (text, "human"))
    mysql.connection.commit()
    cursor.close()
    e = strip_accents(text)  # ignore accents
    response = chatbot_response(e, lang)
    message = {"answer": response}
    cursor = mysql.connection.cursor()
    cursor.execute(''' INSERT INTO logs(phrase,operator) VALUES(%s,%s)''', (response, "bot"))
    mysql.connection.commit()
    cursor.close()
    return jsonify(message)


@app.post("/init")
def init():
    lang = request.get_json().get("lang")
    name = request.get_json().get("name")
    if lang is None:
        message = {"answer": translations["common"]["askForLang"], "action": "setLang"}
        return jsonify(message)
    if name is None:
        message = {"answer": translations[lang]["askForName"], "action": "setName"}
        return jsonify(message)
    message = {"answer": (translations[lang]["welcomeText"]).replace("name_placeholder", name), "action": ""}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, ssl_contex="adhoc")

    

    #host="0.0.0.0", port=5000, debug=true , ssl_contex="adhoc"
