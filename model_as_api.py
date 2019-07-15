# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:22:03 2019

@author: chise
"""

from flask import Flask, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
import pickle
import os
#from model import NLPModel
#initalize our flask app
app = Flask(__name__)
os.chdir('D:/repositories/anybill/')

@app.route('/')
def home():
    return '<h1>Predicting products. Add /predict to the browser</h1>'

class inputForm(FlaskForm):
    product_in = StringField('Product',
                validators=[DataRequired(),Length(min=2,max=20)])
    submit = SubmitField('Submit')

@app.route('/predict', methods=['GET','POST'])
def get_prediction():
    
	#get the raw data format
    inputData = inputForm()
    #inputData = request.get_data()
    #inputData = float(request.args.get('d'))
    # load the vectorizer to convert to bags of words
    filename = 'vectorizer_model.pkl'
    vectorizer = pickle.load(open(filename, 'rb'))
	#encode it into a suitable format
    bow_input = vectorizer.transform(inputData).toarray()

    # load trained classifier
    modelname = 'edeka_model.pkl'
    predictor = pickle.load(open(modelname, 'rb'))
    
    out = predictor.predict(bow_input)
    #out_proba = predictor.predict_proba(bow_input)
    # round the predict proba value and set to new variable
    #confidence = round(out_proba[0], 3)
    # create JSON object
    #output = {'prediction': out, 'confidence': confidence}
 
    return out   #output
  
if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
    #app.run(port=5000,host='0.0.0.0')
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port, debug=True)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
    
#bow_input = vectorizer.transform(['10 katzensticks mit gefluegel leber','2 eierbecher']).toarray()
#print(predictor.predict(bow_input))

