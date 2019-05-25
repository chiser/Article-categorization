# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:33:36 2019

@author: chise
"""
import csv
import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, Phrases
import nltk
import itertools

## Funktionen
def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned    
    
def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)

## Change pfad

os.chdir('D:/')

## Importieren
df=pd.read_csv('edeka_gross.csv', sep=',',header=None)

sentences=df.values[:,0]
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_features = vectorizer.fit_transform(sentences)


df_test=pd.DataFrame(df.values[3000:-1,:])
df=pd.DataFrame(df.values[1:3000,:])
#df_test=pd.read_csv('edeka.csv', sep=',',header=None)
#df.values


## Split wörtern in produkte. Für Kategorien bleiben in moment so wie es ist

#product_split = [df.values[i,0].split() for i in range(df.values[:,0])]
#newlist = [word for line in mylist for word in line.split()]

## Produkte und Kategorie array
Produkte=np.array(df.values[:,0])
Kategorie=np.array(df.values[:,1])

'''
## Create a dictionary: die ganze Wörter Vielfahlt für beides: Kategorien und Produkte

sentences=df.values[:,0]
#sentences = ["Acer spin","Acer Aspire","McBook Pro","McBook Air","Lenovo Yoga"]
vocabulary = tokenize_sentences(sentences)

## Create vectors

bagofwords(df.values[1,0], vocabulary)

#bags_train = bagofwords(Produkte, vocabulary)
'''
### Alles wieder aber mit sklearn funktionen

#vectorizer.transform([df.values[1,0],df.values[2,0]]).toarray()
bags_training = vectorizer.transform(Produkte).toarray()

## Try Word2Vec
sentences = nltk.sent_tokenize(Produkte)

all_words = [sent.split(" ") for sent in sentences]
merged_all_words = list(itertools.chain(*all_words))

word2vec = Word2Vec(all_words, min_count=1)  
vocabulary = word2vec.wv.vocab  
print(vocabulary)
v1 = word2vec.wv['240']
sim_words = word2vec.wv.most_similar('240')    

for item in vocabulary:
    word2vec.wv[item]
    
'''
## Kategorie in one hot encoding konvertieren

sentences_cat=df.values[:,1]
#sentences = ["Acer spin","Acer Aspire","McBook Pro","McBook Air","Lenovo Yoga"]
vocabulary_kat = tokenize_sentences(df.values[:,1])
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_cat = vectorizer.fit_transform(sentences_cat)
'''
## Naive Bayes für multinomial data

#X = np.random.randint(5, size=(6, 100)) {array-like, sparse matrix}, shape = [n_samples, n_features]
#y = np.array([1, 2, 3, 4, 5, 6]) array-like, shape = [n_samples]
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(bags_training, Kategorie)

## Test data in bags of words
sentences_test=df_test.values[:,0]
Produkte_test=np.array(df_test.values[:,0])
Kategorie_test=np.array(df_test.values[:,1])
bags_test = vectorizer.transform(Produkte_test).toarray()

## Compare predictions for test data with groundtruth

prediction = clf.predict(bags_test)
predicted=pd.DataFrame(prediction) 
groundtruth=pd.DataFrame(Kategorie_test)
log_prediction = clf.predict_log_proba(bags_training[:,1])
coeficients = clf._get_coef()

## Checking results
frames = [predicted, groundtruth,pd.DataFrame(df.values[:,0])]
result = pd.concat(frames,axis=1)

a=np.array(predicted)
b=np.array(groundtruth)
ratio=np.sum(a == b)/len(a)
print(ratio)   ### 0.49 vorhersagbarkeit
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#print(clf.predict(X[2:3]))