# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:33:36 2019

@author: chise
"""

### Import libraries
import os
import pandas as pd
import numpy as np
#import re
import matplotlib.pyplot as plt
#from gensim.models import Word2Vec
#import nltk
#import multiprocessing
import seaborn as sns
import pickle
#from gensim import corpora, models
#from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.utils import resample
from sklearn.metrics import confusion_matrix,roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#nltk.download('punkt')
#nltk.download('stopwords')

### Functions
'''
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

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
#data['comment_text'] = data['comment_text'].str.lower()
#data['comment_text'] = data['comment_text'].apply(cleanHtml)
#data['comment_text'] = data['comment_text'].apply(cleanPunc)
#data['comment_text'] = data['comment_text'].apply(keepAlpha)
'''
### Change directory
os.chdir('D:/repositories/anybill')

### Import data
df=pd.read_csv('edeka_gross.csv', sep=',')

## Check how balanced are the classes. Two options: balance data or F1-score
#sns.countplot(df['category'],label="Count")
#plt.show()

## Delete too small categories
categ_classes=df.groupby('category').agg('count')
exclude=categ_classes[categ_classes['name']<10]
df_clean=df[(df['category'] != exclude.index[0]) & (df['category'] != exclude.index[1]) & (df['category'] != exclude.index[2])]
categ_classes=categ_classes[(categ_classes.index != exclude.index[0]) & (categ_classes.index != exclude.index[1]) & (categ_classes.index != exclude.index[2])]

########## Down and upsample under and overrepresented categories ###################
'''
# Separate majority and minority classes
df_majority = df_clean[df_clean.category == 'nonfood ha']
df_minority = df_clean[df_clean.category == 'brotaufstriche']
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=300,    # to match majority class
                                 random_state=123) # reproducible results
df_majority_downsampled = resample(df_majority, 
                                 replace=False,     # sample without replacement
                                 n_samples=300,    # to match majority class
                                 random_state=123) # reproducible results
  
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority_downsampled, df_minority_upsampled])
 
# Display new class counts
df_upsampled.category.value_counts()
'''
################ Divide data in training and test data  ##################
# Products and Categories in numpy array for further processing
Products=df_clean['name']
Categories=df_clean['category']
X_train, X_test, y_train, y_test = train_test_split(Products, Categories, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#cut=int(np.round(len(Categories)*0.75))
#df_test=pd.DataFrame(df.values[cut:-1,:])
#df_training=pd.DataFrame(df.values[1:cut,:])

## Split words in products

# Create a dictionary: for all the words in Categories and Products. A different way independent of sklearn
#vocabulary = tokenize_sentences(df_clean['name'])
# Create vectors
#bagofwords(df.values[1,0], vocabulary)
#bags_train = bagofwords(list(Products), vocabulary)
##With pd.get_dummies(). It needs tokenizing before
#bow_dummies=pd.get_dummies(Products)

## Save in sentences the product names, tokenize them and create a vocabulary list
## Create a bag of words for the products with sklearn functions
#sentences_bow=df['name']
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_features = vectorizer.fit_transform(np.array(df_clean['name']))
filename = 'vectorizer_model.pkl'
pickle.dump(vectorizer, open(filename, 'wb'))
features=vectorizer.vocabulary_
bow_training = vectorizer.transform(np.array(X_train)).toarray()
bow_val = vectorizer.transform(np.array(X_val)).toarray()
bow_test = vectorizer.transform(np.array(X_test)).toarray()
####################### LDA  ################################
'''
tfidf = models.TfidfModel()
corpus_tfidf = tfidf[bow_training]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model = gensim.models.LdaMulticore(bow_training, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
'''
################################## Try Word2Vec  #####################
'''
##Import text from edeka products and save it in corpus_raw. Sorting by category does not work well
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into a list of words remove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

#sentences = nltk.sent_tokenize(Products)
sorted_df = df.sort_values(by= [' category'], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
Products=np.array(sorted_df['name'])
all_words = [sent.split(" ") for sent in Products]
#merged_all_words = list(itertools.chain(*all_words))

#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 1
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
# Context window length.
context_size = 7
# Downsample setting for frequent words.0 - 1e-5 is good for this
downsampling = 1e-3
# Seed for the RNG, to make the results reproducible.random number generator deterministic, good for debugging
seed = 1

prod2vec = Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
prod2vec.build_vocab(all_words)
print("Word2Vec vocabulary length:", len(prod2vec.wv.vocab))
prod2vec.train(all_words, total_examples=1, epochs=1)

if not os.path.exists("trained"):
    os.makedirs("trained")
prod2vec.save(os.path.join("trained", "prod2vec.w2v"))
prod2vec = Word2Vec.load(os.path.join("trained", "prod2vec.w2v"))

## Make t-SNE just to visualize
tsne = TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = prod2vec.syn1neg
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

## Put dimension-reduced data in a data fram
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[prod2vec.wv.vocab[word].index])
            for word in prod2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)

## Plot t-SNE
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(10, 10))

## Zoom into the plot regions
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

plot_region(x_bounds=(-60, -50), y_bounds=(-15, 30))
plot_region(x_bounds=(-25, 5), y_bounds=(-40, -20))

## Rule of three relations
def nearest_similarity(start1, end1, end2):
    similarities = prod2vec.wv.most_similar(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

nearest_similarity("cola", "mix", "alkaline")

## check the vectors for each word and the similar semantics. We need to see if the mass/volume/units dimensions are important
v1 = prod2vec.wv['zahnseide']
sim_words = prod2vec.wv.most_similar('alkaline')    #'zitronenlimonade','zwiebelringe'

## word2vec for the vocabulary
bags_prod2vec=[]
for key in vocabulary.keys():
    bags_prod2vec.append(word2vec.wv[key])

## word2vec for the products.   
bags_prod2vec=[]
for item in all_words:
    bags_prod2vec.append(word2vec.wv[item])
    print(item)

#bags_word2vec = np.asarray(bags_word2vec)
#bags_word2vec.astype(int)
    
clf = MultinomialNB()
clf.fit(bags_word2vec, Categories)

## Kategorie in one hot encoding konvertieren

sentences_cat=df.values[:,1]
#sentences = ["Acer spin","Acer Aspire","McBook Pro","McBook Air","Lenovo Yoga"]
vocabulary_kat = tokenize_sentences(df.values[:,1])
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_cat = vectorizer.fit_transform(sentences_cat)
'''
################# Naive Bayes für multinomial data  #########################

nb_cl = MultinomialNB(alpha=0.000001)
nb_cl.fit(bow_training, y_train)
#nb_cl_bow = MultinomialNB()
#nb_cl_bow.fit(bow_training, bow__y_training)
# save the model to disk
filename = 'edeka_model.pkl'
pickle.dump(nb_cl, open(filename, 'wb'))

#Save the model
# serialize model to JSON
#model_json = nb_cl.to_json()
#with open("edeka.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#nb_cl.save_weights("edeka.h5")
#print("Saved model to disk") 
'''
## Set the parameters by cross-validation
nb_cl_parameters = [{'alpha': [0,0.01, 0.1, 0.5, 1]}]# the best is alpha 0.01
nb_cl_scores = ['balanced_accuracy', 'f1_weighted','roc_auc']#['precision', 'recall']

for score in nb_cl_scores:
    print("# Tuning hyper-parameters for %s" % score)
    nb_cl = GridSearchCV(MultinomialNB(), nb_cl_parameters, cv=5,
                       scoring=score)
    nb_cl.fit(bow_training, y_train)
    print("Best parameters set found on development set:")
    print(nb_cl.best_params_)
    print("Grid scores on development set:")
    means = nb_cl.cv_results_['mean_test_score']
    stds = nb_cl.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, nb_cl.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    nb_pred=nb_cl.predict(bow_val)
    print(classification_report(y_val, nb_pred))
    
##Plot a confusion matrix
mat = confusion_matrix(y_val, nb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categ_classes.index, yticklabels=categ_classes.index)
plt.xlabel('true label')
plt.ylabel('predicted label')

## Compare predictions for test data with groundtruth
prediction = pd.Series(nb_cl.predict(bow_test))
log_prediction = nb_cl.predict_log_proba(bow_test)
coeficients = nb_cl._get_coef()

## Checking results: How's our accuracy?
### 0.49 vorhersagbarkeit. After taking away small classes 0.61
print(accuracy_score(y_test, prediction) )
# Calculate accuracy with ROC
lb = LabelBinarizer()
lb.fit(y_test)

truth = lb.transform(y_test)
pred = lb.transform(prediction)
roc_auc_score(truth, pred, average='macro')
f1_score(truth, pred, average='macro')
 
# What about AUROC with class probabilities
prob_y = nb_cl.predict_proba(bow_test)
classes_predicted=nb_cl.classes_

print(roc_auc_score(truth, prob_y) )

## SVM to test bags of words
svm_cl = SVC(gamma='scale', decision_function_shape='ovo',class_weight='balanced', probability=True)
svm_cl.fit(bow_training, y_train) 

## Compare predictions for test data with groundtruth
prediction = pd.Series(svm_cl.predict(bow_test))
#log_prediction = svm_cl.predict_log_proba(bow_test)
coeficients = svm_cl._get_coef()

## Checking results: How's our accuracy?
## 0.55 after eliminating low representative classes
print(accuracy_score(y_test, prediction) )

## Calculate accuracy with ROC
pred = lb.transform(prediction)
roc_auc_score(truth, pred, average='macro') #0.75
f1_score(truth, pred, average='macro') #0.57

######### Eventually I would implement all like this ###################
# Set the parameters by cross-validation
svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm_scores = ['precision', 'recall']

for score in svm_scores:
    print("# Tuning hyper-parameters for %s" % score)
    svm_cl = GridSearchCV(SVC(), svm_parameters, cv=5,
                       scoring='%s_macro' % score)
    svm_cl.fit(bow_training, y_train)
    print("Best parameters set found on development set:")
    print(svm_cl.best_params_)
    print("Grid scores on development set:")
    means = svm_cl.cv_results_['mean_test_score']
    stds = svm_cl.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svm_cl.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print(classification_report(y_val, svm_cl.predict(bow_val)))

############################## Logistic regression  #########################
logreg = LogisticRegression()
logreg.fit(bow_training, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(bow_training, y_train))) ##0.83
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(bow_test, y_test))) ##0.60

#######################  Decision trees  #####################################
DT_cl = DecisionTreeClassifier().fit(bow_training, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(DT_cl.score(bow_training, y_train))) ##1.00
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(DT_cl.score(bow_test, y_test)))  ##0.63
#Visualize
dot_data = export_graphviz(DT_cl, out_file=None, 
                     feature_names=features,  
                     class_names=categ_classes,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("edeka") 

##############################  Kmeans classifier  ##########################
knn = KNeighborsClassifier()
knn.fit(bow_training, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(bow_training, y_train)))#0.73
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(bow_test, y_test)))#0.48

#######################  Linear Discriminant Analysis #######################
lda = LinearDiscriminantAnalysis()
lda.fit(bow_training, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(bow_training, y_train))) #0.99
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(bow_test, y_test))) #0.52

####################Random Forest. Its good for imbalanced data  ############
RF_cl = RandomForestClassifier(n_estimators=100)

# Train
RF_cl.fit(bow_training, y_train)
# Extract single tree
estimator = RF_cl.estimators_[5]

# Export as dot file
export_graphviz(RF_cl, out_file=None, 
                feature_names = features,
                class_names = categ_classes,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

RF_cl_parameters = [{'criterion': ['gini'], 'n_estimators': [5, 10, 50, 100, 500]},
                    {'criterion': ['entropy'], 'n_estimators': [5, 10, 50, 100, 500]}]
RF_cl_scores = ['balanced_accuracy', 'f1_weighted','roc_auc']

for score in RF_cl_scores:
    print("# Tuning hyper-parameters for %s" % score)
    RF_cl = GridSearchCV(RandomForestClassifier(), RF_cl_parameters, cv=5,
                       scoring='%s_macro' % score)
    RF_cl.fit(bow_training, y_train)
    print("Best parameters set found on development set:")
    print(RF_cl.best_params_)
    print("Grid scores on development set:")
    means = RF_cl.cv_results_['mean_test_score']
    stds = RF_cl.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, RF_cl.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    RF_pred=RF_cl.predict(bow_val)
    print(classification_report(y_val, RF_pred))
    
##Plot a confusion matrix
mat = confusion_matrix(y_val, nb_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categ_classes.index, yticklabels=categ_classes.index)
plt.xlabel('true label')
plt.ylabel('predicted label')


## Check multilabel classification where several labels could belong to one product
#Micro and macro averaging for multilabel that count the true positives and negatives etc. for all classes

#OneVSrest logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])
for category in categories:
    print('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n")
    
#Binary Relevance Naive Bayes: independency assumption
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))

##Classifier chains
# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
# initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())
# Training logistic regression model on train data
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")

## Just for fun to generate some wordclouds
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))
# clean
subset = data_raw[data_raw.clean==True]
text = subset.comment_text.values
cloud_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))
plt.axis('off')
plt.title("Clean",fontsize=40)
plt.imshow(cloud_clean)
# Same code can be used to generate wordclouds of other categories.
'''