import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree, cross_validation, neighbors
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import re
import nltk
import string
import unicodedata
import numpy as np
import json
import io
import datetime as dt
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
import time


start_time = time.time()

nlp = spacy.load('en', parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()

path = 'Tweets.csv'
csv_data = pd.read_csv(path)
print 'Number of observations are: '+ str(len(csv_data))
data = csv_data.text.dropna()
data = data.reset_index(drop=True)
print 'Number of observations are: '+str(len(data))

tweet_dictionary = {}
i = 0
for line in data:
        tweet_dictionary[i] = line.lower()
        i += 1

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=strip_links(tweet_dictionary[i])

def strip_mentions(text):
    entity_prefixes = ['@']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=strip_mentions(tweet_dictionary[i])

def strip_hashtags(text):
    entity_prefixes = ['#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=strip_hashtags(tweet_dictionary[i])

for i in range(0,len(data)):
    tweet_dictionary[i] = tweet_dictionary[i].replace('RT', '')

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = unicode(text,"utf-8")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=remove_special_characters(tweet_dictionary[i], remove_digits=True)

stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=remove_stopwords(tweet_dictionary[i])

print "Lemmatizing"
def lemmatize_text(text):
    if type(text) == str:
        text = unicode(text,"utf-8")
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=simple_stemmer(tweet_dictionary[i])

print "Building model"
tweets = tweet_dictionary.values()
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
vector = vectorizer.fit_transform(tweets)
print vectorizer.idf_
print len(vectorizer.vocabulary_)
# encode document
print vector.shape
vecarray = vector.toarray()
print len(vecarray)



clf = tree.DecisionTreeClassifier(criterion='entropy')
folds = 10
kf = cross_validation.KFold(len(vecarray), n_folds=folds, shuffle=True)
foldid = 0
totacc = 0.
ytlog = []
yplog = []
for train_index, test_index in kf:
    foldid += 1
    print("Starting Fold %d" % foldid)
    print("\tTRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = vecarray[train_index], vecarray[test_index]
    y_train, y_test = csv_data['airline_sentiment'][train_index], csv_data['airline_sentiment'][test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_pred, y_test)
    totacc += acc
    ytlog += list(y_test)
    yplog += list(y_pred)
    
    print('\tPrediction: ', y_pred)
    print('\tCorrect:    ', y_test)
    print('\tAccuracy:', acc)

print("Average Accuracy: %0.3f" % (totacc / folds,))
print(classification_report(ytlog, yplog, target_names=csv_data.airline_sentiment.unique()))
print("--- %s seconds ---" % (time.time() - start_time))


