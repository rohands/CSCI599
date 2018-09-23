import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree, cross_validation, neighbors
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

path = '/Users/rohands/Desktop/DS/Tweets.csv'
data = pd.read_csv(path)
print list(data)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit(data['text'])
print vectorizer.idf_
print len(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(data['text'])
print vector.shape
vecarray = vector.toarray()

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
    y_train, y_test = data['airline_sentiment'][train_index], data['airline_sentiment'][test_index]
    
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
print(classification_report(ytlog, yplog, target_names=data.airline_sentiment.unique()))


