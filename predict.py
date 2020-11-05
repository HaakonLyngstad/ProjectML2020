import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

trainheadlines = []
for row in range(0, len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))


testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))


vectorizer = CountVectorizer(ngram_range=(2,2))

vecorized_train = vectorizer.fit_transform(trainheadlines)
vectorized_test = vectorizer.transform(testheadlines)

classifiers = [AdaBoostClassifier, RandomForestClassifier, xgboost, MultinomialNB, BernoulliNB, SVC]
for index, classifier in enumerate(classifiers):
    train_model(

    )


def train_model(classifier, name, feature_vector)


""" 
classifier = AdaBoostClassifier()
classifier = classifier.fit(vecorized_train, train["Label"])

predictions = classifier.predict(vectorized_test)
accuracy = accuracy_score(test["Label"], predictions)
print(accuracy) """



""" predictions = basicmodel.predict(basictest)

basicresults = pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
print(basicresults)
accuracy = accuracy_score(test["Label"], predictions)
print(accuracy) """
