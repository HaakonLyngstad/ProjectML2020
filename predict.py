import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn import metrics

data = pd.read_csv('Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

trainheadlines = []
for row in range(0, len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row, 2:27]))

print(len(trainheadlines))


testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))


vectorizer = CountVectorizer(ngram_range=(1, 2))

xtrain = vectorizer.fit_transform(trainheadlines)
ytrain = train["Label"]
xtest = vectorizer.transform(testheadlines)
ytest = test["Label"]

print((ytest == 1).sum(), (ytest == 0).sum())


def train_model(classifier, name, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    accuracy = metrics.accuracy_score(valid_y, predictions)
    f1 = metrics.f1_score(valid_y, predictions)
    precision = metrics.precision_score(valid_y, predictions)
    recall = metrics.recall_score(valid_y, predictions)

    print("\n\n", name)
    print("ACCURACY: ", accuracy)
    print("F1: ", f1)
    print("PRECISION: ", precision)
    print("RECALL: ", recall)
    results = pd.crosstab(ytest, predictions, rownames=["Actual"], colnames=["Predicted"])
    print(results)
    return


classifiers = [AdaBoostClassifier(), RandomForestClassifier(), MultinomialNB(), BernoulliNB(), SVC()]
classifier_names = ["AdaBoost", "RandomForest", "MultinomialNB", "BernoulliNB", "SVC"]
for index, classifier in enumerate(classifiers):
    train_model(
        classifier,
        classifier_names[index],
        xtrain,
        ytrain,
        xtest,
        ytest
    )
