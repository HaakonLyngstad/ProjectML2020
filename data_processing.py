from sklearn import (
    model_selection,
    preprocessing,
    linear_model,
    naive_bayes,
    metrics,
    svm
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn import (
    decomposition,
    ensemble
)

import pandas
import xgboost
import numpy
import textblob
import string
import csv
from keras.preprocessing import (
    text,
    sequence
)
from keras import (
    layers,
    models,
    optimizers
)


def import_dataset(filename):
    col_list = ["title", "text"]
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df


def get_dataframe():
    trueDf = import_dataset('True.csv')
    trueDf['class'] = 1
    fakeDf = import_dataset('Fake.csv')
    fakeDf['class'] = 0

    return pandas.concat([trueDf, fakeDf])


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    return metrics.accuracy_score(predictions, valid_y)


# Import concatenated dataframe with true and false news
dataframe = get_dataframe()

# Split dataset into training and validation sets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    dataframe['text'], dataframe['class'], train_size=0.01, test_size=0.99
)

# Encode target column
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(dataframe['text'])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print(accuracy)
