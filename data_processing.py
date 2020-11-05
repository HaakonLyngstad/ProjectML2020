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


def get_dataframe(col_list, filename):
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df.drop_duplicates().sample(frac=1).reset_index(drop=True)


def train_model(classifier, name, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    #print("\n\nFVT: \n", feature_vector_train.shape, "\n\n\nLABEL: \n", label)
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
    return

# Define dataset
train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"

# Import dataframe from csv
cols = [train_col, valid_col]
df = get_dataframe(col_list=cols, filename=filename)

# Split dataset into training and validation sets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    df[train_col], df[valid_col], train_size=0.5, test_size=0.5
)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df[train_col])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

classifier_list = [naive_bayes.MultinomialNB(), svm.SVC(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier(), xgboost.XGBClassifier()]
classifier_names = ["Naive Bayes", "SVM", "RFC", "AdaBoost", "XGBC"]
for index, classifier in enumerate(classifier_list):
    train_model(
        classifier,
        classifier_names[index],
        xtrain_count,
        train_y,
        xvalid_count,
        valid_y
    )

# predict the labels on validation dataset
