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


def count_vectors(train_x, valid_x, train_col):
    vectors = []
    # transform the training and validation data using count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_col)

    xtrain_count = count_vect.transform(train_x)
    xvalid_count = count_vect.transform(valid_x)

    vectors.append((xtrain_count, xvalid_count))
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train_col)
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    vectors.append((xtrain_tfidf, xvalid_tfidf))


    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(train_col)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    vectors.append((xtrain_tfidf_ngram, xvalid_tfidf_ngram))

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(train_col)
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x) 
    # predict the labels on validation dataset
    vectors.append((xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars))

    return vectors


def train_model(classifier, name, train_x, train_y, valid_x, valid_y):
    # fit the training dataset on the classifier
    #print("\n\nFVT: \n", feature_vector_train.shape, "\n\n\nLABEL: \n", label)
    classifier.fit(train_x, train_y)
    # predict the labels on validation dataset
    predictions = classifier.predict(valid_x)

    accuracy = metrics.accuracy_score(valid_y, predictions)
    f1 = metrics.f1_score(valid_y, predictions)
    precision = metrics.precision_score(valid_y, predictions)
    recall = metrics.recall_score(valid_y, predictions)

    print("\n\n", name)
    print("ACCURACY: ", accuracy)
    print("F1: ", f1)
    print("PRECISION: ", precision)
    print("RECALL: ", recall)
    print(pandas.crosstab(valid_y, predictions, rownames=["Actual"], colnames=["Predicted"]))
    return

# Define dataset
train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"

# Import dataframe from csv
cols = [train_col, valid_col]
df = get_dataframe(col_list=cols, filename=filename)

vizualise_data(df, train_col, valid_col)
exit()

# Split dataset into training and validation sets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    df[train_col], df[valid_col], train_size=0.5, test_size=0.5
)

training_sets = count_vectors(train_x=train_x,
                              valid_x=valid_x,
                              train_col=df[train_col])

print(count_vectors[0][1])

classifier_list = [naive_bayes.MultinomialNB(), svm.SVC(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier(), xgboost.XGBClassifier()]
classifier_names = ["Naive Bayes", "SVM", "RFC", "AdaBoost", "XGBC"]
for index, classifier in enumerate(classifier_list):
    for (train_x, valid_x) in training_sets[0:2]:
        train_model(
            classifier,
            classifier_names[index],
            train_x,
            train_y,
            valid_x,
            valid_y
        )
