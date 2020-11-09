from sklearn import (
    model_selection,
    preprocessing,
    naive_bayes,
    metrics,
    svm,
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn import ensemble

import pandas
import xgboost

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def import_dataset(filename):
    col_list = ["title", "text"]
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df


def get_dataframe(col_list, filename):
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df.drop_duplicates().sample(frac=1).reset_index(drop=True)


def count_vectors(train_x, valid_x, train_col):
    # transform the training and validation data using count vectorizer object
    count_vect = CountVectorizer(analyzer='word',
                                 token_pattern=r'\w{1,}')
    count_vect.fit(train_col)

    train_x_count = count_vect.transform(train_x)
    valid_x_count = count_vect.transform(valid_x)

    return (train_x_count, valid_x_count)

def tfid_vectors(train_x, valid_x, train_col):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 max_features=5000)

    tfidf_vect.fit(train_col)
    train_x_tfid = tfidf_vect.transform(train_x)
    valid_x_tfid = tfidf_vect.transform(valid_x)
    return (train_x_tfid, valid_x_tfid)


def ngram_vectors(train_x, valid_x, train_col):
    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word',
                                       token_pattern=r'\w{1,}',
                                       ngram_range=(2, 3),
                                       max_features=5000)
    tfidf_vect_ngram.fit(train_col)
    train_x_ngram = tfidf_vect_ngram.transform(train_x)
    valid_x_ngram = tfidf_vect_ngram.transform(valid_x)

    return (train_x_ngram, valid_x_ngram)


def tokenize_text(train, test, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    texts = train + test

    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    xtrain = tokenizer.texts_to_sequences(train)
    xtrain = pad_sequences(xtrain, maxlen=MAX_SEQUENCE_LENGTH)

    xtest = tokenizer.texts_to_sequences(test)
    xtest = pad_sequences(xtest, maxlen=MAX_SEQUENCE_LENGTH)

    return xtrain, xtest


def train_model(classifier, name, train_x, train_y, valid_x, valid_y):
    # fit the training dataset on the classifier
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
    print(pandas.crosstab(valid_y, predictions,
                          rownames=["Actual"],
                          colnames=["Predicted"]))
    return


def data_processor(train_x, valid_x, train_col):
    processed_data = dict()

    processed_data["NB"] = count_vectors(train_x=train_x,
                                         valid_x=valid_x,
                                         train_col=train_col)
    processed_data["SVM"] = tfid_vectors(train_x=train_x,
                                         valid_x=valid_x,
                                         train_col=train_col)
    ngram_vector = ngram_vectors(train_x=train_x,
                                 valid_x=valid_x,
                                 train_col=train_col)
    processed_data["RFC"] = ngram_vector
    processed_data["ADA"] = ngram_vector
    processed_data["XGBC"] = ngram_vector
    processed_data["BG"] = ngram_vector


    tokenized_sequence = tokenize_text(train_x.to_numpy().tolist(),
                                       valid_x.to_numpy().tolist(),
                                       MAX_NB_WORDS,
                                       MAX_SEQUENCE_LENGTH)

    processed_data["RCNN"] = tokenized_sequence
    processed_data["LSTM"] = tokenized_sequence

    return processed_data


def get_processed_dataset_dict(train_col_name, valid_col_name, filename, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    # Import dataframe from csv
    cols = [train_col, valid_col]
    df = get_dataframe(col_list=cols, filename=filename)

    # Split dataset into training and validation sets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        df[train_col], df[valid_col], train_size=0.5, test_size=0.5)

    return data_processor(train_x=train_x,
                          valid_x=valid_x,
                          train_col=df[train_col])



train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"
# This is fixed.
EMBEDDING_DIM_LSTM = 16
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500

classifier_list = [naive_bayes.MultinomialNB(),
                   svm.SVC(),
                   ensemble.RandomForestClassifier(),
                   ensemble.AdaBoostClassifier(),
                   xgboost.XGBClassifier(),
                   ensemble.BaggingClassifier()]
classifier_names = ["NB", "SVM", "RFC", "ADA", "XGBC", "BG", "RCNN", "LSTM"]
for index, classifier in enumerate(classifier_list):
    (train_x, valid_x) = processed_data[classifier_names[index]]
    train_model(
        classifier,
        classifier_names[index],
        train_x,
        train_y,
        valid_x,
        valid_y
    )
