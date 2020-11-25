from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer)
from sklearn import model_selection

import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pickle


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
                                       ngram_range=(1, 3),
                                       max_features=5000)
    tfidf_vect_ngram.fit(train_col)
    train_x_ngram = tfidf_vect_ngram.transform(train_x)
    valid_x_ngram = tfidf_vect_ngram.transform(valid_x)
    with open('models/ngram_vectorizer.pk', 'wb') as fin:
        pickle.dump(tfidf_vect_ngram, fin)
    return (train_x_ngram, valid_x_ngram)


def tokenize_text(train, test, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    texts = train + test

    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    xtrain = tokenizer.texts_to_sequences(train)
    xtrain = pad_sequences(xtrain, maxlen=MAX_SEQUENCE_LENGTH)

    xtest = tokenizer.texts_to_sequences(test)
    xtest = pad_sequences(xtest, maxlen=MAX_SEQUENCE_LENGTH)
    with open('models/tokenize_vectorizer.pk', 'wb') as fin:
        pickle.dump(tokenizer, fin)

    return xtrain, xtest


def data_processor(train_x, valid_x, train_col, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    processed_data = dict()

    ngram_vector = ngram_vectors(train_x=train_x,
                                 valid_x=valid_x,
                                 train_col=train_col)
                                 
    processed_data["RFC"] = ngram_vector
    processed_data["ADA"] = ngram_vector
    processed_data["XGBC"] = ngram_vector
    processed_data["BG"] = ngram_vector
    processed_data["NB"] = ngram_vector
    processed_data["SVC"] = ngram_vector

    tokenized_sequence = tokenize_text(train_x.to_numpy().tolist(),
                                       valid_x.to_numpy().tolist(),
                                       MAX_NB_WORDS,
                                       MAX_SEQUENCE_LENGTH)

    processed_data["RCNN"] = tokenized_sequence
    processed_data["LSTM"] = tokenized_sequence

    return processed_data


def get_processed_dataset_dict(train_col,
                               valid_col,
                               filename,
                               MAX_NB_WORDS,
                               MAX_SEQUENCE_LENGTH):
    # Import dataframe from csv
    cols = [train_col, valid_col]
    df = get_dataframe(col_list=cols, filename=filename)

    # Split dataset into training and validation sets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        df[train_col], df[valid_col], train_size=0.7, test_size=0.3, stratify=df[valid_col])

    # Resampling the training data
    rus = RandomUnderSampler(0.1)
    ros = RandomOverSampler(0.15)

    train_x = train_x.to_numpy().reshape(-1, 1)
    train_y = train_y.to_numpy().reshape(-1, 1)

    train_x, train_y = rus.fit_sample(train_x, train_y)
    train_x, train_y = ros.fit_sample(train_x, train_y)

    train_x = pandas.Series(train_x.reshape(-1,))
    train_y = pandas.Series(train_y.reshape(-1,))
    print(train_y.value_counts())

    return (data_processor(train_x=train_x,
                           valid_x=valid_x,
                           train_col=df[train_col],
                           MAX_NB_WORDS=MAX_NB_WORDS,
                           MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH),
            train_y,
            valid_y)
