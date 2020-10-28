from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string, csv
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

def import_dataset(filename):
    file = open(filename, encoding="utf-8")
    data = list(csv.reader(file, skipinitialspace=True))
    col_list = ["title", "text"]
    df = pandas.read_csv(filename, usecols=col_list, encoding="utf-8")
    return df
    

def get_dataframe():
    trueDf = import_dataset('True.csv')
    trueDf['class'] = 1
    fakeDf = import_dataset('Fake.csv')
    fakeDf['class'] = 0

    return pandas.concat([trueDf, fakeDf])

print(get_dataframe())