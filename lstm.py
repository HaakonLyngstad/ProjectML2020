from nltk.corpus import stopwords
import re
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping
from tokenize_text import tokenize_text
from keras.metrics import Precision, Recall


df = pd.read_csv('Combined_News_DJIA.csv')

train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']

print(f"Train size = {len(train)/len(df['Label'])}%, Test size = {len(test)/len(df['Label'])}%")

# This is fixed.
EMBEDDING_DIM = 16
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 250


def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()

    # specific for this dataset
    text = re.compile("b'").sub('', text)
    text = re.compile('b"').sub('', text)

    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


trainheadlines = []
for i in range(len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[i, 2:27]))
    trainheadlines[i] = clean_text(trainheadlines[i])

testheadlines = []
for i in range(len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[i, 2:27]))
    testheadlines[i] = clean_text(testheadlines[i])

#print(f"Length of train set: {len(trainheadlines)}")
#print(f"Length of test set: {len(testheadlines)}")
#print(f"Total length: {len(data['Label'])}")
#print(f"Train length: {len(train)}")
#print(f"Test length: {len(test)}")


xtrain, xtest = tokenize_text(trainheadlines, testheadlines, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
ytrain = train['Label']
ytest = test['Label']


#print(xtrain.shape, ytrain.shape)
#print(xtest.shape, ytest.shape)

print(xtrain[2])

model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=xtrain.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy', Precision(), Recall()])

model.summary()

epochs = 10
batch_size = 64

history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.1)