from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
