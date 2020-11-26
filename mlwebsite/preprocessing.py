import pandas
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer


def init_stopword():
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    return stop


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_url(text):
    text = re.sub(r'#URL+', '', text)
    return re.sub(r'http\S+', '', text)


def remove_special_characters(text):
    return re.sub('[^A-Za-z0-9]+', ' ', text)


def remove_multiple_spaces(text):
    return re.sub(r"\s\s+", " ", text) 


# Remove stopwords
def remove_stopwords(text, stop):
    """
    Removel all stopwords from a text string (it, the, etc.),
    i.e. words that dont contibute meaning to the machine learning algorithms.
    Based on a stopword list from the nltk package.
    """
    text_adapted = []
    for i in text.split():
        if i.strip().lower() not in stop:
            text_adapted.append(i.strip())
    return " ".join(text_adapted)


# Collection function for text denoising
def remove_noise(text, stop):
    """
    Noise removal function. A collection function to perform all functions 
    declared above.
    """
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_special_characters(text)
    text = remove_multiple_spaces(text)
    text = re.sub("nbsp", "", text)
    text = re.sub("amp", "", text)
    text = re.sub("'", "", text)
    text = re.sub("nan", "", text)
    return remove_stopwords(text, stop)


def import_dataset(filename):
    """
    Imports the preset columns of our datasat into a dataframe.
    """
    col_list = ["company_profile", "description", "requirements", "fraudulent"]
    df = pandas.read_csv(filename, usecols=col_list)
    return df


# ngrams function for data visualization
def get_top_text_ngrams(corpus, n, g):
    """
    Finds the n-grams for a given corpus
    """
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def vizualise_data(df, text_col, valid_col):
    """
    Function for creating visualizations of the processed dataset.
    Creates a bar graph plotting fradulent and real data,
    separete word clouds for the real and fradulent data 
    and bigrams for the real and fradulent data.
    """
    sns.set_style("white")
    countplot = sns.countplot(df[valid_col])
    bar1 = countplot.get_figure()
    bar1.savefig('images/valid_bar.pdf', format='pdf')
    plt.show()

    sns.set_style("dark")
    fig1 = plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=100, width=1000, height=600, stopwords=STOPWORDS, background_color="white").generate(" ".join(df[df[valid_col] == 1][text_col]))
    plt.imshow(wc, interpolation='bilinear')
    plt.grid(False)
    fig1.savefig('images/real_wordcloud.pdf', format='pdf')
    plt.show()

    fig2 = plt.figure(figsize=(20, 20))  # Text that Fake
    wc = WordCloud(max_words=100, width=1000, height=600, stopwords=STOPWORDS, background_color="white").generate(" ".join(df[df[valid_col] == 0][text_col]))
    plt.grid(False)
    plt.imshow(wc, interpolation='bilinear')
    fig2.savefig('images/fake_wordcloud.pdf', format='pdf')
    plt.show()

    plt.rcParams.update({'font.size': 14})

    bar2 = plt.figure(figsize=(16, 9))
    most_common_bi = get_top_text_ngrams(df[df[valid_col] == 0][text_col], 10, 3)
    most_common_bi = dict(most_common_bi)
    sns.barplot(x=list(most_common_bi.values()), y=list(most_common_bi.keys()))
    plt.yticks(rotation=45)
    bar2.savefig('images/ngram_bar_real.pdf', format='pdf')
    plt.show()

    bar3 = plt.figure(figsize=(16, 9))
    most_common_bi = get_top_text_ngrams(df[df[valid_col] == 1][text_col], 10, 3)
    most_common_bi = dict(most_common_bi)
    sns.barplot(x=list(most_common_bi.values()), y=list(most_common_bi.keys()))
    plt.yticks(rotation=45)
    bar3.savefig('images/ngram_bar_fake.pdf', format='pdf')
    plt.show()


def preprocessing():
    """
    General function for preprocessing of data.
    The function:

    -loads the original dataset
    -Removes duplicates
    -Adds the three desired columns from the original data set into a single column
        in a new dataframe
    -Removes noise such as stopwords
    -Saves the new dataframe to a csv formated file
    -Visualizes the dataset

    Variables:
    - df: old dataframe
    - train_col: name of coloumn containing text data in new datframe
    - valid_col: name of column containing binary information in new dataframe
    - df_adapted: New dataframe with processed data
    """

    df = import_dataset('fake_job_postings.csv')

    train_col = "text"
    valid_col = "fake"

    df = df.drop_duplicates(subset=["description", "requirements"],
                            keep="first").reset_index(drop=True)

    columns = [train_col, valid_col]
    index = range(0, len(df))

    df_adapted = pandas.DataFrame(index=index, columns=columns)

    for index, row in df.iterrows():
        row = row.copy()
        new_text = str(row["company_profile"]) + str(row["description"]) + str(row["requirements"])
        df_adapted.loc[index, train_col] = new_text
        df_adapted.loc[index, valid_col] = row["fraudulent"]

    df_adapted = df_adapted.drop_duplicates(subset=[train_col], keep="first").reset_index(drop=True)

    stop = init_stopword()
    df_adapted[train_col] = df_adapted.apply(lambda x: remove_noise(x[train_col], stop), axis=1)

    df_adapted.to_csv('fake_job_postings_processed_switched.csv', index=False)
    vizualise_data(df_adapted, train_col, valid_col)


if __name__ == "__main__":
    preprocessing()
