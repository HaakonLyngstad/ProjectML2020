import pandas
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer


def import_dataset(filename):
    col_list = ["company_profile", "description", "requirements", "fraudulent"]
    df = pandas.read_csv(filename, usecols=col_list)
    return df


def init_stopword():
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    return stop


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_url(text):
    return re.sub(r'http\S+', '', text)


# Remove stopwords
def remove_stopwords(text, stop):
    text_adapted = []
    for i in text.split():
        if i.strip().lower() not in stop:
            text_adapted.append(i.strip())
    return " ".join(text_adapted)


# Collection function for text denoising
def remove_noise(text, stop):
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    return remove_stopwords(text, stop)


def switch_num(valid):
    if valid == 1:
        return 0
    else:
        return 1


# ngrams function for data visualization
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def vizualise_data(df, text_col, valid_col):

    print((df[valid_col] == 1).sum())
    sns.set_style("dark")
    countplot = sns.countplot(df[valid_col])
    bar1 = countplot.get_figure()
    bar1.savefig('images/valid_bar.eps', format='eps')
    plt.show()
    print("---------------- NaN rows ----------------")
    print(df.isna().sum())
   
    print("\n\n---------------- Count ----------------")
    print(df[text_col].count())

    fig1 = plt.figure(figsize=(20, 20))  # Text that is not Fake
    wc = WordCloud(max_words=2000, width=1000, height=600, stopwords=STOPWORDS).generate(" ".join(df[df[valid_col] == 1][text_col]))
    plt.imshow(wc, interpolation='bilinear')
    plt.grid(False)
    fig1.savefig('images/real_wordcloud.eps', format='eps')
    plt.show()

    fig2 = plt.figure(figsize=(20, 20))  # Text that Fake
    wc = WordCloud(max_words=2000, width=1000, height=600, stopwords=STOPWORDS).generate(" ".join(df[df[valid_col] == 0][text_col]))
    plt.imshow(wc, interpolation='bilinear')
    plt.grid(False)
    fig2.savefig('images/fake_wordcloud.eps', format='eps')
    plt.show()

    bar2 = plt.figure(figsize=(16, 9))
    most_common_bi = get_top_text_ngrams(df["text"], 10, 3)
    most_common_bi = dict(most_common_bi)
    sns.barplot(x=list(most_common_bi.values()), y=list(most_common_bi.keys()))
    bar2.savefig('images/ngram_bar.eps', format='eps')
    plt.show()


df = import_dataset('fake_job_postings.csv')

train_col = "text"
valid_col = "fake"

columns = [train_col, valid_col]
index = range(0, len(df))

df_adapted = pandas.DataFrame(index=index, columns=columns)

for index, row in df.iterrows():
    row = row.copy()
    new_text = str(row["company_profile"]) + str(row["description"]) + str(row["requirements"])
    df_adapted.loc[index, train_col] = new_text
    df_adapted.loc[index, valid_col] = row["fraudulent"]

stop = init_stopword()
df_adapted[train_col] = df_adapted.apply(lambda x: remove_noise(x[train_col], stop), axis=1)
df_adapted = df_adapted.drop_duplicates(subset=[train_col], keep=False)

df_adapted[valid_col] = df_adapted.apply(lambda x: switch_num(x[valid_col]), axis=1)

df_adapted.to_csv('fake_job_postings_processed_switched.csv', index=False)

vizualise_data(df_adapted, train_col, valid_col)
