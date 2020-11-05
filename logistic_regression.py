import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

# extract one article, lowercase, vectorize, count
""" example = train.iloc[3,10]
example = example.lower()
example = CountVectorizer().build_tokenizer()(example)
print(example)
ex = pd.DataFrame([[x, example.count(x)] for x in set(example)], columns= ['Word', 'Count'])
print(ex) """


# creating a matrix of headlines
trainheadlines = []
for row in range(0, len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
#print(basictrain.shape)

basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])

testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)

basicresults = pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
print(basicresults)
accuracy = accuracy_score(test["Label"], predictions)
print(accuracy)
