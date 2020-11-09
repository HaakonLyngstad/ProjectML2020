from sklearn import metrics
import pickle

def train_model(classifier, name, train_x, train_y, valid_x, valid_y):
    pass
    # fit the training dataset on the classifier
    if name in ["RCNN", "LSTM"]:
        classifier.fit(train_x, train_y)
        classifier.model.save('models/' + name)
        # pickle.dump(classifier, open('models/' + name + ".pickle", "wb"))
        return classifier.evaluate(valid_x, valid_y)
    else:
        classifier.fit(train_x, train_y)
        # predict the labels on validation dataset
        predictions = classifier.predict(valid_x)
        pickle.dump(classifier, open('models/' + name + ".pickle", "wb"))
        return [metrics.accuracy_score(valid_y, predictions),
                metrics.precision_score(valid_y, predictions),
                metrics.recall_score(valid_y, predictions)]