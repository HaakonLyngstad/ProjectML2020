from sklearn import metrics
import numpy as np
import pickle


def train_model(classifier, name, train_x, train_y, valid_x, valid_y):
    if name in ["RCNN", "LSTM"]:
        classifier.fit(train_x, train_y)
        classifier.model.save('models/' + name)
        output = classifier.predict(valid_x)
        results = classifier.evaluate(valid_x, valid_y)
        output = np.argmax(output, axis=1)
        return results, output
    else:
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(valid_x)
        pickle.dump(classifier, open('models/' + name + ".pickle", "wb"))
        return [metrics.accuracy_score(valid_y, predictions),
                metrics.precision_score(valid_y, predictions),
                metrics.recall_score(valid_y, predictions)], predictions
