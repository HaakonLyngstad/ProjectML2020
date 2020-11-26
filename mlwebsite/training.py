from sklearn import metrics
import pickle


def train_model(classifier, name, train_x, train_y, valid_x, valid_y):
    """
    Fits each machine learning model to the training data, then saves the model
    and returns the metrics for each validation. This is split based on keras
    and sci-kit models.
    """
    if name in ["RCNN", "LSTM"]:
        classifier.fit(train_x, train_y)

        classifier.model.save('models/' + name)

        return classifier.evaluate(valid_x, valid_y)
    else:
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(valid_x)

        pickle.dump(classifier, open('models/' + name + ".pickle", "wb"))

        return [metrics.accuracy_score(valid_y, predictions),
                metrics.precision_score(valid_y, predictions, pos_label=1),
                metrics.recall_score(valid_y, predictions, pos_label=1),
                metrics.f1_score(valid_y, predictions, pos_label=1)]
