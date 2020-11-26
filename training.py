from sklearn import metrics
import numpy as np
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
        output = classifier.predict(valid_x)
        results = classifier.evaluate(valid_x, valid_y)
        output = np.argmax(output, axis=1)
        return results, output
    else:
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(valid_x)

        pickle.dump(classifier, open('models/' + name + ".pickle", "wb"))
        #cm_cv = metrics.confusion_matrix(valid_y, predictions)
        #cm_cv = pd.DataFrame(cm_cv, index=[0, 1], columns=[0, 1])
        #cm_cv.index.name = 'Actual'
        #cm_cv.columns.name = 'Predicted'
        #plt.figure(figsize=(10, 10))
        #sns.heatmap(cm_cv, cmap="Blues", annot=True, fmt='')
        #plt.show()
        return [metrics.accuracy_score(valid_y, predictions),
                metrics.precision_score(valid_y, predictions),
                metrics.recall_score(valid_y, predictions),
                metrics.f1_score(valid_y, predictions, pos_label=1)], predictions

