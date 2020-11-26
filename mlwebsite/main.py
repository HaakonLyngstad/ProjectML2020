from data_processing import get_processed_dataset_dict
from training import train_model
import stacking
from lstm import LSTM_model
from rcnn import RCNN_model
from sklearn import (
    svm,
    tree,
)
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import xgboost
import pandas
import os
import shutil
from matplotlib import pyplot as plt

train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 2000

# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500

EMBEDDING_DIM_RCNN = 16
RCNN_EPOCHS = 10
RCNN_BATCH_SIZE = 128

EMBEDDING_DIM_LSTM = 16

LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 128

print("------- Initiating data processing ------")
processed_data, train_y, valid_y = get_processed_dataset_dict(
    train_col=train_col,
    valid_col=valid_col,
    filename=filename,
    MAX_NB_WORDS=MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

dct = tree.DecisionTreeClassifier(max_depth=1)
print("------- Loading classifiers ------")
classifier_list = [
    ensemble.AdaBoostClassifier(dct, n_estimators=350, learning_rate=0.5),
    ensemble.BaggingClassifier(svm.SVC(C=10, gamma=1), n_estimators=100, n_jobs=-1),
    ensemble.RandomForestClassifier(),
    xgboost.XGBClassifier(max_depth=10,
                          min_child_weight=1,
                          scale_pos_weight=2),
    LSTM_model(input_length=MAX_SEQUENCE_LENGTH,
               EMBEDDING_DIM=EMBEDDING_DIM_LSTM,
               MAX_NB_WORDS=MAX_NB_WORDS,
               MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
               EPOCH_SIZE=LSTM_EPOCHS,
               BATCH_SIZE=LSTM_BATCH_SIZE),
    RCNN_model(input_length=MAX_SEQUENCE_LENGTH,
               EMBEDDING_DIM=EMBEDDING_DIM_RCNN,
               MAX_NB_WORDS=MAX_NB_WORDS,
               EPOCH_SIZE=RCNN_EPOCHS,
               MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
               BATCH_SIZE=RCNN_BATCH_SIZE)
]

classifier_names = ["ADA", "BG", "RFC", "XGBC", "LSTM", "RCNN"]

dir = 'models'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

print("------- Initiating training -------")
results_df = pandas.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score"])
for clfl, clfn in zip(classifier_list, classifier_names):
    (train_x, valid_x) = processed_data[clfn]
    print(f"Training {clfn}:")
    metrics, _ = train_model(
        classifier=clfl,
        name=clfn,
        train_x=train_x,
        valid_x=valid_x,
        train_y=train_y,
        valid_y=valid_y
    )
    results_df.loc[len(results_df)] = [clfn] + metrics
    print([clfn] + metrics)

print(results_df)
results_df.to_csv("models/metrics.csv", index=False)

plot = results_df.plot.bar(x='Classifier', rot=0, title='Classifiers', figsize=(20, 10), fontsize=14)
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.show()
