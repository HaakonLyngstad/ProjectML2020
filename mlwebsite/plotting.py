import matplotlib.pyplot as plt
import pandas as pd

columns=['a', 'b', 'c', 'd']
df = pd.read_csv('models/stacking_metrics.csv')

#df1.plot(x='Classifier', y='Precision', kind='bar')
#df1.plot(x='Classifier', y='Recall', kind='bar')

df.plot.bar(x='Classifier', rot=0, title='Classifiers', figsize=(20, 10), fontsize=14)

plt.show()