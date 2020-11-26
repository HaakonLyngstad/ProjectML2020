import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('models/metrics.csv')
print(type(df1))
#df2 = pd.read_csv('models/stacking_metrics.csv')
print(df1)

df1.plot(x='Classifier', y='Precision', kind='bar')
df1.plot(x='Classifier', y='Recall', kind='bar')

plt.show()
