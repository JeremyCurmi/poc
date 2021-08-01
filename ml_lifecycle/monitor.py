import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

df_results = pd.DataFrame(columns=['actual','prediction'])

for i in range(1, 1000):
    try:
        r = requests.get(f'http://0.0.0.0:8002/validate/{i}')
        df_results = df_results.append(eval(r.text), ignore_index=True)
    except Exception as err:
        print(f"index {i} could not be computed!")
    cm = confusion_matrix(list(df_results['actual']), list(df_results['prediction']))
    accuracy = accuracy_score(list(df_results['actual']), list(df_results['prediction']))
    sns.heatmap(cm, annot=True)
    plt.title(f'Accuracy Score: {round(accuracy*100,2)}% on {i} samples')
    plt.show(block=False)
    plt.pause(0.0001)
    plt.close()
