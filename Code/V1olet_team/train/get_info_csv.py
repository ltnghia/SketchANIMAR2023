import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visual_df_metric(df, metric):
    val_metric = 'val_' + metric

    print(f'MIN {val_metric}:', np.min(df[val_metric]), 'at epoch:', np.argmin(df[val_metric]) + 1)
    print(f'MAX {val_metric}:', np.max(df[val_metric]), 'at epoch:', np.argmax(df[val_metric]) + 1)

    plt.figure()
    plt.plot(df[metric], label=f'train {metric}')
    plt.plot(df[val_metric], label=f'test {metric}')
    plt.title(f'Plot History: Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend([f'Train {metric}', f'Test {metric}'], loc='upper left')
    plt.show()
    plt.savefig(f'df_plot_{metric}.png')

csv_path = 'training.csv'

df = pd.read_csv(csv_path)

visual_df_metric(df, "loss")

visual_df_metric(df, "accuracy")
