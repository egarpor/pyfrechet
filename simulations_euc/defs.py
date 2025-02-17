import multiprocessing as mp
import numpy as np
import pandas as pd
import os

def process_file(file, results_dir):
    if file.endswith('.npy'):
        infile = open(os.path.join(results_dir, file), 'rb')
        result = np.load(infile, allow_pickle=True).item()
        infile.close()
        print(int(file.split('_')[1][4:]))
        return pd.DataFrame({
            'sample_index': int(file.split('_')[1][4:]),
            'train_size': int(file.split('_')[2][1:]),
            'sigma': file.split('_')[3][5:],
            'y_data': [result['y_data']],
            'x_data': [result['x_data']],
            'forest': [result['forest']],
        }, index=pd.RangeIndex(0, 1))
    else:
        return None


def compute_oob_errors(row):
    return row['forest'].oob_errors()

def compute_oob_quantile(row, sign_level):
    return np.percentile(row['OOB_errors'], (1 - sign_level) * 100, method='inverted_cdf')

