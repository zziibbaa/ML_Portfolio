import os
import numpy as np
import pandas as pd

def load_data(path='../data/processed.cleveland.data'):
    """
    Load the Cleveland Heart Disease dataset,
    remove missing values and return a clean DataFrame.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, path)

    data = pd.read_csv(full_path, sep=',', header=None)

    data.columns = [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','DISEASE'
    ]

    data = data.replace('?', np.nan)
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)

    return data