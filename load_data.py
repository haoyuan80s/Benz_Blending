"""
Functions to load the dataset.
"""

import numpy as np
from sklearn.decomposition import PCA, FastICA
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data(file_name):
    """This function is adapted from:
    https://github.com/benhamner/BioResponse/blob/master/Benchmarks/csv_io.py
    """
    f = open(file_name)
    # skip header
    f.readline()
    samples = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples


def load():
    """Conveninence function to load all data as numpy arrays.
    """
    print "Loading data..."
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # process columns, apply LabelEncoder to categorical features
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder() 
            lbl.fit(list(train[c].values) + list(test[c].values)) 
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))

    
    n_comp = 10

    # PCA
    pca = PCA(n_components=n_comp, random_state=42)
    pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=42)
    ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp+1):
        train['pca_' + str(i)] = pca2_results_train[:,i-1]
        test['pca_' + str(i)] = pca2_results_test[:, i-1]

        train['ica_' + str(i)] = ica2_results_train[:,i-1]
        test['ica_' + str(i)] = ica2_results_test[:, i-1]

    y_train = train["y"]
    X_train = train.drop('y', axis=1)
    X_test = test
    # shape
    print('Shape train: {}\nShape test: {}'.format(X_train.shape, X_test.shape))
    return np.array(X_train), np.array(y_train), np.array(X_test)

if __name__ == '__main__':

    X_train, y_train, X_test = load()
