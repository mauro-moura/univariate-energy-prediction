import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def get_base(train_path, test_path):
    ds_train = pd.read_csv(train_path, sep=",", parse_dates=True)
    ds_test = pd.read_csv(test_path, sep=",", parse_dates=True)

    rows_train = len(ds_train.index) - 4
    rows_test = len(ds_test.index) - 4

    base = pd.read_csv(train_path)
    base = base.dropna()

    base_train = base.iloc[:, 1:2].values
    
    normalizer = MinMaxScaler(feature_range=(0,1))
    base_train_norm = normalizer.fit_transform(base_train)

    return base, rows_train, rows_test, normalizer, base_train_norm

def load_train_data(train_path, test_path):
    _, rows_train, _, _, base_train_norm = get_base(train_path, test_path)

    inputs = []
    outputs = []

    # Slide Window = 90
    for i in range(90, rows_train):
        inputs.append(base_train_norm[i-90:i, 0])
        outputs.append(base_train_norm[i,0])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputs = np.reshape(inputs,
                        (inputs.shape[0], inputs.shape[1]))
    return inputs, outputs

def load_test_data(train_path, test_path):
    base, _, rows_test, normalizer, _ = get_base(train_path, test_path)

    base_test = pd.read_csv(test_path)
    all_data = pd.concat((base['potencia'], base_test['potencia']), axis=0)
    #y_test = base_test.iloc[90:-4,1:2].values
    base_test_norm = all_data[len(all_data) - len(base_test) - 90:].values
    base_test_norm = base_test_norm.reshape(-1, 1)
    base_test_norm = normalizer.transform(base_test_norm)
    
    X_test = []
    y_test = []
    for i in range(90, rows_test):
        X_test.append(base_test_norm[i-90:i, 0])
        y_test.append(base_test_norm[i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    
    y_test = np.array(y_test)
    y_test = normalizer.inverse_transform(y_test.reshape(-1, 1))

    return X_test, y_test

def get_normalizer(train_path, test_path):
    _, _, _, normalizer, _ = get_base(train_path, test_path)
    return normalizer
