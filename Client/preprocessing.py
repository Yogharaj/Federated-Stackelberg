import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(classname, base_path):
    x_train = np.load(f"{base_path}/{classname}.csv_Xtrain.npy")
    y_train = np.load(f"{base_path}/{classname}.csv_Ytrain.npy")
    x_test = np.load(f"{base_path}/{classname}.csv_Xtest.npy")
    y_test = np.load(f"{base_path}/{classname}.csv_Ytest.npy")
    min_test_size = min(x_test.shape[0], y_test.shape[0])
    x_test = x_test[:min_test_size]
    y_test = y_test[:min_test_size]

    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, x_test):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
