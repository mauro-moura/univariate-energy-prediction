
def get_data(out_name: str):
    from load_data import load_test_data, get_normalizer

    if (out_name == 'UCI'):
        train_path = './Datasets_LABIC_UCI/ds_trein_uci.csv'
        test_path = './Datasets_LABIC_UCI/ds_test_uci.csv'
    elif (out_name == 'Labic'):
        train_path = './Datasets_LABIC_UCI/prev_trein_2020.csv'
        test_path = './Datasets_LABIC_UCI/prev_test_2020.csv'
    
    X_test, y_test = load_test_data(train_path, test_path)

    normalizer = get_normalizer(train_path, test_path)

    return X_test, y_test, normalizer

def get_model(model_name = ".model"):
    import xgboost as xgb
    
    cpu_dict = {
        'objective': 'reg:squarederror'
    }

    gpu_dict = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist'
    }
    n_estimator = 1000
    reg = xgb.XGBRegressor(n_estimators=n_estimator, **cpu_dict)
    reg.load_model(model_name)

    return reg

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X_test, y_test, normalizer = get_data("Labic")

    reg = get_model("./outputs/GPU/1000/xboost_model_0.model")
    predictions = reg.predict(X_test)
    predictions = normalizer.inverse_transform(predictions.reshape(-1, 1))

    for i in range(len(predictions)):
        print("Esperado: ", y_test[i], "Predito: ", predictions[i])

    plt.plot(y_test)
    plt.plot(predictions)
    plt.legend(["Expected", "Predicted"])
    plt.show()
