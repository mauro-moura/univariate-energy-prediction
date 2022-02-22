
def handle_data(data_files):
    from sklearn.preprocessing import MinMaxScaler
    
    data = []
    
    for file in data_files:
        df = pd.read_csv(file)
        data +=  list(df.values)
    
    data = np.asarray(data)
    demand = data[:-48, 4]
    
    normalizer = MinMaxScaler(feature_range=(0,1))
    data_norm = normalizer.fit_transform(demand.reshape(-1, 1))
    return data_norm, normalizer

def slide_window(data_norm):
    X = []
    y = []
    
    # Slide Window = 90
    for i in range(90, data_norm.shape[0]):
        X.append(data_norm[i-90:i, 0])
        y.append(data_norm[i,0])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def get_model_xgboost(model_name):
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

def get_model_lstm(model_name):
    n_estimator = 1000
    reg = xgb.XGBRegressor(n_estimators=n_estimator, **cpu_dict)
    reg.load_model(model_name)
    return

def get_model_random_forest(model_name):
    n_estimator = 1000
    reg = xgb.XGBRegressor(n_estimators=n_estimator, **cpu_dict)
    reg.load_model(model_name)
    return

def plot_data(plot_values):   
    for value in plot_values:
        #plt.plot(plot_value[-350:])
        plt.plot(value)
        #plt.ylim((0, 1))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    import glob
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    
    files_path = glob.glob('./data/*.csv')
    data_norm, normalizer = handle_data(files_path)

    X_test, y_test = slide_window(data_norm)
    y_test = normalizer.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = []
    mae = []
    mape = []
    r2 = []
    
    model_name = 'xgboost' # lstm || rf || xgboost
    #model_number = 0
    model_path = glob.glob('./models/%s/*'%(model_name))
    
    out_path = './outputs/%s/'%model_name
    
    for model_number in range(len(model_path)):
        reg = get_model_xgboost(model_path[model_number])
        
        predictions = reg.predict(X_test)
        predictions = normalizer.inverse_transform(predictions.reshape(-1, 1))
        
        rmse.append(mean_squared_error(y_true=y_test, y_pred=predictions, squared=False))
        mae.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        mape.append(mean_absolute_percentage_error(y_true=y_test, y_pred=predictions))
        r2.append(r2_score(y_true=y_test, y_pred=predictions))
        
        np.savetxt(out_path + 'prediction_%s_%i.txt'%(model_name, model_number), predictions)
        
        plt.plot(y_test)
        plt.plot(predictions)
        plt.legend(["Expected", "Predicted"])
        plt.show()
    
    d = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            }
            
    df = pd.DataFrame(d)
    df.to_excel(out_path + 'all_metrics_%s.xlsx'%(model_name))
