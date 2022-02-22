import os
import time

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import xgboost as xgb
#from xgboost import plot_importance, plot_tree

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

time1 = time.time()

def create_folder(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Diretorio " , dirName ,  " Criado ")
    else:    
        print("Diretorio " , dirName ,  " ja existe")

out_path_orig = './outputs/GPU/'
out_name = 'Labic' # UCI || Labic

out_path_orig += out_name + '/'

if (out_name == 'UCI'):
  train_path = './Datasets_LABIC_UCI/ds_trein_uci.csv'
  test_path = './Datasets_LABIC_UCI/ds_test_uci.csv'
elif (out_name == 'Labic'):
  train_path = './Datasets_LABIC_UCI/prev_trein_2020.csv'
  test_path = './Datasets_LABIC_UCI/prev_test_2020.csv'

cpu_dict = {
    'objective': 'reg:squarederror'
}

gpu_dict = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist'
}

ds_train = pd.read_csv(train_path, sep=",", parse_dates=True)
ds_test = pd.read_csv(test_path, sep=",", parse_dates=True)

rows_train = len(ds_train.index) - 4
rows_test = len(ds_test.index) - 4

base = pd.read_csv(train_path)
base = base.dropna()
base_train = base.iloc[:, 1:2].values

normalizer = MinMaxScaler(feature_range=(0,1))
base_train_norm = normalizer.fit_transform(base_train)

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

n_estimators = [500, 1000, 5000, 30000]

kfold = TimeSeriesSplit(max_train_size=None, n_splits=10) # , test_size=2, gap=2)

for n_estimator in n_estimators:
    n_exec = 0

    rmse = []
    mae = []
    mape = []
    r2 = []

    out_path = out_path_orig + '%s/'%n_estimator
    create_folder(out_path)
    for train_index, test_index in kfold.split(inputs):
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = outputs[train_index], outputs[test_index]
        
        print("Executando Teste", n_exec)
        reg = xgb.XGBRegressor(n_estimators=n_estimator, **gpu_dict)
        
        reg.fit(X_train, y_train)
    
        reg.save_model(out_path + 'xboost_model_%i.model'%n_exec)
    
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
        
        predictions = reg.predict(X_test)
        
        predictions = normalizer.inverse_transform(predictions.reshape(-1, 1))
        
        #y_test = normalizer.inverse_transform(y_test.reshape(-1, 1))
        
        time2 = time.time()
        exec_time = time2 - time1
        
        print("Tempo de Execução", round(exec_time,3), "segundos")
        
        rmse.append(mean_squared_error(y_true=y_test, y_pred=predictions, squared=False))
        mae.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        mape.append(mean_absolute_percentage_error(y_true=y_test, y_pred=predictions))
        r2.append(r2_score(y_true=y_test, y_pred=predictions))

        d = {
             "RMSE": rmse[n_exec],
             "MAE": mae[n_exec],
             "MAPE": mape[n_exec],
             "R2": r2[n_exec],
             "Tempo": exec_time
             }
        
        with open('%sresults_%s_%s.txt'%(out_path, out_name, str(n_exec)), 'w') as f:
            f.write(str(d))
        
        np.savetxt('%spredictions_%s_%s.txt'%(out_path, out_name, str(n_exec)), predictions)
        
        for i in range(len(predictions)):
          print("Esperado: ", y_test[i], "Predito: ", predictions[i])
        
        print(d)

        n_exec += 1
      
    d = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            }
            
    df = pd.DataFrame(d)
    df.to_excel(out_path + 'all_metrics.xlsx')
