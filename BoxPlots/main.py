# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:14:38 2022

@author: mauro
"""

def plot_boxplot(input_data, fig_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams.update({'font.size' : 10})
    axis_font = {'fontname' : 'Arial', 'size' : '16'}
    
    sns.boxplot(data=input_data)
    #plt.ylim(0,22)
    plt.xlabel("Case", **axis_font)
    plt.ylabel("RMSE", **axis_font)
    plt.savefig(fig_name)
    plt.show()

if __name__ == '__main__':
    import glob
    
    import numpy as np
    import pandas as pd
    
    #file_path = glob.glob('./data/[!_]*.xlsx')
    file_path = glob.glob('./data/*.xlsx')
    
    df = pd.read_excel(file_path[0])
    
    # XGBoost
    
    xgboost_labic = df.values[1:,:1].reshape(-1).astype(np.float32)
    xgboost_uci = df.values[1:,1:2].reshape(-1).astype(np.float32)
    xgboost_dincon = df.values[1:,2:3].reshape(-1).astype(np.float32)
    
    # Random Forest
    
    rf_labic = df.values[1:,3:4].reshape(-1).astype(np.float32)
    rf_uci = df.values[1:,4:5].reshape(-1).astype(np.float32)
    rf_dincon = df.values[1:,5:6].reshape(-1).astype(np.float32)
    
    # LSTM
    
    lstm_labic = df.values[1:,6:7].reshape(-1).astype(np.float32)
    lstm_uci = df.values[1:,7:8].reshape(-1).astype(np.float32)
    lstm_dincon = df.values[1:,8:9].reshape(-1).astype(np.float32)
    
    # Labic
    
    labic = pd.DataFrame()
    labic['LSTM'] = lstm_labic
    labic['XGBoost'] = xgboost_labic
    labic['RandomForest'] = rf_labic
    
    plot_boxplot(labic, './figs/labic_dataset.png')
    
    # UCI
    
    uci = pd.DataFrame()
    uci['LSTM'] = lstm_uci
    uci['XGBoost'] = xgboost_uci
    uci['RandomForest'] = rf_uci
    
    plot_boxplot(uci, './figs/uci_dataset.png')
    
    # Load
    
    dincon = pd.DataFrame()
    dincon['LSTM'] = lstm_dincon
    dincon['XGBoost'] = xgboost_dincon
    dincon['RandomForest'] = rf_dincon
    
    plot_boxplot(dincon, './figs/dincon_dataset.png')
