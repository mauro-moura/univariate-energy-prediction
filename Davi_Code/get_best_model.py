# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:47:12 2022

@author: mauro
"""

def find_min_rmse(out_name: str, n_estimators: list):
    import numpy as np
    rmse_mean = []
    
    for n_estimator in n_estimators:
        rmse_list = []
        for i in range(10):
            filename = './outputs/GPU/%s/%s/results_%s_%i.txt'%(out_name, n_estimator, out_name, i)
            
            with open(filename, 'r') as f:
                results = eval(f.read())
            
            rmse_list.append(results['RMSE'])
        
        rmse_mean.append(np.mean(rmse_list))
    
    print(rmse_mean)
    return rmse_mean.index(min(rmse_mean))
    
if __name__ == '__main__':
    out_name = 'Labic' # UCI || Labic
    n_estimators = [500, 1000, 5000, 30000]
    
    print(find_min_rmse(out_name, n_estimators))

