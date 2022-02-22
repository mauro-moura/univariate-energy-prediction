#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
ini=time.time()

train_path = './Datasets_LABIC_UCI/prev_trein_2020.csv'
test_path = './Datasets_LABIC_UCI/prev_test_2020.csv'

with open('result.txt', 'w') as f:
    #quantidade de linhas do arquivo
    ds_trein = pd.read_csv(train_path, sep=',', parse_dates = True)
    ds_test = pd.read_csv(test_path, sep=',', parse_dates = True)
    rows_trein = len(ds_trein.index);  rows_trein = rows_trein - 4
    rows_test = len(ds_test.index); rows_test = rows_test - 4

    base = pd.read_csv(train_path)
    base = base.dropna() #exclui linhas no dataframe
    base_treinamento = base.iloc[:, 1:2].values

    #Norm Proporc ( 0 ou 1) (Durante o treino evita grandes saltos no gradiente)
    #Facilita o aprendizado por ser escala enrte 0 e 1
    normalizador = MinMaxScaler(feature_range=(0,1))
    base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

    #print(base_treinamento_normalizada)
    previsores = []
    valor_real = []
    #slide window
    #90 é o tamanho da janela deslizante
    for i in range(90,rows_trein):
        previsores.append(base_treinamento_normalizada[i-90:i,0])
        valor_real.append(base_treinamento_normalizada[i,0])
    previsores, valor_real = np.array(previsores), np.array(valor_real)
    previsores = np.reshape(previsores, 
                (previsores.shape[0],previsores.shape[1],1)) # Redimens. do Keras

    model = Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(previsores.shape[1],1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='rmsprop',loss='mean_squared_error',
                metrics=['mean_absolute_error'])
    #rmse = sqrt(mean_squared_error) --- 
    model.fit(previsores,valor_real,epochs=100,batch_size=32)

    #Salvar modelo em arquivo
    #model.save('preconsumo.h5')

    base_teste = pd.read_csv(test_path)
    valor_real_teste = base_teste.iloc[:,1:2].values
    base_completa = pd.concat((base['potencia'],base_teste['potencia']),axis=0)
    entradas=base_completa[len(base_completa)-len(base_teste)-90:].values
    #Para prever valores open preciso pegar valores da base antiga 
    entradas = entradas.reshape(-1,1)
    entradas=normalizador.transform(entradas)

    #112 é o valor da variável entradas
    X_test = []
    for i in range(90,rows_test):
        X_test.append(entradas[i-90:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    print("X_text", X_test)
    #Previsão
    previsoes = model.predict(X_test)
    print ("previsoes", previsoes)
    previsoes = normalizador.inverse_transform(previsoes)
    fim=time.time()
    durac=fim-ini
    print("Tempo de Execução", round(durac,3), "segundos")
        
    plt.plot(valor_real_teste, color = 'red', label = 'Real value')
    plt.plot(previsoes, color = 'blue', label = 'Forecasts')
    plt.title('Energy Consumption (w)')
    plt.xlabel('Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.savefig('result.png', format='png')
    plt.show()


