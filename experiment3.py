import ESN_WM
import ESN_standard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd

# URL to the CSV file


def EEGdata(readurl):

    df = pd.read_csv(readurl)

    arr = df.to_numpy()

    print(arr.shape)

    arr = arr[:,1:]
    
    appendedArrX = np.ones((1, 500, 4))
    
    appendedArrY = np.ones((1, 500, 5))


    for j in range(48):

        newarr = np.array([arr[j*500:(j+1)*500]])

        maxs = np.ones(arr.shape[1])

        for i in range (arr.shape[1]):
            maxs[i] = np.max(newarr[0,:,i])
            newarr[0,:,i] = newarr[0,:,i]/maxs[i]


        threshold = np.ones(arr.shape[1])*0.5

        newY = np.zeros((1,500,arr.shape[1]+1))
        counter = 0
        numcounted = 0


        for i in range(500):
            if (0 <= counter <= 3):
                if (newarr[0,i,0] > 0.2):
                    print("here")
                    newY[0][i][0] = 1 
                    counter+=1
                    numcounted += 1
                newY[0][i][1] = (1+counter%4)*0.25

            elif (4 <= counter <= 7):
                if (newarr[0,i,1] > 0.2):
                    newY[0][i][0] = 1 
                    counter+=1
                    numcounted += 1
                newY[0][i][2] =  (1+counter%4)*0.25
            elif (8 <= counter <= 11):
                if (newarr[0,i,2] > 0.2):
                    newY[0][i][0] = 1 
                    counter+=1
                    numcounted += 1
                newY[0][i][3] =  (1+counter%4)*0.25
            elif (12 <= counter <= 15):
                if (newarr[0,i,3] > 0.2):
                    newY[0][i][0] = 1 
                    counter+=1
                    counter = counter%16
                    numcounted += 1
                newY[0][i][4] =  (1+counter%4)*0.25

        appendedArrX = np.append(appendedArrX, newarr,axis=0)
        appendedArrY = np.append(appendedArrY, newY,axis=0)

        # print(appendedArrY)
        # print(appendedArrX)
        # print(numcounted)


    print(appendedArrY)

    return (appendedArrX[1:], appendedArrY[1:])


url = ".\datasets\s01_ex01_s01.csv"

df11X, df11Y = EEGdata(url)

url = ".\datasets\s01_ex09.csv"

df9X, df9Y = EEGdata(url)

url = ".\datasets\s01_ex05.csv"

df5X, df5Y = EEGdata(url)

url = ".\datasets\s01_ex01_s02.csv"

df12X, df12Y = EEGdata(url)

url = ".\datasets\s01_ex01_s03.csv"

df13X, df13Y = EEGdata(url)

X = np.array([df11X])

Y = np.array([df11Y])

testX = df13X
testY = df13Y

wm_dim = 4
leaky = 0.8
sw = 1.2

Ystandard = Y[:,:,:,0]

print(Ystandard.shape)

Ystandard.reshape((1,48,500,1))

# ESN_WM1, lossWM = ESN_WM.train_ESN_WM(X_train=X, Y_train=Y, output_layer_size=1, epochs = 50, wm_size = wm_dim, units = 300, connectivity = 0.1, leaky = leaky, sw=sw,spectral_radius = 0.5, experiment_name="experiment3")

ESN_std1, lossSTD = ESN_standard.train_ESN_standard(X_train=X, Y_train=Ystandard, output_layer_size=1, epochs = 50, units = 300, connectivity = 0.1, leaky = leaky, sw=sw, spectral_radius = 0.5, experiment_name="experiment3")

print(lossWM)
print(lossSTD)














