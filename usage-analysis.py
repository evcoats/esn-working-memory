import ESN_WM
import ESN_standard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import tensorflow.keras as keras

tf.get_logger().setLevel('INFO')

batches = 20
batch_size = 70
timesteps = 100
input_dim = 3
wm_dim = 2

def usage_analysis_wm(leaky, sw, iter_num):
    
    x = np.zeros((batches, batch_size, timesteps, input_dim))
    y = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))

    for k in range(batches):
        for i in range(batch_size):
            xnew = np.zeros((timesteps, input_dim))
            ynew = np.ones((timesteps, input_dim+wm_dim))*-0.5

            openbrackets = 0
            for j in range(timesteps):
                randomnum = random.random()
                if (openbrackets == 0):
                    if (randomnum < 0.5):
                        openbrackets = 1
                        xnew[j][0] = 1
                        xnew[j][1] = 0 
                        xnew[j][2] = 0 
                        ynew[j][0] = 0.333
                        ynew[j][1] = 0.333
                        ynew[j][2] = 0.333
                        ynew[j][3] = 0.5
                        continue                        
                    else:
                        xnew[j][0] = 0
                        xnew[j][1] = 0
                        xnew[j][2] = 1 
                        ynew[j][0] = 0.5
                        ynew[j][1] = 0
                        ynew[j][2] = 0.5
                        continue                    
                elif (openbrackets == 1):
                    if (randomnum < 0.33):
                        openbrackets = 2
                        xnew[j][0] = 1
                        xnew[j][1] = 0 
                        xnew[j][2] = 0 
                        ynew[j][0] = 0
                        ynew[j][1] = 0.5
                        ynew[j][2] = 0.5
                        ynew[j][3] = 0.5
                        ynew[j][4] = 0.5
                        continue
                    elif (randomnum < 0.66):
                        openbrackets = 0
                        xnew[j][0] = 0
                        xnew[j][1] = 1
                        xnew[j][2] = 0 
                        ynew[j][0] = 0.5
                        ynew[j][1] = 0
                        ynew[j][1] = 0.5
                        continue
                    else:
                        openbrackets = 1
                        xnew[j][0] = 0
                        xnew[j][1] = 0
                        xnew[j][2] = 1
                        ynew[j][0] = 0.333
                        ynew[j][1] = 0.333
                        ynew[j][2] = 0.333
                        ynew[j][3] = 0.5
                        continue
                elif (openbrackets == 2):
                    if (randomnum < 0.5):
                        openbrackets = 1
                        xnew[j][0] = 0
                        xnew[j][1] = 1
                        xnew[j][2] = 0
                        ynew[j][0] = 0.333
                        ynew[j][1] = 0.333
                        ynew[j][2] = 0.333
                        ynew[j][3] = 0.5
                        continue
                    else:
                        openbrackets = 2
                        xnew[j][0] = 0
                        xnew[j][1] = 0 
                        xnew[j][2] = 1
                        ynew[j][0] = 0
                        ynew[j][1] = 0.5
                        ynew[j][2] = 0.5
                        ynew[j][3] = 0.5
                        ynew[j][4] = 0.5
                        continue

            x[k][i] = xnew
            y[k][i] = ynew

    X_train = x[:batches-2]
    X_test = x[batches-2:]
    Y_train = y[:batches-2]
    Y_test = y[batches-2:]

    ESN_WM1, loss = ESN_WM.train_ESN_WM(X_train=X_train, Y_train=Y_train, output_layer_size=3, epochs = 1, wm_size = wm_dim, units = 300, connectivity = 0.1, leaky = leaky, sw=sw,spectral_radius = 0.5, experiment_name="experiment2")

    ESN_WM1.load_weights(".\models\experiment20")



    weights = ESN_WM1.layers[1].get_weights()
    
    weights[2] = np.array([0.,0.])
    
    ESN_WM1.layers[1].set_weights(weights)

    # print(weights)

    loss_fn = keras.losses.MeanSquaredError(reduction='sum_over_batch_size')

    loss_value_standard = None

    loss_value_wm = None

    for x_batch_train, y_batch_train in zip(X_train, Y_train):
        # Open a GradientTape.
        predictions = ESN_WM1(x_batch_train)

        loss_value_standard = loss_fn(y_true = y_batch_train[:,:,0], y_pred = predictions[0][:,:,0])

        loss_value_wm = loss_fn(y_true = y_batch_train[:,:,X_train.shape[-1]:], y_pred = predictions[1][:,:,:])

    print("loss value std: " +str(float(loss_value_standard)))
    print("loss value wm: " +  str(float(loss_value_wm)))

    # Y_pred1 = ESN_WM1(X_test[0])

    # plt.plot(Y_test[0][0,:,2], color="purple",linewidth=5)
    # plt.plot(Y_test[0][0,:,1], color="orange",linewidth=7)
    # plt.plot(Y_test[0][0,:,3], color="black",linewidth=3)
    # plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
    # plt.plot(Y_test[0][0,:,4], color="black",linewidth=5)
    # plt.plot(Y_pred1[0][0,:,:], color="red",linewidth=1)
    # plt.plot(Y_pred1[1][0,:,:], color="green")

    # plt.savefig('models\\figures\\experiment2\\ESNWM' + str(iter_num)+ '.png')
    # ESN_WM1.save_weights('models\\experiment2'+str(iter_num))
    # plt.clf()

    return loss
    

usage_analysis_wm(0.6,1.1,100)