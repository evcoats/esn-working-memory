import ESN_WM
import ESN_standard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random



tf.get_logger().setLevel('INFO')

batches = 20
batch_size = 70
timesteps = 100
input_dim = 3
wm_dim = 2

def bracket_experiment_wm(leaky, sw, iter_num):
    
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

    ESN_WM1, loss = ESN_WM.train_ESN_WM(X_train=X_train, Y_train=Y_train, output_layer_size=3, epochs = 10, wm_size = wm_dim, units = 300, connectivity = 0.1, leaky = leaky, sw=sw,spectral_radius = 0.5, experiment_name="experiment2")

    Y_pred1 = ESN_WM1(X_test[0])

    plt.plot(Y_test[0][0,:,2], color="purple",linewidth=5)
    plt.plot(Y_test[0][0,:,1], color="orange",linewidth=7)
    plt.plot(Y_test[0][0,:,3], color="black",linewidth=3)
    plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
    plt.plot(Y_test[0][0,:,4], color="black",linewidth=5)
    plt.plot(Y_pred1[0][0,:,:], color="red",linewidth=1)
    plt.plot(Y_pred1[1][0,:,:], color="green")

    plt.savefig('models\\figures\\experiment2\\ESNWM' + str(iter_num)+ '.png')
    ESN_WM1.save_weights('models\\experiment2'+str(iter_num))
    plt.clf()

    return loss
    
def bracket_experiment_std(leaky, sw, iter_num):
    
    x = np.zeros((batches, batch_size, timesteps, input_dim))
    y = np.zeros((batches, batch_size, timesteps, input_dim))

    for k in range(batches):
        for i in range(batch_size):
            xnew = np.zeros((timesteps, input_dim))
            ynew = np.ones((timesteps, input_dim))*-0.5

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
                        continue

                    else:
                        openbrackets = 0
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
                        continue

                    elif (randomnum < 0.66):
                        openbrackets = 0
                        xnew[j][0] = 0
                        xnew[j][1] = 1
                        xnew[j][2] = 0 
                        ynew[j][0] = 0.5
                        continue

                    else:
                        openbrackets = 1
                        xnew[j][0] = 0
                        xnew[j][1] = 0
                        xnew[j][2] = 1
                        ynew[j][0] = 0.333
                        ynew[j][1] = 0.333
                        ynew[j][2] = 0.333
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
                        continue

                    else:
                        openbrackets = 2
                        xnew[j][0] = 0
                        xnew[j][1] = 0 
                        xnew[j][2] = 1
                        ynew[j][0] = 0
                        ynew[j][1] = 0.5
                        ynew[j][2] = 0.5
                        continue

            x[k][i] = xnew
            y[k][i] = ynew


    X_train = x[:batches-2]
    X_test = x[batches-2:]
    Y_train = y[:batches-2]
    Y_test = y[batches-2:]

    ESN_standard1, loss = ESN_standard.train_ESN_standard(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 10, units = 300, connectivity = 0.1, leaky = leaky, spectral_radius = 0.5, sw = sw, experiment_name = "experiment2")

    Y_pred1 = ESN_standard1(X_test[0])

    plt.plot(Y_test[0][0,:,2], color="purple",linewidth=5)
    plt.plot(Y_test[0][0,:,1], color="orange",linewidth=7)
    plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
    plt.plot(Y_pred1[0,:,:], color="red",linewidth=1)

    plt.savefig("models\\figures\\experiment2\\ESNSTD"+str(iter_num) + '.png')
    ESN_standard1.save_weights('models\\experiment2\\ESNSTD'+str(iter_num))
    plt.clf()

    return loss

losses = []
idx = 0 
for leaky_iter in range(2):
   for sw_iter in range(3):
      losses.append(bracket_experiment_wm(0.6+(leaky_iter)*0.2,1.1+(0.3*sw_iter),idx))

      print("Loss for %s with leaky %s and sw %s: %s" % ("ESN_WM", str(0.6+(leaky_iter)*0.2), str(1.1+(0.3*sw_iter)), losses[idx]))
      idx += 1

      losses.append(bracket_experiment_std(0.6+(leaky_iter)*0.2,1.1+(0.3*sw_iter),idx))

      print("Loss for %s with leaky %s and sw %s: %s" % ("ESN_STANDARD", str(0.6+(leaky_iter)*0.2), str(1.1+(0.3*sw_iter)), losses[idx]))
      idx += 1




