import ESN_WM
import ESN_standard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# comparison of ESN w/ WM to standard ESN on task based flipping a sin function upon remembering certain combinations of inputs
# task1 has a two dimensional input: the first dimension is a sin function, while the second dimension includes mostly zeros but randomly has runs of ones. The output should be the flipped sin function if the second dimension has had two or three runs of ones, and should be flipped again upon the forth. Then it will remain flipped until the last timestep.
# on task1, the ESN with 5 WM units trained corresponding memory of the runs of ones on the first four, and the sign to flip to on the fifth, outperforms a standard ESN on the task while it is flipped. 
# this is a step towards demonstrating the practical memory limit of a standard ESN 

def esn_wm_task1(leaky, sw):
  batches = 20
  batch_size = 500
  timesteps = 200
  input_dim = 1
  wm_dim = 5

  x = np.zeros((batches, batch_size, timesteps, input_dim+1))
  y = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))


  for k in range(batches):
      for i in range(batch_size):

         t = np.linspace(0, 2, timesteps)
         t = t.reshape(timesteps,1)
         xnew = np.sin(t*6*np.pi)
         xnew = np.concatenate((xnew,np.zeros((timesteps,1))), axis=1)
         ynew = t*6*np.pi
         sin = np.sin(ynew)*0.5 # generate the sinusoid

         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))*(-0.5)), axis=1)
         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))*(-0.5)), axis=1)
         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))*(-0.5)), axis=1)
         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))*(-0.5)), axis=1)
         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))), axis=1)

         sign = 0 
         num1s = 0
         numImpulse = 0
        
         for j in range(sin.shape[0]):
      
            if  -0.05 < sin[j][0] < 0.05:  # check if the sinusoid crosses zero
               sign = np.random.choice([-1, 1])  # generate ones
               if sign == 1 and numImpulse == 0 and j > 3:
                  numImpulse = 5
                  num1s+=1
                  if (num1s > 4):
                     num1s = 4
                  sign = 0
            
            if (numImpulse > 0):
               numImpulse -= 1
               xnew[j][1] = 1
                  
            if (num1s == 1):
               sin[j][1] = 0.5
               sin[j][2] = -0.5
               sin[j][3] = -0.5
               sin[j][4] = -0.5

            if (num1s == 2):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][3] = -0.5
               sin[j][4] = -0.5
               sin[j][5] = -1
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones seen

            if (num1s == 3):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][3] = 0.5
               sin[j][4] = -0.5
               sin[j][5] = -1
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones seen

            if (num1s == 4):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][3] = 0.5
               sin[j][4] = 0.5

         x[k][i] = xnew

         y[k][i] = sin



  X_train = x[:batches-2]
  X_test = x[batches-2:]
  Y_train = y[:batches-2]
  Y_test = y[batches-2:]

  # print(Y_test.shape)


  ESN_WM1, loss = ESN_WM.train_ESN_WM(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 100, wm_size = wm_dim, units = 500, connectivity = 0.1, leaky = leaky, sw=sw,spectral_radius = 1.0)

  Y_pred1 = ESN_WM1(X_test[0])

  # print(Y_pred.shape)

  # plt.plot(Y_test[0,0,:], color="blue",linewidth=5)
  plt.plot(Y_test[0][0,:,2], color="purple",linewidth=5)
  plt.plot(Y_test[0][0,:,1], color="orange",linewidth=7)
  plt.plot(Y_test[0][0,:,3], color="black",linewidth=8)
  plt.plot(Y_test[0][0,:,4], color="red",linewidth=9)
  plt.plot(Y_test[0][0,:,5], color="red",linewidth=9)
  plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
  plt.plot(Y_pred1[0][0,:,:], color="red",linewidth=1)
  plt.plot(Y_pred1[1][0,:,:4], color="green")
  plt.plot(Y_pred1[1][0,:,5], color="gray")


  plt.show()
  
  
  return loss

def esn_standard_task1(leaky, sw):
  batches = 20
  batch_size = 500
  timesteps = 200
  input_dim = 1

  x = np.zeros((batches, batch_size, timesteps, input_dim+1))
  y = np.zeros((batches, batch_size, timesteps, input_dim))


  for k in range(batches):
      for i in range(batch_size):

         t = np.linspace(0, 2, timesteps)
         t = t.reshape(timesteps,1)
         xnew = np.sin(t*6*np.pi)
         xnew = np.concatenate((xnew,np.zeros((timesteps,1))), axis=1)
         ynew = t*6*np.pi
         sin = np.sin(ynew)*0.5 # generate the sinusoid


         sign = 0
         num1s = 0
         numImpulse = 0
      
         for j in range(sin.shape[0]):
            if  -0.05 < sin[j][0] < 0.05: # check if the sinusoid crosses zero
               sign = np.random.choice([-1, 1])  # generate ones
               if sign == 1 and numImpulse == 0 and j > 3:
                  numImpulse = 5
                  num1s+=1
                  if (num1s > 4):
                     num1s = 4
               sign = 0
            
            if (numImpulse > 0):
               numImpulse -= 1
               xnew[j][1] = 1
                  
            if (num1s == 2 or num1s == 3):
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones

            
            

         x[k][i] = xnew

         y[k][i] = sin


  X_train = x[:batches-2]
  X_test = x[batches-2:]
  Y_train = y[:batches-2]
  Y_test = y[batches-2:]


  ESN_standard1, loss = ESN_standard.train_ESN_standard(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 100, units = 500, connectivity = 0.1, leaky = leaky, spectral_radius = 1.0, sw = sw)
  
  Y_pred = ESN_standard1(X_test[0])


  plt.plot(Y_test[0,0,:], color="blue",linewidth=5)
  plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
  plt.plot(Y_pred[0,:,:], color="red",linewidth=1)

  plt.show()


  return loss


losses = []

for leaky_iter in range(1):
   for sw_iter in range(2):
      losses.append(esn_wm_task1(0.6+(leaky_iter)*0.2,1.2+(0.3*sw_iter)))
      losses.append(esn_standard_task1(0.6+(leaky_iter)*0.2,1.2+(0.3*sw_iter)))

print(losses)
idx = 0 
for leaky_iter in range(1):
   for sw_iter in range(2):
      for comp_iter in range(2):
         type = "wm_esn"
         if (comp_iter == 1):
            type = "standard_esn"
         print("Loss for %s with leaky %s and sw %s: %s" % (type, str(0.6+(leaky_iter)*0.2), str(1.2+(0.3*sw_iter)), losses[idx]))
         idx+=1

# print(losses)

