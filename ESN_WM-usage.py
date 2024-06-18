import ESN_WM
import ESN_standard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def initial_task():
  batches = 20
  batch_size = 1000
  timesteps = 100
  input_dim = 1
  wm_dim = 1

  physical_devices = tf.config.list_physical_devices('GPU')
  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



  x = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))
  y = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))


  for k in range(batches):
      for i in range(batch_size):

         t = np.linspace(0, 1, timesteps)

         t = t.reshape(timesteps,1)

         xnew = np.exp(-200*t)

         xnew = np.concatenate((xnew,np.zeros((timesteps,1))), axis=1)

         ynew = t*6*np.pi

         
         # print(xnew.shape)

         sin = np.sin(ynew) # generate the sinusoid

         # xnew = np.sin(xnew) # generate the sinusoid

         
         #fill newly created dimension with zeros
         # print(sin.shape)
         # print(sin.shape[0])
         sin = np.concatenate((sin, np.ones((sin.shape[0], 1))), axis=1)

         # print(sin)
         # print(sin)

         sign = 1 # initialize the sign as positive

         for j in range(sin.shape[0]):
            if  -0.05 < sin[j][0] < 0.05:  # check if the sinusoid crosses zero
               sign = np.random.choice([-1, 1])  # randomly switch between negative and positive
               xnew[j][1] = sign

            sin[j][1] = sign*0.5
            sin[j][0] *= sign  # apply the sign to the sinusoid value

         x[k][i] = xnew


         # sin[:,1] *= 0.5

         y[k][i] = sin

         # print(sin)
         # print(xnew)



         # xplot, = plt.plot(x[k,i,:], color="blue")
         # tplot, = plt.plot(y[k,i,:,0], color="black")
         

         # # plt.xlabel("Timesteps")
         # plt.show()



  X_train = x[:batches-2]
  X_test = x[batches-2:]
  Y_train = y[:batches-2]
  Y_test = y[batches-2:]

  print(Y_test.shape)


  ESN_WM1 = ESN_WM.train_ESN_WM(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 2, wm_size = 1, units = 300, connectivity = 0.05, leaky = 0.8, spectral_radius = 0.8)

  Y_pred = ESN_WM1(X_test[0])

  # print(Y_pred.shape)

  # plt.plot(Y_test[0,0,:], color="blue",linewidth=5)
  plt.plot(Y_test[0][0,:,1], color="blue",linewidth=10)
  plt.plot(Y_test[0][0,:,1], color="orange",linewidth=5)
  plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
  plt.plot(Y_pred[0][0,:,:], color="red",linewidth=1)
  plt.plot(Y_pred[1][0,:,0], color="green")
  plt.plot(Y_pred[1][0,:,1], color="purple")


  plt.show()

  print(Y_train.shape)


def comparison_task1(leaky, sw):
  batches = 20
  batch_size = 100
  timesteps = 500
  input_dim = 1
  wm_dim = 4

  x = np.zeros((batches, batch_size, timesteps, input_dim+2))
  y = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))


  for k in range(batches):
      for i in range(batch_size):

         t = np.linspace(0, 1, timesteps)

         t = t.reshape(timesteps,1)

         xnew = np.sin(t*6*np.pi)

         xnew = np.concatenate((xnew,np.zeros((timesteps,1))), axis=1)
         
         xnew = np.concatenate((xnew,np.exp(-200*t)), axis=1)

         ynew = t*6*np.pi

         
         # print(xnew.shape)

         sin = np.sin(ynew) # generate the sinusoid

          # xnew = np.sin(xnew) # generate the sinusoid

          
          #fill newly created dimension with zeros
          # print(sin.shape)
          # print(sin.shape[0])
         sin = np.concatenate((sin, np.zeros((sin.shape[0], 1))), axis=1)
         sin = np.concatenate((sin, np.zeros((sin.shape[0], 1))), axis=1)
         sin = np.concatenate((sin, np.zeros((sin.shape[0], 1))), axis=1)
         sin = np.concatenate((sin, np.zeros((sin.shape[0], 1))), axis=1)

         # print(sin)
         # print(sin)

         sign = 1 # initialize the sign as positive
         num1s = 0
         numImpulse = 0
        
         for j in range(sin.shape[0]):
      
            if  -0.05 < sin[j][0] < 0.05:  # check if the sinusoid crosses zero
               sign = np.random.choice([-1, 1])  # generate ones
               if sign == 1 and numImpulse == 0:
                  numImpulse = 5
                  num1s+=1
                  if (num1s > 2):
                     num1s = 2
            
            if (numImpulse > 0):
               numImpulse -= 1
               xnew[j][1] = 10
                  
            if (num1s == 1):
               sin[j][1] = 0.5

            if (num1s == 2):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones seen

            if (num1s == 3):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][3] = 0.5
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones seen

            if (num1s == 3):
               sin[j][1] = 0.5
               sin[j][2] = 0.5
               sin[j][3] = 0.5
               sin[j][4] = 0.5
              

         x[k][i] = xnew


         # sin[:,1] *= 0.5

         y[k][i] = sin

    
          # print(sin)
          # print(xnew)



          # xplot, = plt.plot(x[k,i,:], color="blue")
          # tplot, = plt.plot(y[k,i,:,0], color="black")
          

          # # plt.xlabel("Timesteps")
          # plt.show()



  X_train = x[:batches-2]
  X_test = x[batches-2:]
  Y_train = y[:batches-2]
  Y_test = y[batches-2:]

  # print(Y_test.shape)


  ESN_WM1, loss = ESN_WM.train_ESN_WM(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 3, wm_size = wm_dim, units = 300, connectivity = 0.1, leaky = leaky, sw=sw,spectral_radius = 0.8)

  Y_pred = ESN_WM1(X_test[0])

  # # print(Y_pred.shape)

  # plt.plot(Y_test[0,0,:], color="blue",linewidth=5)
  plt.plot(Y_test[0][0,:,2], color="purple",linewidth=5)
  plt.plot(Y_test[0][0,:,1], color="orange",linewidth=5)
  plt.plot(Y_test[0][0,:,3], color="purple",linewidth=5)
  plt.plot(Y_test[0][0,:,4], color="orange",linewidth=5)
  plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
  plt.plot(Y_pred[0][0,:,:], color="red",linewidth=1)
  plt.plot(Y_pred[1][0,:,:], color="green")

  plt.show()

  # print(Y_train.shape)
  return loss

def standard_task1(leaky, sw):
  batches = 20
  batch_size = 100
  timesteps = 500
  input_dim = 1
  wm_dim = 2

  x = np.zeros((batches, batch_size, timesteps, input_dim+2))
  y = np.zeros((batches, batch_size, timesteps, input_dim+wm_dim))


  for k in range(batches):
      for i in range(batch_size):

         t = np.linspace(0, 1, timesteps)

         t = t.reshape(timesteps,1)

         xnew = np.sin(t*6*np.pi)

         xnew = np.concatenate((xnew,np.zeros((timesteps,1))), axis=1)
         
         xnew = np.concatenate((xnew,np.exp(-200*t)), axis=1)


         ynew = t*6*np.pi

         
         # print(xnew.shape)

         sin = np.sin(ynew) # generate the sinusoid

         # xnew = np.sin(xnew) # generate the sinusoid

         
         #fill newly created dimension with zeros
         # print(sin.shape)
         # print(sin.shape[0])

         # print(sin)
         # print(sin)

         sign = 1 # initialize the sign as positive
         num1s = 0
         numImpulse = 0
      
         for j in range(sin.shape[0]):
            if  -0.05 < sin[j][0] < 0.05:  # check if the sinusoid crosses zero
               sign = np.random.choice([-1, 1])  # generate ones
               if sign == 1 and numImpulse == 0:
                  numImpulse = 5
                  num1s+=1
                  if (num1s > 4):
                     num1s = 4
            
            if (numImpulse > 0):
               numImpulse -= 1
               xnew[j][1] = 10
                  
            if (num1s == 2 or num1s == 3):
               sin[j][0] *= -1  # if two ones are seen, switch to negative until 4 ones

            
            

         x[k][i] = xnew
         # sin[:,1] *= 0.5

         y[k][i] = sin


  X_train = x[:batches-2]
  X_test = x[batches-2:]
  Y_train = y[:batches-2]
  Y_test = y[batches-2:]

  # print(Y_test.shape)


  ESN_standard1, loss = ESN_standard.train_ESN_standard(X_train=X_train, Y_train=Y_train, output_layer_size=1, epochs = 200, units = 300, connectivity = 0.1, leaky = leaky, spectral_radius = 0.8, sw = sw)
  
  Y_pred = ESN_standard1(X_test[0])

  # print(Y_pred.shape)

  plt.plot(Y_test[0,0,:], color="blue",linewidth=5)
  # plt.plot(X_test[0][0,:,2], color="purple",linewidth=5)
#   plt.plot(X_test[0][0,:,1], color="orange",linewidth=5)
  plt.plot(Y_test[0][0,:,0], color="black",linewidth=5)
  plt.plot(Y_pred[0,:,:], color="red",linewidth=1)

  plt.show()

#   print(Y_train.shape)

  return loss

# standard_task1()
# comparison_task1()

losses = []

for leaky_iter in range(1):
   for sw_iter in range(1):
      losses.append(comparison_task1(0.6+(leaky_iter)*0.2,1.2+(0.3*sw_iter)))
      losses.append(standard_task1(0.6+(leaky_iter)*0.2,1.2+(0.3*sw_iter)))

print(losses)
idx = 0 
for leaky_iter in range(1):
   for sw_iter in range(1):
      for comp_iter in range(2):
         type = "wm_esn"
         if (comp_iter == 1):
            type = "standard_esn"
         print("Loss for %s with leaky %s and sw %s: %s" % (type, str(0.6+(leaky_iter)*0.2), str(1.2+(0.3*sw_iter)), losses[idx]))
         idx+=1

# print(losses)

