
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sys, os

from tensorflow.python.ops import math_ops

import tensorflow.keras as keras
from typing import Union, Callable, List


Initializer = Union[None, dict, str, Callable]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]



# @keras.saving.register_keras_serializable()
class ESN_WM_Cell(keras.layers.AbstractRNNCell):
    """Echo State recurrent Network (ESN) cell with Working Memory Units.

    With support for trainable working memory units, as outlined in the paper
        R. Pascanu, H. Jaeger
        "A neurodynamical model for working memory" 
        https://doi.org/10.1016/j.neunet.2010.10.003.


    This implements the recurrent cell from the paper:
        H. Jaeger
        "A neurodynamical model for working memory".
        GMD Report148, German National Research Center for Information Technology, 2001.
        https://www.researchgate.net/publication/215385037


    Arguments:
        wm_size: Positive integer, dimensionality of the working memory unit.
        units: Positive integer, dimensionality in the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky
            integration.
            Default: 1.
        spectral_radius: Float between 0 and 1.
            Desired spectral radius of recurrent weight matrix.
            Default: 0.9.
        use_norm2: Boolean, whether to use the p-norm function (with p=2) as an upper
            bound of the spectral radius so that the echo state property is satisfied.
            It  avoids to compute the eigenvalues which has an exponential complexity.
            Default: False.
        use_bias: Boolean, whether the layer uses a bias vector.
            Default: True.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
    Call arguments:
        inputs: A 2D tensor (batch x num_units).
        states: List of state tensors corresponding to the previous timestep.
    """

    def __init__(
        self,
        wm_size: int,
        units: int,
        connectivity: float = 0.1,
        leaky: float = 1,
        sw: float = 1,
        spectral_radius: float = 0.5,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation: Activation = "tanh",
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "glorot_uniform",
        bias_initializer: Initializer = "zeros",
        wm_back_connectivity: float = 0.1,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.wm_size = wm_size
        self.units = units
        self.connectivity = connectivity
        self.leaky = leaky
        self.sw = sw
        self.spectral_radius = spectral_radius
        self.use_norm2 = use_norm2
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.wm_back_connectivity = wm_back_connectivity
        self._state_size = (units, wm_size)
        self._output_size = units


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size


    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        def _esn_recurrent_initializer(shape, dtype, partition_info=None):
            recurrent_weights = tf.keras.initializers.get(self.recurrent_initializer)(
                shape, dtype
            )

            connectivity_mask = tf.cast(
                tf.math.less_equal(tf.random.uniform(shape), self.connectivity,), dtype
            )
            recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

            # Satisfy the necessary condition for the echo state property `max(eig(W)) < 1`
            if self.use_norm2:
                # This condition is approximated scaling the norm 2 of the reservoir matrix
                # which is an upper bound of the spectral radius.
                recurrent_norm2 = tf.math.sqrt(
                    tf.math.reduce_sum(tf.math.square(recurrent_weights))
                )
                is_norm2_0 = tf.cast(tf.math.equal(recurrent_norm2, 0), dtype)
                scaling_factor = self.spectral_radius / (
                    recurrent_norm2 + 1 * is_norm2_0
                )
            else:
                abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
                scaling_factor = tf.math.divide_no_nan(
                    self.spectral_radius, tf.reduce_max(abs_eig_values)
                )

            recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)

            return recurrent_weights

        def _wm_back_initializer(shape, dtype, partition_info=None):
            kernel_weights = tf.keras.initializers.get(self.kernel_initializer)(
                shape, dtype
            )
            connectivity_mask = tf.cast(
                tf.math.less_equal(tf.random.uniform(shape), self.wm_back_connectivity), dtype
            )

            kernel_weights = tf.math.multiply(kernel_weights, connectivity_mask)

            return kernel_weights
        
        
        self.sw = tf.Variable(self.sw, name="sw",
                        dtype=tf.float32,
                        trainable=False)
        
        
        self.leaky = tf.Variable(self.leaky, name="leaky",
                dtype=tf.float32,
                trainable=False)

        self.wmb = tf.Variable(1, name="wmb",
                dtype=tf.float32,
                trainable=False)

        self.wmscale = tf.Variable(1, name="wmscale",
                dtype=tf.float32,
                trainable=False)
        
        self.wmleaky = tf.Variable(1, name="wmleaky",
                dtype=tf.float32,
                trainable=True)

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=_esn_recurrent_initializer,
            trainable=False,
            dtype=self.dtype,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype,
            )

        self.wm_kernel = self.add_weight(
            name="wm_kernel",
            shape=[self.units + self.wm_size + input_size, self.wm_size],
            trainable=True,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
        )

        self.wm_bias = self.add_weight(
            name="wm_bias",
            shape=[self.wm_size],
            initializer=self.bias_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        self.wm_hidden = self.add_weight(
            name = "wm_hidden",
            shape = [self.wm_size, 50],
            trainable = True,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            dtype = self.dtype
        )


        self.wm_self = self.add_weight(
            name = "wm_self",
            shape = [50, self.wm_size],
            trainable = True,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            dtype = self.dtype

        )

        self.wm_kernel_back = self.add_weight(
            name="wm_kernel_back",
            shape=[self.wm_size, self.units],
            trainable=False,
            initializer=_wm_back_initializer,
            dtype=self.dtype,
        )
            

        self.built = True

    def call(self, inputs, state):

        def half_activation(x):
            return tf.where(x < 0, x*0-0.5, x*0+0.5)


        in_matrix = tf.concat([inputs*self.sw, state[0]], axis=1)
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0)

        output = tf.linalg.matmul(in_matrix, weights_matrix)


        if self.use_bias:
            output = output + self.bias

        output = self.activation(output)

        output = (1 - self.leaky) * state[0] + self.leaky * output

        wm_input = tf.concat([output, self.wmleaky*state[1]], axis=1)

        wm_input = tf.concat([wm_input, inputs], axis=1)

        wm_output = tf.linalg.matmul(self.wmscale*wm_input, self.wm_kernel) + self.wm_bias

        wm_output = tf.linalg.matmul(wm_output, self.wm_hidden)

        wm_output = tf.linalg.matmul(wm_output, self.wm_self)        

        change = tf.linalg.matmul((half_activation(wm_output)), self.wm_kernel_back)

        output = output + (self.wmb*change)
        
        return (output,wm_output), (output,half_activation(wm_output))

    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}


def ESN_WM_Model(input_shape, num_outputs, num_timesteps, wm_size, units, connectivity, leaky, sw, spectral_radius):
    

    inputs = keras.Input((num_timesteps,input_shape))

    rnn = (keras.layers.RNN(
        ESN_WM_Cell(
            wm_size=wm_size,
            units=units,
            connectivity=connectivity,
            leaky=leaky,
            sw = sw,
            spectral_radius=spectral_radius,
        ),
        input_shape=(num_timesteps,input_shape), name = "ESN_WM", return_sequences = True
    ))

    modelOutputs = rnn(inputs)


    outputs = keras.layers.Dense(num_outputs, name="outputs")

    outputsDense = outputs(modelOutputs[0])
    
    model = keras.Model(inputs, (outputsDense,modelOutputs[1]), name="ESN_WM_Model")

    model.summary()


    return model

## Y_train should be of shape (num_samples, num_timesteps, (num_outputs, num_wm_units)) to account for WM units

def train_ESN_WM(X_train, Y_train, output_layer_size, epochs, wm_size, units, connectivity, leaky,sw, spectral_radius, experiment_name):

    loss_fn = keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)
    loss_value_standard = None
    loss_value_wm = None

    model = ESN_WM_Model(input_shape = X_train.shape[-1], num_outputs = output_layer_size, num_timesteps=X_train.shape[2], wm_size=wm_size, units=units, connectivity=connectivity, leaky=leaky, sw=sw, spectral_radius=spectral_radius)
    
    f = open("losses\losses.txt", "a")
    f.write("Experiment: {} \n".format(experiment_name))
    f.close()

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch,))

        step1 = 0
    
        for x_batch_train, y_batch_train in zip(X_train, Y_train):

            # Open a GradientTape.

            with tf.GradientTape(persistent=True) as tape:
                # Forward pass.

                predictions = model(x_batch_train)

                # print(y_batch_train[:,:,1:])

                # print(predictions[1][:,:,:])

                loss_value_standard = loss_fn(y_true = y_batch_train[:,:,0], y_pred = predictions[0][:,:,0])

                loss_value_wm = loss_fn(y_true = y_batch_train[:,:,1:], y_pred = predictions[1][:,:,:])



            # print()
            # print(model.get_layer("ESN_WM").variables)


            model.get_layer("ESN_WM").trainable = False
            model.get_layer("outputs").trainable = True
            
            # 'ESN_WM/esn_wm__cell/wmb:0'


            # Get gradients of loss wrt the *trainable* weights.
            gradients_standard = tape.gradient(loss_value_standard, model.trainable_weights)


            optimizer.apply_gradients(zip(gradients_standard, model.trainable_weights))

            model.get_layer("ESN_WM").trainable = True
            model.get_layer("outputs").trainable = False

            gradients_wm = tape.gradient(loss_value_wm, model.trainable_weights)


            optimizer.apply_gradients(zip(gradients_wm, model.trainable_weights))
            
            step1 += 1

            # if step1 % 5 == 0:
            print(
                "Training loss (for general loss) at step %d: %.4f"
                % (step1, float(loss_value_standard))
            )
            print(
                "Training loss (for wm loss) at step %d: %.4f"
                % (step1, float(loss_value_wm))
            )

    f = open("losses\losses.txt", "a")

    f.write("ESNWM w/ {} {} {} {} {} {} {} {} \n".format(output_layer_size, epochs, wm_size, units, connectivity, leaky,sw, spectral_radius))
    f.write("Standard Loss: {} \n".format(loss_value_standard))
    f.write("WM Loss: {} \n".format(loss_value_wm))
    f.write("\n")

    f.close()

    return (model,float(loss_value_standard))





