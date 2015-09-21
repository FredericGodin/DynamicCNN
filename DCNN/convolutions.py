__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

from lasagne import *
from lasagne.layers import Layer
import lasagne.utils
import theano.tensor as T

# Adapted from Lasagne
class Conv1DLayerSplitted(Layer):

    def __init__(self, incoming, num_filters, filter_size,
                 border_mode="valid",
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Conv1DLayerSplitted, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = lasagne.utils.as_tuple(1, 1)
        self.border_mode = border_mode


        # If it is an image the input shape will be 3
        # If it is a stack of filter ouputs after a previous convolution, the input shape will be 4
        if len(self.input_shape)==3:
            self.num_input_channels = 1
            self.num_of_rows = self.input_shape[1]
        elif len(self.input_shape)==4:
            self.num_input_channels = self.input_shape[1]
            self.num_of_rows = self.input_shape[2]

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            bias_temp_shape = self.get_output_shape_for(self.input_shape)
            biases_shape = (bias_temp_shape[1],bias_temp_shape[2])
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        return (self.num_filters,self.num_input_channels, self.num_of_rows, self.filter_size)

    def get_output_shape_for(self, input_shape):

        output_length = lasagne.layers.conv.conv_output_length(input_shape[-1],
                                           self.filter_size,
                                           self.stride[0],
                                           self.border_mode)

        return (input_shape[0], self.num_filters, self.num_of_rows, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):

        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        # We split the input shape and the filters into seperate rows to be able to execute a row wise 1D convolutions
        # We cannot convolve over the columns
        # However, we do need to convolve over multiple channels=output filters previous layer
        # See paper of Kalchbrenner for more details
        if self.border_mode in ['valid', 'full']:

            if len(self.input_shape)==3:
                input_shape_row= (self.input_shape[0], 1, 1, self.input_shape[2])
                new_input = input.dimshuffle(0,'x', 1, 2)
            elif len(self.input_shape)==4:
                input_shape_row= (self.input_shape[0], self.input_shape[1], 1,  self.input_shape[3])
                new_input = input

            filter_shape_row =(filter_shape[0],filter_shape[1],1,filter_shape[3])
            conveds = []

            #Note that this for loop is only to construct the Theano graph and will never be part of the computation
            for i in range(self.num_of_rows):
                conveds.append(T.nnet.conv.conv2d(new_input[:,:,i,:].dimshuffle(0,1,'x',2),
                               self.W[:,:,i,:].dimshuffle(0,1,'x',2),
                               image_shape=input_shape_row,
                               filter_shape=filter_shape_row,
                               border_mode=self.border_mode,
                               ))

            conved = T.concatenate(conveds,axis=2)



        elif self.border_mode == 'same':
            raise NotImplementedError("Not implemented yet ")
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)


        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x',0,1,'x')


        return self.nonlinearity(activation)

