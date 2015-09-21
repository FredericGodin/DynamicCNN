__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

from lasagne.layers.base import Layer
import lasagne

class NonLinearLayer(Layer):

    def __init__(self,incoming,activation=lasagne.nonlinearities.linear,**kwargs):
        super(NonLinearLayer, self).__init__(incoming, **kwargs)
        self.activation = activation

    def get_output_shape_for(self, input_shape):

        return input_shape

    def get_output_for(self, input, input_shape=None, **kwargs):

        return self.activation(input)