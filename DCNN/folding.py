__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'


import theano.tensor as T

from lasagne.layers.base import Layer

class FoldingLayer(Layer):

    def __init__(self,incoming,**kwargs):
        super(FoldingLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[2]/2, input_shape[3])

    def get_output_for(self, input, **kwargs):
        # The paper defines that every consecutive 2 rows are merged into 1 row.
        # For efficiency reasons, we use a reshape function which means that we merge every x and x + n/2 row.
        # For a NN, this is the same implementation

        # make 2 long rows
        long_shape = (self.input_shape[0],self.input_shape[1],2,-1)
        long_rows = T.reshape(input,long_shape)
        # sum the two rows
        summed = T.sum(long_rows,axis=2,keepdims=True)
        # reshape them back
        folded_output = T.reshape(summed,(self.input_shape[0], self.input_shape[1],self.input_shape[2]/2, -1))

        return folded_output