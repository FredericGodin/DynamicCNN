__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

import theano.tensor as T
from lasagne.layers.base import Layer


class KMaxPoolLayer(Layer):

    def __init__(self,incoming,k,**kwargs):
        super(KMaxPoolLayer, self).__init__(incoming, **kwargs)
        self.k = k

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.k)

    def get_output_for(self, input, **kwargs):
        return self.kmaxpooling(input,self.k)


    def kmaxpooling(self,input,k):

        sorted_values = T.argsort(input,axis=3)
        topmax_indexes = sorted_values[:,:,:,-k:]
        # sort indexes so that we keep the correct order within the sentence
        topmax_indexes_sorted = T.sort(topmax_indexes)

        #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
        dim0 = T.arange(0,self.input_shape[0]).repeat(self.input_shape[1]*self.input_shape[2]*k)
        dim1 = T.arange(0,self.input_shape[1]).repeat(k*self.input_shape[2]).reshape((1,-1)).repeat(self.input_shape[0],axis=0).flatten()
        dim2 = T.arange(0,self.input_shape[2]).repeat(k).reshape((1,-1)).repeat(self.input_shape[0]*self.input_shape[1],axis=0).flatten()
        dim3 = topmax_indexes_sorted.flatten()
        return input[dim0,dim1,dim2,dim3].reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2], k))



class DynamicKMaxPoolLayer(KMaxPoolLayer):

    def __init__(self,incoming,ktop,nroflayers,layernr,**kwargs):
        super(DynamicKMaxPoolLayer, self).__init__(incoming,ktop, **kwargs)
        self.ktop = ktop
        self.layernr = layernr
        self.nroflayers = nroflayers

    def get_k(self,input_shape):
        return T.cast(T.max([self.ktop,T.ceil((self.nroflayers-self.layernr)/float(self.nroflayers)*input_shape[3])]),'int32')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], None)

    def get_output_for(self, input, **kwargs):

        k = self.get_k(input.shape)

        return self.kmaxpooling(input,k)


