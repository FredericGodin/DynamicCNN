__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'



import theano.tensor as T
from lasagne import init
from lasagne.layers import EmbeddingLayer

class SentenceEmbeddingLayer(EmbeddingLayer):

    def __init__(self, incoming, vocab_size, embedding_size,
                 W=init.Normal(), padding='no', **kwargs):
        super(SentenceEmbeddingLayer, self).__init__(incoming, input_size=vocab_size, output_size=embedding_size,
                 W=W, **kwargs)

        if padding=='first':
            self.sentence_W=T.concatenate([T.zeros((1,embedding_size)),self.W])
        elif padding=='last':
            self.sentence_W=T.concatenate([self.W,T.zeros((1,embedding_size))])
        else:
            self.sentence_W=self.W

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-1] + (self.output_size, ) + (input_shape[-1],)

    def get_output_for(self, input, **kwargs):
        t_size = len(self.input_shape)+1
        t_shape = tuple(range(0,t_size-2))+(t_size-1,t_size-2)
        return T.transpose(self.sentence_W[input],t_shape)

