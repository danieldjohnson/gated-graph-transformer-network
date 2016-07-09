import theano
import theano.tensor as T
import numpy as np
from util import *

class Layer(object):

    def __init__(self, input_size, output_size, bias_shift=0.0, name='layer', activation=lambda x:x):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.name = name if name is not None else get_unique_name(type(self))
        self._W = theano.shared(init_params([input_size, output_size]), self.name+"_W")
        self._b = theano.shared(init_params([output_size], shift=bias_shift), self.name+"_W")

    @property
    def params(self):
        return [self._W, self._b]

    def process(self, ipt):
        xW = T.dot(ipt, self._W)
        b = T.shape_padleft(self._b)
        return self.activation( xW + b )

class LayerStack(object):
    def __init__(self, input_size, output_size, hidden_sizes=[], bias_shift=0.0, name=None, hidden_activation=T.tanh, activation=lambda x:x):
        self.input_size = input_size
        self.output_size =output_size
        self.name = name if name is not None else get_unique_name(type(self))

        self.layers = []
        for i, isize, osize in zip(itertools.count(),
                                    [input_size]+hidden_sizes,
                                    hidden_sizes+[output_size]):
            if i == len(hidden_sizes):
                # Last layer
                self.layers.append(Layer(isize, osize, bias_shift=bias_shift, name="{}[output]".format(self.name), activation=activation))
            else:
                self.layers.append(Layer(isize, osize, name="{}[hidden{}]".format(self.name,i), activation=hidden_activation))

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def process(self, ipt):
        val = ipt
        for layer in self.layers:
            val = layer.process(val)
        return val



