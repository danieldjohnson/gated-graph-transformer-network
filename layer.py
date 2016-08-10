import theano
import theano.tensor as T
import numpy as np
from util import *

class Layer(object):

    def __init__(self, input_size, output_size, bias_shift=0.0, name='layer', activation=identity, dropout_keep=1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.name = name if name is not None else get_unique_name(type(self))
        self._W = theano.shared(init_params([input_size, output_size]), self.name+"_W")
        self._b = theano.shared(init_params([output_size], shift=bias_shift), self.name+"_W")
        self.dropout_keep = dropout_keep

    @property
    def params(self):
        return [self._W, self._b]

    def dropout_masks(self, srng):
        if self.dropout_keep == 1:
            return []
        else:
            return [make_dropout_mask((self.input_size,), self.dropout_keep, srng)]

    def split_dropout_masks(self, dropout_masks):
        if dropout_masks is None:
            return [], None
        idx = (self.dropout_keep != 1)
        return dropout_masks[:idx], dropout_masks[idx:]

    def process(self, ipt, dropout_masks=Ellipsis):
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True
        if self.dropout_keep != 1 and dropout_masks not in ([], None):
            ipt = apply_dropout(ipt, dropout_masks[0])
            dropout_masks = dropout_masks[1:]
        xW = T.dot(ipt, self._W)
        b = T.shape_padleft(self._b)
        if append_masks:
            return self.activation( xW + b ), dropout_masks
        else:
            return self.activation( xW + b )

class LayerStack(object):
    def __init__(self, input_size, output_size, hidden_sizes=[], bias_shift=0.0, name=None, hidden_activation=T.tanh, activation=identity, dropout_keep=1, dropout_input=True, dropout_output=False):
        self.input_size = input_size
        self.output_size =output_size
        self.name = name if name is not None else get_unique_name(type(self))

        self.dropout_keep = dropout_keep
        self.dropout_output = dropout_output

        self.layers = []
        for i, isize, osize in zip(itertools.count(),
                                    [input_size]+hidden_sizes,
                                    hidden_sizes+[output_size]):
            cur_dropout_keep = 1 if (i==0 and not dropout_input) else dropout_keep
            if i == len(hidden_sizes):
                # Last layer
                self.layers.append(Layer(isize, osize, bias_shift=bias_shift, name="{}[output]".format(self.name), activation=activation, dropout_keep=cur_dropout_keep))
            else:
                self.layers.append(Layer(isize, osize, name="{}[hidden{}]".format(self.name,i), activation=hidden_activation, dropout_keep=cur_dropout_keep))

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def dropout_masks(self, srng):
        masks = [mask for layer in self.layers for mask in layer.dropout_masks(srng)]
        if self.dropout_keep != 1 and self.dropout_output:
            masks.append(make_dropout_mask((self.output_size,), self.dropout_keep, srng))
        return masks

    def split_dropout_masks(self, dropout_masks):
        if dropout_masks is None:
            return [], None
        used = []
        for layer in self.layers:
            new_used, dropout_masks = layer.split_dropout_masks(dropout_masks)
            used.extend(new_used)
        if self.dropout_keep != 1 and self.dropout_output:
            used.append(dropout_masks[0])
            dropout_masks = dropout_masks[1:]
        return used, dropout_masks

    def process(self, ipt, dropout_masks=Ellipsis):
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True
        val = ipt
        for layer in self.layers:
            val, dropout_masks = layer.process(val, dropout_masks)
        if self.dropout_keep != 1 and self.dropout_output and dropout_masks not in ([], None):
            val = apply_dropout(val, dropout_masks[0])
            dropout_masks = dropout_masks[1:]
        if append_masks:
            return val, dropout_masks
        else:
            return val



