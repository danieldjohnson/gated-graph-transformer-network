import theano
import theano.tensor as T
import numpy as np

from util import *

class BaseGRULayer( object ):
    """
    Implements a GRU layer
    """

    def __init__(self, input_width, output_width, activation_shift=0.0, name=None, dropout_keep=1, dropout_input=False, dropout_output=True):
        """
        Params:
            input_width: Width of input
            output_width: Width of the GRU output
            activation_shift: How to shift the biases of the activation
        """
        self._input_width = input_width
        self._output_width = output_width

        prefix = "" if name is None else name + "_"

        self._reset_W = theano.shared(init_params([input_width + output_width, output_width]), prefix+"reset_W")
        self._reset_b = theano.shared(init_params([output_width], shift=1.0), prefix+"reset_b")

        self._update_W = theano.shared(init_params([input_width + output_width, output_width]), prefix+"update_W")
        self._update_b = theano.shared(init_params([output_width], shift=1.0), prefix+"update_b")

        self._activation_W = theano.shared(init_params([input_width + output_width, output_width]), prefix+"activation_W")
        self._activation_b = theano.shared(init_params([output_width], shift=activation_shift), prefix+"activation_b")

        self._dropout_keep = dropout_keep
        self._dropout_input = dropout_input
        self._dropout_output = dropout_output

    @property
    def input_width(self):
        return self._input_width

    @property
    def output_width(self):
        return self._output_width

    @property
    def params(self):
        return [self._reset_W, self._reset_b, self._update_W, self._update_b, self._activation_W, self._activation_b]

    def initial_state(self, batch_size):
        """
        The initial state of the network
        Params:
            batch_size: The batch size to construct the initial state for
        """
        return T.zeros([batch_size, self.output_width])

    def dropout_masks(self, srng, use_output=None):
        if self._dropout_keep == 1:
            return []
        else:
            masks = []
            if self._dropout_input:
                masks.append(make_dropout_mask((self._input_width,), self._dropout_keep, srng))
            if self._dropout_output:
                if use_output is not None:
                    masks.append(use_output)
                else:
                    masks.append(make_dropout_mask((self._output_width,), self._dropout_keep, srng))
            return masks

    def split_dropout_masks(self, dropout_masks):
        if dropout_masks is None:
            return [], None
        idx = (self._dropout_keep != 1) * (self._dropout_input + self._dropout_output)
        return dropout_masks[:idx], dropout_masks[idx:]

    def step(self, ipt, state, dropout_masks=Ellipsis):
        """
        Perform a single step of the network

        Params:
            ipt: The current input. Should be an int tensor of shape (n_batch, self.input_width)
            state: The previous state. Should be a float tensor of shape (n_batch, self.output_width)
            dropout_masks: Masks from get_dropout_masks

        Returns: The next output state
        """
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True

        if self._dropout_keep != 1 and self._dropout_input and dropout_masks is not None:
                ipt_masks = dropout_masks[0]
                ipt = apply_dropout(ipt, ipt_masks)
                dropout_masks = dropout_masks[1:]

        cat_ipt_state = T.concatenate([ipt, state], 1)
        reset = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._reset_W, self._reset_b )
        update = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._update_W, self._update_b )
        candidate_act = do_layer( T.tanh, T.concatenate([ipt, (reset * state)], 1),
                            self._activation_W, self._activation_b )

        newstate = update * state + (1-update) * candidate_act

        if self._dropout_keep != 1 and self._dropout_output and dropout_masks is not None:
                newstate_masks = dropout_masks[0]
                newstate = apply_dropout(newstate, newstate_masks)
                dropout_masks = dropout_masks[1:]

        if append_masks:
            return newstate, dropout_masks
        else:
            return newstate
