import theano
import theano.tensor as T
import numpy as np

from util import *

class StrengthWeightedGRULayer( object ):
    """
    Implements a strength-weighted GRU layer
    """

    def __init__(self, input_width, output_width, activation_shift=0.0, name=None):
        """
        Params:
            input_width: Width of input.
            output_width: Width of the GRU output
            activation_shift: How to shift the biases of the activation
        """
        self._input_width = input_width
        self._output_width = output_width

        prefix = "" if name is None else name + "_"

        self._reset_W = theano.shared(init_params([input_width + output_width, output_width]), prefix+"reset_W")
        self._reset_b = theano.shared(init_params([output_width], shift=1.0), prefix+"reset_b")

        self._update_W = theano.shared(init_params([input_width + output_width, output_width+1]), prefix+"update_W")
        self._update_b = theano.shared(init_params([output_width+1], shift=1.0), prefix+"update_b")

        self._activation_W = theano.shared(init_params([input_width + output_width, output_width]), prefix+"activation_W")
        self._activation_b = theano.shared(init_params([output_width], shift=activation_shift), prefix+"activation_b")

        self._strength_W = theano.shared(init_params([input_width + output_width, 1]), prefix+"strength_W")
        self._strength_b = theano.shared(init_params([1], shift=1.0), prefix+"strength_b")

    @property
    def input_width(self):
        return self._input_width

    @property
    def output_width(self):
        return self._output_width

    @property
    def params(self):
        return [self._reset_W, self._reset_b, self._update_W, self._update_b, self._activation_W, self._activation_b, self._strength_W, self._strength_b]

    @property
    def num_dropout_masks(self):
        return 2

    def get_dropout_masks(self, srng, keep_frac):
        """
        Get dropout masks for the GRU.
        """
        return [T.shape_padleft(T.cast(srng.binomial((self._input_width,), p=keep_frac), 'float32') / keep_frac),
                T.shape_padleft(T.cast(srng.binomial((self._output_width,), p=keep_frac), 'float32') / keep_frac)]

    def step(self, ipt, state, state_strength, dropout_masks=None):
        """
        Perform a single step of the network

        Params:
            ipt: The current input. Should be an int tensor of shape (n_batch, self.input_width)
            state: The previous state. Should be a float tensor of shape (n_batch, self.output_width)
            state_strength: Strength of the previous state. Should be a float tensor of shape
                (n_batch)
            dropout_masks: Masks from get_dropout_masks

        Returns: The next output state, and the next output strength
        """
        if dropout_masks is not None:
            ipt_masks, state_masks = dropout_masks
            ipt = ipt*ipt_masks
            state = state*state_masks

        obs_state = state * T.shape_padright(state_strength)
        cat_ipt_state = T.concatenate([ipt, obs_state], 1)
        reset = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._reset_W, self._reset_b )
        update = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._update_W, self._update_b )
        update_state = update[:,:-1]
        update_strength = update[:,-1]

        cat_reset_ipt_state = T.concatenate([ipt, (reset * obs_state)], 1)
        candidate_act = do_layer( T.tanh, cat_reset_ipt_state,
                            self._activation_W, self._activation_b )
        candidate_strength = do_layer( T.nnet.sigmoid, cat_reset_ipt_state,
                            self._strength_W, self._strength_b ).reshape(state_strength.shape)

        newstate = update_state * state + (1-update_state) * candidate_act
        newstrength = update_strength * state_strength + (1-update_strength) * candidate_strength

        return newstate, newstrength
