import theano
import theano.tensor as T
import numpy as np

from util import *

class BaseGRULayer( object ):
    """
    Implements a GRU layer
    """

    def __init__(self, input_width, output_width, activation_shift=0.0, name=None):
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

    def step(self, ipt, state):
        """
        Perform a single step of the network

        Params:
            ipt: The current input. Should be an int tensor of shape (n_batch, self.input_width)
            state: The previous state. Should be a float tensor of shape (n_batch, self.output_width)

        Returns: The next output state
        """
        cat_ipt_state = T.concatenate([ipt, state], 1)
        reset = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._reset_W, self._reset_b )
        update = do_layer( T.nnet.sigmoid, cat_ipt_state,
                            self._update_W, self._update_b )
        candidate_act = do_layer( T.tanh, T.concatenate([ipt, (reset * state)], 1),
                            self._activation_W, self._activation_b )

        newstate = update * state + (1-update) * candidate_act
        return newstate

    def process(self, inputs):
        """
        Process a set of inputs and return the final state

        Params:
            input_words: List of inputs. Should be an int tensor of shape (n_batch, input_len, self.input_size)

        Returns: The last output state
        """
        n_batch, input_len, _ = inputs.shape
        outputs_info = [self.initial_state(n_batch)]
        valseq = inputs.dimshuffle([1,0,2])
        all_out, _ = theano.scan(self.step, sequences=[valseq], outputs_info=outputs_info)

        # all_out is of shape (input_len, n_batch, self.output_width). We want last timestep
        return all_out[-1,:,:]
