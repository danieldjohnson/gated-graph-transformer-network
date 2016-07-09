import theano
import theano.tensor as T
import numpy as np

import itertools
import collections

EPSILON = np.array(1e-8, np.float32)

def init_params(shape, stddev=0.1, shift=0.0):
    """Get an initial value for a parameter"""
    return np.float32(np.random.normal(shift, stddev, shape))

def do_layer(activation, ipt, weights, biases):
    """
    Perform a layer operation, i.e. out = activ( xW + b )
        activation: An activation function
        ipt: Tensor of shape (n_batch, X)
        weights: Tensor of shape (X, Y)
        biases: Tensor of shape (Y)

    Returns: Tensor of shape (n_batch, Y)
    """
    xW = T.dot(ipt, weights)
    b = T.shape_padleft(biases)
    return activation( xW + b )

def broadcast_concat(tensors, axis):
    """
    Broadcast tensors together, then concatenate along axis
    """
    ndim = tensors[0].ndim
    assert all(t.ndim == ndim for t in tensors), "ndims don't match for broadcast_concat: {}".format(tensors)
    broadcast_shapes = []
    for i in range(ndim):
        if i == axis:
            broadcast_shapes.append(1)
        else:
            dim_size = next((t.shape[i] for t in tensors if not t.broadcastable[i]), 1)
            broadcast_shapes.append(dim_size)
    broadcasted_tensors = []
    for t in tensors:
        tile_reps = [bshape if t.broadcastable[i] else 1 for i,bshape in enumerate(broadcast_shapes)]
        if all(rep is 1 for rep in tile_reps):
            # Don't need to broadcast this tensor
            broadcasted_tensors.append(t)
        else:
            broadcasted_tensors.append(T.tile(t, tile_reps))
    return T.concatenate(broadcasted_tensors, axis)

def pad_to(tensor, shape):
    """
    Pads tensor to shape with zeros
    """
    current = tensor
    for i in range(len(shape)):
        padding = T.zeros([(fs-ts if i==j else fs if j<i else ts) for j,(ts,fs) in enumerate(zip(tensor.shape, shape))])
        current = T.concatenate([current, padding], i)
    return current

def set_params(params, saved_params):
    """
    Copies saved_params into params (both must be theano shared variables)
    """
    for param,saved in zip(params, saved_params):
        param.set_value(saved.get_value())

def reduce_log_sum(tensor, axis=None, guaranteed_finite=True):
    """
    Sum probabilities in the log domain, i.e return
        log(e^vec[0] + e^vec[1] + ...)
        = log(e^x e^(vec[0]-x) + e^x e^(vec[1]-x) + ...)
        = log(e^x [e^(vec[0]-x) + e^(vec[1]-x) + ...])
        = log(e^x) + log(e^(vec[0]-x) + e^(vec[1]-x) + ...)
        = x + log(e^(vec[0]-x) + e^(vec[1]-x) + ...)
    For numerical stability, we choose x = max(vec)
    Note that if x is -inf, that means all values are -inf,
    so the answer should be -inf. In this case, choose x = 0
    """
    maxval = T.max(tensor, axis)
    maxval_full = T.max(tensor, axis, keepdims=True)
    if not guaranteed_finite:
        maxval = T.switch(T.isfinite(maxval), maxval, T.zeros_like(maxval))
        maxval_full = T.switch(T.isfinite(maxval_full), maxval_full, T.zeros_like(maxval_full))
    reduced_sum = T.sum(T.exp(tensor - maxval_full), axis)
    logsum = maxval + T.log(reduced_sum)
    return logsum

idx_map = collections.defaultdict(lambda:0)
def get_unique_name(cls):
    name = "{}{}".format(cls.__name__, idx_map[cls])
    idx_map[cls] += 1
    return name


