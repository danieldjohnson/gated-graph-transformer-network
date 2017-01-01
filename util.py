import theano
import theano.tensor as T
import numpy as np
import pickle
import hashlib
import json
import enum
import inspect
import os

import itertools
import collections

EPSILON = np.array(1e-8, np.float32)

def identity(x):
    return x

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

def save_params(params, file):
    """
    Save params into a pickle file
    """
    values = [param.get_value() for param in params]
    pickle.dump(values, file)

def load_params(params, file):
    """
    Load params from a pickle file
    """
    values = pickle.load(file)
    for param,value in zip(params, values):
        try:
            param.set_value(value)
        except:
            param.set_value(value.get_value())

def set_params(params, saved_params):
    """
    Copies saved_params into params (both must be theano shared variables)
    """
    for param,saved in zip(params, saved_params):
        param.set_value(saved.get_value())

def reduce_log_sum(tensor, axis=None, guaranteed_finite=False):
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

def shape_padaxes(tensor, axes):
    for axis in axes:
        tensor = T.shape_padaxis(tensor, axis)
    return tensor

idx_map = collections.defaultdict(lambda:0)
def get_unique_name(cls):
    name = "{}{}".format(cls.__name__, idx_map[cls])
    idx_map[cls] += 1
    return name

def independent_best(tensor):
    """
    tensor should be a tensor of probabilities
    Return a new tensor of maximum likelihood, i.e. 1 in each position if p>0.5,
    else 0
    """
    return T.cast(T.ge(tensor, 0.5), 'floatX')

def categorical_best(tensor):
    """
    tensor should be a tensor of shape (..., categories)
    Return a new tensor of the same shape but one-hot at position of best category
    """
    flat_tensor = tensor.reshape([-1, tensor.shape[-1]])
    argmax_posns = T.argmax(flat_tensor, 1)
    flat_snapped = T.zeros_like(flat_tensor)
    flat_snapped = T.set_subtensor(flat_snapped[T.arange(flat_tensor.shape[0]), argmax_posns], 1.0)
    snapped = flat_snapped.reshape(tensor.shape)
    return snapped

def make_dropout_mask(shape, keep_frac, srng):
    return T.shape_padleft(T.cast(srng.binomial(shape, p=keep_frac), 'float32') / keep_frac)

def apply_dropout(ipt, dropout):
    return ipt * dropout

def sample_gumbel(srng, shape):
    u = srng.uniform(shape)
    return -T.log(-T.log(u))

def make_concrete_sampler(srng, temp):
    """
    Create a function that will sample from a Concrete distribution
    with the given temperature, and with parameters logalpha, where
        logalpha in (-\infty, \infty)
    This functions like a softmax.
    """
    def sample_concrete(logalpha):
        gumbel = sample_gumbel(srng, logalpha.shape)
        x = (logalpha + gumbel)/temp
        return T.nnet.softmax(x)
    return sample_concrete

def sample_logistic(srng, shape):
    u = srng.uniform(shape)
    return T.log(u) - T.log(1-u)

def make_binconcrete_sampler(srng, temp):
    """
    Create a function that will sample from a BinConcrete distribution
    with the given temperature, and with parameters logalpha, where
        logalpha in (-\infty, \infty)
    This functions like a sigmoid.
    """
    def sample_binconcrete(logalpha):
        L = sample_logistic(srng, logalpha.shape)
        x = (logalpha + L)/temp
        return T.nnet.sigmoid(x)
    return sample_binconcrete

def object_hash(thing):
    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, enum.Enum):
                return obj.name
            return super().default(obj)
    strform = json.dumps(thing, sort_keys=True, cls=EnumEncoder)
    h = hashlib.sha1()
    h.update(strform.encode('utf-8'))
    return h.hexdigest()

def get_compatible_kwargs(function, kwargs):
    kwargs = dict(kwargs)
    sig = inspect.signature(function)
    for param in sig.parameters.values():
        if param.name not in kwargs:
            if param.default is inspect.Parameter.empty:
                raise TypeError("kwargs missing required argument '{}'".format(param.name))
            else:
                kwargs[param.name] = param.default
    return kwargs

def find_recent_params(outputdir):
    files_list = list(os.listdir(outputdir))
    numbers = [int(x[6:-2]) for x in files_list if x[:6]=="params" and x[-2:]==".p"]
    if len(numbers) == 0:
        return None
    most_recent = max(numbers)
    return most_recent, os.path.join(outputdir,"params{}.p".format(most_recent))
