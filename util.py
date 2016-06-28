import theano
import theano.tensor as T
import numpy as np

def init_params(shape, stddev=0.1, shift=0.0):
    """Get an initial value for a parameter"""
    return np.random.normal(shift, stddev, shape)

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
    b = T.shape_padright(biases)
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
    newtensor = T.zeros(shape)
    slices = tuple(slice(0,s) for s in tensor.shape)
    return T.set_subtensor(newtensor[slices], tensor)
