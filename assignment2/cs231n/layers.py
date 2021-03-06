from builtins import range
import numpy as np
from functools import reduce


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_shape = x.shape
    x = np.reshape(x, (x_shape[0], np.prod(x_shape[1:])))
    out = np.matmul(x, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, x_shape)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b, x_shape = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.matmul(dout, w.T)
    dx = dx.reshape(x_shape)
    dw = np.matmul(x.T, dout)
    db = np.sum(dout, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x > 0, dout, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.mean(x, 0, keepdims=True)
        sample_var = np.var(x, 0, keepdims=True)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta

        cache = x, sample_mean, sample_var, x_hat, gamma, beta, eps

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / (np.sqrt(running_var) + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

"""
mu = 1. / N * np.sum(x, axis=0)
# step2: subtract mean vector of every trainings example
xmu = x - mu
# step3: following the lower branch - calculation denominator
sq = xmu ** 2
# step4: calculate variance
var = 1. / N * np.sum(sq, axis=0)
# step5: add eps for numerical stability, then sqrt
sqrtvar = np.sqrt(var + eps)
# step6: invert sqrtwar
ivar = 1. / sqrtvar
# step7: execute normalization
xhat = xmu * ivar
# step8: Nor the two transformation steps
gammax = gamma * xhat
# step9
alt_out = gammax + beta
cache = (cache, (xhat, gamma, xmu, ivar, sqrtvar, var, eps))
"""


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, sample_mean, sample_var, x_hat, gamma, beta, eps = cache

    N, D = x.shape

    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    dbeta = np.sum(dout, 0)  # dbeta = sum of each column
    dgamma = np.sum(x_hat * dout, 0)  # dgamma = sum of each column of element-wise product
    dx_hat = dout * gamma
    dnumerator = dx_hat / np.sqrt(sample_var + eps)
    ddenominator = np.sum(dx_hat * (x - sample_mean), 0)

    dvariance = ddenominator * (-0.5) * ((sample_var + eps) ** (-1.5))
    # dvariancesqrt = -1.0 / (sample_var + eps) * ddenominator
    # dvariance = 0.5 / np.sqrt(sample_var + eps) * dvariancesqrt

    dvariance_sum = dvariance / N
    dvariance_square = dvariance_sum * 2 * (x - sample_mean)

    dx_minus_xbar_in = (dnumerator + dvariance_square)
    dxbar = -1 * np.sum(dx_minus_xbar_in, 0)
    dxbar_summation = dxbar / N
    dx = dx_minus_xbar_in + dxbar_summation

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

"""
my_cache, other_cache = cache
x, sample_mean, sample_var, x_hat, gamma, beta, eps = my_cache

...

old_dx = dx

xhat, gamma, xmu, ivar, sqrtvar, var, eps = other_cache
# get the dimensions of the input/output
N, D = dout.shape
# step9
dbeta = np.sum(dout, axis=0)
dgammax = dout  # not necessary, but more understandable
# step8
dgamma = np.sum(dgammax * xhat, axis=0)
dxhat = dgammax * gamma
# step7
divar = np.sum(dxhat * xmu, axis=0)
dxmu1 = dxhat * ivar
# step6
dsqrtvar = -1. / (sqrtvar ** 2) * divar
# step5
dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
# step4
dsq = 1. / N * np.ones((N, D)) * dvar
# step3
dxmu2 = 2 * xmu * dsq
# step2
dx1 = (dxmu1 + dxmu2)
# print('{:#^60}'.format(' denumerator '))
# print(dnumerator)
# print()
# print('{:#^60}'.format(' dxmu1 '))
# print(dxmu1)
# print()
# print(dxmu1 - dnumerator)
# print()
# print('{:#^60}'.format(' dvariance_square '))
# print(dvariance_square)
# print()
# print('{:#^60}'.format(' dxmu2 '))
# print(dxmu2)
# print()
# print(dxmu2 - dvariance_square)
# print()
# print('{:#^60}'.format(' dx_minus_xbar_in '))
# print(dx_minus_xbar_in)
# print()
# print('{:#^60}'.format(' dx1 '))
# print(dx1)
# print()
print(dx1 - dx_minus_xbar_in)
# print()

dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
# step1
dx2 = 1. / N * np.ones((N, D)) * dmu
# step0
dx = dx1 + dx2
print()
print(old_dx - dx)

"""



def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, sample_mean, sample_var, x_hat, gamma, beta, eps = cache
    N, D = x.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    dbeta = np.sum(dout, 0)  # dbeta = sum of each column
    dgamma = np.sum(x_hat * dout, 0)  # dgamma = sum of each column of element-wise product

    # TODO: כןס איןד שא דםצק פםןמא

    dx = dout * gamma / N * ((sample_var + eps) ** (-0.5)) * (N - 1 - ((x - sample_mean) ** 2 / (sample_var + eps)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (7_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    out = np.zeros((N, F, int(1 + (H + 2 * pad - HH) / stride), int(1 + (W + 2 * pad - WW) / stride)))
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), ), mode='constant')
    for index in range(N):
            for h_start in range(0, H + 2 * pad - HH + 1, stride):
                for w_start in range(0, W + 2 * pad - WW + 1, stride):
                    section = padded_x[index, :, h_start:h_start + HH, w_start:w_start + WW]
                    for filter_index in range(F):
                        conv_filter = w[filter_index]
                        bias = b[filter_index]
                        # print(index, filter_index, int(h_start / stride), int(w_start / stride))
                        out[index, filter_index, int(h_start / stride), int(w_start / stride)] = \
                            np.sum(section * conv_filter) + bias

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # print(dout.shape, dx.shape, dw.shape, db.shape, stride, pad)

    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad),), mode='constant')
    d_padded_x = np.zeros_like(padded_x)

    for index in range(N):
        for h_start in range(0, H + 2 * pad - HH + 1, stride):
            for w_start in range(0, W + 2 * pad - WW + 1, stride):
                section = padded_x[index, :, h_start:h_start + HH, w_start:w_start + WW]
                for filter_index in range(F):
                    conv_filter = w[filter_index]
                    curr_gradient = dout[index, filter_index, int(h_start / stride), int(w_start / stride)]
                    db[filter_index] += curr_gradient
                    dw[filter_index] += curr_gradient * section
                    d_padded_x[index, :, h_start:h_start + HH, w_start:w_start + WW] += curr_gradient * conv_filter

    dx = d_padded_x[:, :, pad:-1 * pad, pad:-1 * pad]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    height, width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    out = np.zeros((N, C, int(1 + (H - height) / stride), int(1 + (W - width) / stride)))
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    for index in range(N):
        for channel in range(C):
            for h_start in range(0, H - height + 1, stride):
                for w_start in range(0, W - width + 1, stride):
                    section = x[index, channel, h_start:h_start + stride, w_start:w_start + stride]
                    out[index, channel, int(h_start / stride), int(w_start / stride)] = np.amax(section)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    height, width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    dx = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    for index in range(N):
        for channel in range(C):
            for h_start in range(0, H - height + 1, stride):
                for w_start in range(0, W - width + 1, stride):
                    section = x[index, channel, h_start:h_start + stride, w_start:w_start + stride]
                    curr_gradient = dout[index, channel, int(h_start / stride), int(w_start / stride)]
                    max_index = np.argmax(section)
                    dx[index, channel,
                         int(h_start + np.floor(max_index / width)),
                         int(w_start + max_index % width)] = curr_gradient

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape

    out, cache, channel_caches = np.zeros_like(x), None, []
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # gamma.shape = (gamma.shape[0], 1)
    # beta.shape = (beta.shape[0], 1)
    out, bn_cache = batchnorm_forward(x.transpose((0, 2, 3, 1)).reshape((N * H * W, C)),
                                   gamma,
                                   beta,
                                   bn_param)
    out = out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
    cache = (x, gamma, beta, bn_cache)

    # for channel in range(C):
    #     channel_slice = np.reshape(x[:, channel, :, :].copy(), (N, H * W))
    #     channel_out, channel_cache = batchnorm_forward(channel_slice,
    #                                                    np.ones_like(channel_slice),
    #                                                    np.zeros_like(channel_slice),
    #                                                    bn_param)
    #     channel_caches.append((channel_out, channel_cache))
    #     out[:, channel, :, :] = np.reshape(channel_out * gamma[channel] + beta[channel], (N, H, W))
    #
    # cache = (x, gamma, beta, channel_caches)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    x, gamma, beta, bn_cache = cache
    # dx, dgamma, dbeta = np.zeros_like(x), np.zeros_like(gamma), np.zeros_like(beta)

    N, C, H, W = x.shape
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    # dgamma.shape = (gamma.shape[0], 1)
    # dbeta.shape = (beta.shape[0], 1)
    dx, dgamma, dbeta = batchnorm_backward(dout.transpose((0, 2, 3, 1)).reshape((N * H * W, C)), bn_cache)
    dx = dx.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

    # dbeta = np.sum(dout, (0, 2, 3))
    # dgamma = np.sum(dout * x, (0, 2, 3))
    #
    # for channel in range(C):
    #     channel_out, channel_cache = channel_caches[channel]
    #
    #     channel_slice = np.reshape(x[:, channel, :, :].copy(), (N, H * W))
    #     dout_channel_slice = np.reshape(dout[:, channel, :, :].copy(), (N, H * W)) * gamma[channel]
    #
    #     dchannel_slice, _, _ = batchnorm_backward(dout_channel_slice, channel_cache)
    #     dx[:, channel, :, :] = np.reshape(dchannel_slice, (N, H, W))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
