import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    training_set_size = X.shape[0]
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for index in range(training_set_size):
        raw_scores = X[index].dot(W)
        raw_scores -= np.max(raw_scores)
        class_score = raw_scores[y[index]]
        shifted_scores = np.exp(raw_scores)
        # loss += -1 * np.log(shifted_scores[y[index]] / sum(shifted_scores))
        loss += -1 * class_score + np.log(np.sum(shifted_scores))

        for j in range(W.shape[1]):
            dW[:, j] = dW[:, j] + (X[index] * shifted_scores[j] / np.sum(shifted_scores))

        dW[:, y[index]] = dW[:, y[index]] - X[index]

    loss /= training_set_size
    loss += reg * np.sum(W * W)

    dW /= training_set_size
    dW += reg * 2 * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    training_set_size = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    raw_scores = np.matmul(X, W)
    row_maxes = raw_scores[np.arange(training_set_size), np.argmax(raw_scores, 1)]
    row_maxes.shape = (training_set_size, 1)
    stable_scores = raw_scores - row_maxes
    loss -= np.sum(stable_scores[np.arange(training_set_size), y])
    loss += np.sum(np.log(np.sum(np.exp(stable_scores), 1)))
    loss /= training_set_size
    loss += reg * np.sum(W * W)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

