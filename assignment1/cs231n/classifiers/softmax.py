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
        stable_scores = raw_scores - np.max(raw_scores)
        class_score = stable_scores[y[index]]
        exp_scores = np.exp(stable_scores)
        # loss += -1 * np.log(exp_scores[y[index]] / sum(exp_scores))
        loss += -1 * class_score + np.log(np.sum(exp_scores))

        for j in range(W.shape[1]):
            dW[:, j] = dW[:, j] + (exp_scores[j] / np.sum(exp_scores)) * X[index]

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
    # compute all scores in single matrix multiplication
    raw_scores = np.matmul(X, W)
    # find the by-row maximums, and use them to shift the scores to ensure stability
    row_maxes = raw_scores[np.arange(training_set_size), np.argmax(raw_scores, 1)]
    row_maxes.shape = (training_set_size, 1)
    stable_scores = raw_scores - row_maxes
    # handle the gradient component for the correct class scores
    loss -= np.sum(stable_scores[np.arange(training_set_size), y])

    # exponentiate scores and find the sum by row, to normalize
    exp_scores = np.exp(stable_scores)
    row_sums = np.sum(exp_scores, 1)
    row_sums.shape = (training_set_size, 1)

    # add the sum of the log of the row sums (second component of loss)
    loss += np.sum(np.log(row_sums))
    # take as a weighted average
    loss /= training_set_size
    # add regularization term
    loss += reg * np.sum(W * W)

    # compute all normalized scores as a matrix
    normalized_scores = exp_scores / row_sums
    # subtract -1's for correct classes
    normalized_scores[np.arange(y.shape[0]), y] -= 1
    # compute gradient as matrix product
    dW = np.matmul(X.T, normalized_scores)

    # take as weighted average
    dW /= training_set_size
    # add regularization gradient
    dW += reg * 2 * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

