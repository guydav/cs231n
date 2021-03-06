import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_classes = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            
            if margin > 0:
                loss += margin
                loss_classes += 1
                dW[:, j] = dW[:, j] + X[i]

        dW[:, y[i]] = dW[:, y[i]] - loss_classes * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # compute all scores as matrix product
    all_scores = np.matmul(X, W)
    # pull aside and reshape correct class (y_i) scores
    correct_class_scores = all_scores[np.arange(y.shape[0]), y]
    correct_class_scores.shape = (correct_class_scores.shape[0], 1)
    # calculate the margins
    margins = all_scores - correct_class_scores + 1
    loss_contributions = np.where(margins > 0, 1, 0)
    # remove the loss contribution for the correct class
    loss = np.sum(np.where(loss_contributions, margins, 0)) - num_train
    # change into weighted average rather than naive sum
    loss /= num_train
    # add regularization
    loss += reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # calculate how many classes contributed to loss in each training example, subtracting 1 for y_i
    loss_contributions_by_row = (np.sum(loss_contributions, 1) - 1) * -1
    augmented_loss_contributions = loss_contributions
    # replace the value for the correct classes (y_i) with the total number of contributing classes
    augmented_loss_contributions[np.arange(y.shape[0]), y] = loss_contributions_by_row
    # calculate gradient as matrix product
    dW = np.matmul(X.T, loss_contributions) / num_train
    # add regularization gradient
    dW += reg * 2 * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
