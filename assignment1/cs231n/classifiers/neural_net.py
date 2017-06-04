from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] < C. This parameter is optional; if it
            is not passed then we only return scores, and if it is passed then we
            instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
            samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
            with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # print('n, d, c', N, D, C)

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # first hidden layer + bias
        y1 = np.matmul(X, W1) + b1
        # print('y1', y1.shape)
        # ReLU activation of first hidden layer
        h1 = np.maximum(y1, 0)
        # print('h1', h1.shape)
        # output layer
        y2 = np.matmul(h1, W2) + b2
        # print('y2', y2.shape)
        scores = y2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # softmax activation of second layer
        # print(y2)
        row_maxes = y2[np.arange(N), np.argmax(y2, 1)]
        # print(row_maxes)
        row_maxes.shape = (N, 1)
        y2_stable = y2 - row_maxes
        # print(y2_stable)
        y2_exp = np.exp(y2_stable)
        # print(y2_exp)
        row_sums = np.sum(y2_exp, 1)
        row_sums.shape = (N, 1)
        # print(row_sums)
        h2 = y2_exp / row_sums
        # print(h2)
        # print(y)
        # print(h2[np.arange(N), y])

        # viewed as L_i = f_y_i + log(sigma_j(e^{f_j}))
        # loss = -1 * np.sum(y2_stable[np.arange(N), y]) + np.sum(np.log(row_sums))
        # viewed as L_i = -log(\frac{e^f_y_i}{sigma_j(e^{f_j})})
        loss = -1 * np.sum(np.log(h2[np.arange(N), y]))
        loss /= N
        loss += reg * np.sum([np.sum(x * x) for x in (W1, b1, W2, b2)])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # subtract by 1 for gradient contribution of correct class
        h2[np.arange(y.shape[0]), y] -= 1
        # these are the gradients of the raw scores
        dscores = h2
        # average each column as bias gradient and add regularization term
        db2 = np.sum(dscores, 0) / N + reg * 2 * b2
        dy2 = dscores
        # take matrix product as hidden layer weight gradient, dividing by number of training examples, adding reg
        dW2 = np.matmul(h1.T, dy2) / N + reg * 2 * W2
        # likewise for the output of the first layer, not dividing since we divide later (in db1 = ... and dW1 = ...)
        dh1 = np.matmul(dy2, W2.T)
        # ReLU gradient, taking only the entries which were positive on entry
        dy1 = np.where(y1 > 0, dh1, 0)
        # bias gradient similar to the calculation of db2
        db1 = np.sum(dy1, 0) / N + reg * 2 * b1
        # weight gradient similar to the calculation of db1
        dW1 = np.matmul(X.T, dy1) / N + reg * 2 * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # print(h1.T)
        # print(dy2)
        # print(dW2)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
                        learning_rate=1e-3, learning_rate_decay=0.95,
                        reg=5e-6, num_iters=100,
                        batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
            X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
            after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            batch_indices = np.random.choice(range(num_train), batch_size, True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for param_name in ('W1', 'b1', 'W2', 'b2'):
                self.params[param_name] -= learning_rate * grads[param_name]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
            classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
            the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
            to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        N = X.shape[0]
        y2 = self.loss(X)
        row_maxes = y2[np.arange(N), np.argmax(y2, 1)]
        # print(row_maxes)
        row_maxes.shape = (N, 1)
        y2_stable = y2 - row_maxes
        # print(y2_stable)
        y2_exp = np.exp(y2_stable)
        # print(y2_exp)
        row_sums = np.sum(y2_exp, 1)
        row_sums.shape = (N, 1)
        # print(row_sums)
        h2 = y2_exp / row_sums

        y_pred = np.argmax(h2, 1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred


