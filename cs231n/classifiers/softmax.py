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
    dW = np.zeros_like(W)
    num_samples = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_samples):
        scores = (X[i] @ W).T  # shape (C, 1)
        max_score = np.max(scores)
        normalised_correct_class_score = \
            np.exp(scores[y[i]] - max_score) \
            / np.sum(np.exp(scores - max_score))
        loss += - np.log(normalised_correct_class_score)
        dW_i = np.zeros_like(dW)
        exp_sum = .0
        for j in range(num_classes):
            if j == y[i]:
                exp_sum += 1
                continue
            score_diff = X[i] @ (W[:, j] - W[:, y[i]])  # TODO: optimise
            exp_score_diff = np.exp(score_diff)
            dW_i[:, j] = exp_score_diff * X[i]
            exp_sum += exp_score_diff
        dW_i[:, y[i]] = - X[i] * (exp_sum - 1)
        dW_i /= exp_sum
        dW += dW_i
    loss /= num_samples
    dW /= num_samples
    loss += reg * np.sum(W ** 2)
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
