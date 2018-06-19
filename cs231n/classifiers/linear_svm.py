import numpy as np


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
    dW = np.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        x_i = X[i]
        scores = x_i.dot(W)
        correct_class = y[i]
        correct_class_score = scores[correct_class]
        dW_i = np.zeros_like(dW)
        for j in range(num_classes):
            if j == correct_class:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW_i[:, j] = x_i
                dW_i[:, correct_class] -= x_i
        dW += dW_i

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
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X @ W
    correct_class_scores = scores[range(num_train), y].reshape(-1, 1)
    margins = scores - correct_class_scores + 1
    loss = np.sum(margins.clip(0))
    loss -= num_train
    loss /= num_train
    loss += reg * np.sum(W ** 2)
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
    indicated_positive_margins = np.where(margins > 0, 1, 0)
    indicated_positive_margins[range(num_train), y] = - (indicated_positive_margins.sum(axis=1) - 1)
    dW = X.T @ indicated_positive_margins
    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
