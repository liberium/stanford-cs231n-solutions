import numpy as np


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
    num_examples = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_examples):
        scores = (X[i] @ W).T  # shape (C, 1)
        dW_i = np.zeros_like(dW)
        margin_exp_sum = .0
        for j in range(num_classes):
            if j == y[i]:
                margin_exp_sum += 1.
                continue
            margin = scores[j] - scores[y[i]]
            margin_exp = np.exp(margin)
            dW_i[:, j] = margin_exp * X[i]
            margin_exp_sum += margin_exp
        loss += np.log(margin_exp_sum)
        dW_i[:, y[i]] = - X[i] * (margin_exp_sum - 1)
        dW_i /= margin_exp_sum
        dW += dW_i
    loss /= num_examples
    dW /= num_examples
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
    num_examples = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X @ W
    correct_class_scores = scores[range(num_examples), y]
    margins = scores - correct_class_scores.reshape((-1, 1))
    exp_margins = np.exp(margins)
    sum_exp_margins = np.sum(exp_margins, axis=1)
    loss = np.sum(np.log(sum_exp_margins))
    partial_loss_derivs = exp_margins
    partial_loss_derivs[range(num_examples), y] = - (sum_exp_margins - 1)
    partial_loss_derivs /= sum_exp_margins.reshape((-1, 1))
    dW = X.T @ partial_loss_derivs
    loss /= num_examples
    dW /= num_examples
    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
