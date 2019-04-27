import numpy as np
from random import shuffle
from random import randrange

def softmax_loss_vectorized(W, X, y, reg=0):
    """
       Softmax loss function, vectorized version.
       Inputs and outputs are the same as softmax_loss_naive.
       """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    def softmax(X):
        e = np.exp(X)
        return e / np.sum(e)

    f = X.dot(W)
    f -= f.max()
    loss = -np.sum(
        np.log(np.exp(f[np.arange(num_train), y]) / np.sum(np.exp(f), axis=1))
    )
    loss /= num_train
    loss += reg * np.sum(W * W)
    ind = np.zeros_like(f)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True) - ind)
    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def grad_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print ('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
