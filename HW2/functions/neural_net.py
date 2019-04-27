import numpy as np
import matplotlib.pyplot as plt
from .layers import *


class ThreeLayerNet(object):
  """
  A three-layer fully-connected neural network. This network has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  In our case, we use the same hidden dimension across all hidden layers.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first two fully
  connected layers. In other words, the network has the following architecture:

  input - fc layer - ReLU - fc layer - ReLu - fc layer - softmax

  The outputs of the third fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-2):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, H)
    b2: Second layer biases; has shape (H,)
    W3: Second layer weights; has shape (H, C)
    b3: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in each of the hidden layers.
    - output_size: The number of classes C.
    """
    self.params = {}
    print('input_size = {}'.format(input_size))
    print('hidden_size = {}'.format(hidden_size))
    print('output_size = {}'.format(output_size))
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a three layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization coefficient.

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
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
    print('W1.shape = {}'.format(W1.shape))
    print('b1.shape = {}'.format(b1.shape))
    print('W2.shape = {}'.format(W2.shape))
    print('b2.shape = {}'.format(b2.shape))
    print('W3.shape = {}'.format(W3.shape))
    print('b3.shape = {}'.format(b3.shape))
    print('x.shape = {}'.format(X.shape))
    print('#####################')
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # 1) fc layer 1 - ReLU
    # Test with function
    out_1, cache_1 = fc_relu_forward(X, W1, b1)
    (X, W1, b1), relu_cache_1 = cache_1
    print('out_1.shape = {}'.format(out_1.shape))

    # Kind of working
    # f = np.dot(X, W1) + b1 #
    # print('f.shape = {}'.format(f.shape))
    # H1 = np.maximum(0, f) # ReLU 1
    # print('H1.shape = {}'.format(H1.shape))
    # H2 = np.dot(H1, W2) + b2
    # print('H2.shape = {}'.format(H2.shape))
    # scores = H2
    # print('scores.shape 1 = {}'.format(scores.shape))

    # Test with function
    out_2, cache_2 = fc_relu_forward(out_1, W2, b2)
    (out_1, W2, b2), relu_cache_2 = cache_2
    print('out_2.shape = {}'.format(out_2.shape))

    # Kind of working
    # 2) fc layer 2 - ReLU
    # out_2, cache_2 = fc_relu_forward()
    # H3 = np.maximum(0, np.dot(scores, W2) + b2)
    # print('H3 = {}'.format(H3.shape))
    # H4 = np.dot(H3, W3) + b3
    # print('H4 = {}'.format(H4.shape))
    # scores = H4
    # print('scores.shape 2 = {}'.format(scores.shape))
    # print('#####################')

    # last regulae fc
    scores, cache_3 = fc_forward(out_2, W3, b3)


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
        print('missing y, returning current scores only!')
        return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1, W2, W3. Store the result #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5. We provided a version of softmax_loss at the  #
    # end of the file. it takes the scores and labels and computes the loss and #
    # derivatives for you.                                                      #
    #############################################################################
    # TODO - use - softmax_loss(x, y)?
    loss, s = softmax_loss(scores, y)
    loos = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

    # scores -= scores.max()
    # scores = np.exp(scores)
    # scores_sums = np.sum(scores, axis=1)
    # cors = scores[range(N), y]
    # loss = cors / scores_sums
    # loss = -np.sum(np.log(loss)) / N + reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    # print('loss.shape = {}'.format(loss.shape))
    # print('scores.shape = {}'.format(scores.shape))
    # print('scores_sums.shape = {}'.format(scores_sums.shape))
    # s = np.divide(scores, scores_sums.reshape(N, 1))
    # s[range(N), y] = - (scores_sums - cors) / scores_sums
    # s /= N
    # print('s.shape = {}'.format(s.shape))
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
    # Test

    dx3, dW3, db3 = fc_backward(s, cache_3)
    print('dx3.shape = {}'.format(dx3.shape))

    dx2, dW2, db2 = fc_relu_backward(dx3, cache_2)
    print('dx2.shape = {}'.format(dx2.shape))

    dx1, dW1, db1 = fc_relu_backward(dx2, cache_1)
    print('dx1.shape = {}'.format(dx1.shape))
    print('#####################')


    # 1)
    # dW3 = H1.T.dot(s)
    print('dW3.shape = {}'.format(dW3.shape))
    # db3 = np.sum(s, axis=0)
    print('db3.shape = {}'.format(db3.shape))

    # hidden_1 = s.dot(W3.T)
    # print('hidden_1.shape 1 = {}'.format(hidden_1.shape))
    # hidden_1[H1 == 0] = 0
    # print('hidden_1.shape 2 = {}'.format(hidden_1.shape))

    # dW2 = H2.T.dot(hidden_1)
    print('dW2.shape = {}'.format(dW2.shape))
    # db2 = np.sum(hidden_1, axis=0)
    print('db2.shape = {}'.format(db2.shape))

    # hidden_2 = hidden_1.dot(W2.T)
    # print('hidden_2.shape 1 = {}'.format(hidden_2.shape))
    # hidden_2[H2 == 0] = 0
    # print('hidden_2.shape 2 = {}'.format(hidden_2.shape))

    # dW1 = X.T.dot(hidden_2)
    print('dW1.shape = {}'.format(dW1.shape))
    # db1 = np.sum(hidden_2, axis=0)
    print('db1.shape = {}'.format(db1.shape))
    # print('#####################')

    grads['W3'] = dW3 + 2 * reg * W3
    grads['b3'] = db3
    # print('grads[W3].shape = {}'.format(grads['W3'].shape))
    # print('grads[b3].shape = {}'.format(grads['b3'].shape))
    grads['W2'] = dW2 + 2 * reg * W2
    grads['b2'] = db2
    # print('grads[W2].shape = {}'.format(grads['W2'].shape))
    # print('grads[b2].shape = {}'.format(grads['b2'].shape))
    grads['W1'] = dW1 + 2 * reg * W1
    grads['b1'] = db1
    # print('grads[W1].shape = {}'.format(grads['W1'].shape))
    # print('grads[b1].shape = {}'.format(grads['b1'].shape))
    # print('#####################')
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training label.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
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
      random_idxs = np.random.choice(num_train, batch_size)
      X_batch = X[random_idxs]
      y_batch = y[random_idxs]
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
      self.params['W3'] -= learning_rate * grads['W3']
      self.params['b3'] -= learning_rate * grads['b3']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy.
      if it % iterations_per_epoch == 0:
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this three-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: data points to classify.

    Returns:
    - y_pred: predicted labels
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    max_1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
    p_1 = np.dot(max_1, self.params['W2'])

    max_2 = np.maximum(0, p_1.dot(self.params['W2']) + self.params['b2'])
    p_2 = np.dot(max_2, self.params['W3'])

    y_pred = np.argmax(p_2 + self.params['b3'], axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data
    - y: Vector of labels

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
