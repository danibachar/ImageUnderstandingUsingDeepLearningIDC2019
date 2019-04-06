import numpy as np
from random import randrange

def perceptron_loss_naive(W, X, y):

  """
  Structured perceptron loss function, naive implementation (with loops)
  Inputs:
  - W: array of weights
  - X: array of data
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Compute the perceptron loss as learned in class. Start by iterating over  #
  # over all instances and calculate the score and true score for each.       #
  # Now, for each class determine if the prediction is correct and update the #
  # loss over all mistakes.                                                   #
  # Compute the gradient of the loss function and store it as dW.             #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed.                                                   #
  #############################################################################
  # print(W.shape)
  classes_count = W.shape[1] # col count = classes count
  example_count = X.shape[0] # row count = number of samples
  for i in range(example_count): # For each sample
      scores = X[i].dot(W) # get the relevant sample vector and "multiple" byt the weights to get the prediction
      correct_class_score = scores[y[i]] # get the correct label - the actual score provided
      loss_contributors_count = 0
      for j in range(classes_count):
          if j == y[i]: # skipping correct predictions
              continue
          margin = scores[j] - correct_class_score + 1
          loss += max(0, margin)
          if margin > 0:
              # incorrect class gradient part
              dW[:, j] += X[i]
              # count contributor terms to loss function
              loss_contributors_count += 1
      # correct class gradient part
      dW[:, y[i]] += (-1) * loss_contributors_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by example_count.
  loss /= example_count
  dW /= example_count
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def perceptron_loss_vectorized(W, X, y):
  """
  Vectorized version of perceptron_loss_naive. instead of loops, should use 
  numpy vectorization.

  Inputs and outputs are the same as perceptron_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the perceptron loss, storing the       #
  # result in loss and the gradient in dW                                     #
  #############################################################################
  scores = X.dot(W)
  example_count = X.shape[0]
  # print(example_count)
  # print(y.shape)
  correct_score = scores[list(range(example_count)),y]
  # print(correct_score)
  correct_score = correct_score.reshape(example_count, -1)
  # print(correct_score.shape)
  scores += 1 - correct_score
  # make sure correct scores themselves don't contribute to loss function
  scores[list(range(example_count)), y] = 0
  # construct loss function - according to the perceptron loss presented in class
  loss = np.sum(np.fmax(scores, 0)) / example_count

  X_mask = np.zeros(scores.shape)
  X_mask[scores > 0] = 1
  X_mask[np.arange(example_count), y] = -np.sum(X_mask, axis=1)
  dW = X.T.dot(X_mask)
  dW /= example_count

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def sigmoid(x):
  """
  Numerically stable Sigmoid function.

  Input: any unnormalized log probabilities vector
  Output: normalized probabilities
  """
  #############################################################################
  # TODO:                                                                     #
  # Implement the function                                                    #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return x


def binary_cross_entropy(W, X, y):
  """
  Structured BCE loss function. Implement this function using vectorized code.
  Inputs:
  - W: array of weights
  - X: array of data
  - y: 1-dimensional array of length N with binary labels (0,1). 
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement the function and store result in loss and the gradint in dW     #
  # Note: in class you defined BCE that takes values from the range (-1,1).   #
  # and the sigmoid function generally outputs values in the range (0,1).     #
  # Make the proper adjustments for your code to work.                        #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
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
