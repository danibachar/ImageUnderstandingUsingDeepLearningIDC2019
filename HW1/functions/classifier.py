import numpy as np
import random
from .losses import *

class LinearClassifier(object):

    def __init__(self, X, y):
        """
        Class constructor. Use this method to initiate the parameters of
        your model (W)
        *** Subclasses will override this. ***

        Inputs:
        - X: array of data - shape = [n_samples, n_features]
        - y: 1-dimensional array of length N with binary labels - shape = [n_samples]

        This function has no return value

        """
        pass        

    def predict(self, X):

        """
        Use the weight of the classifier to predict a label. 
        *** Subclasses will override this. ***

        Input: 2D array of size (num_instances, num_features).
        Output: 1D array of class predictions (num_instances, 1). 
        """
        pass

    def calc_accuracy(self, X, y):

        """
        Calculate the accuracy on a dataset as the percentage of instances 
        that are classified correctly. 

        Inputs:
        - W: array of weights
        - X: array of data
        - y: 1-dimensional array of length N with binary labels
        Returns:
        - accuracy as a single float
        """
        accuracy = 0.0
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################

        y_pred = self.predict(X)
        if len(y_pred) != len(y):
            raise Exception('Fatal Error in dim - please checkout your prediction code!')
        accuracy = np.sum(y_pred == y)/len(y)*100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return accuracy



    def train(self, X, y, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):


        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in every iteration.                       #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch                                                               #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #                                                                       #
        # Next, calculate the loss and gradient and update the weights using    #
        # the learning rate. Use the loss_history array to save the loss on     #
        # iteration to visualize the loss.                                      #
        #########################################################################
        num_train, dim = X.shape

        loss_history = []
        for i in range(num_iters):
            ###########################################################################
            #                          START OF YOUR CODE                             #
            ###########################################################################
            batch_ind = np.random.choice(num_train, batch_size,replace = True)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            loss, grad = self.loss(X_batch, y_batch)
            # print('loss = {}'.format(loss))
            # print('grad shape = {}'.format(grad.shape))
            # print('W shape = {}'.format(self.W.shape))
            loss_history.append(loss)
            # Update the weights
            self.W += - learning_rate * grad
            ###########################################################################
            #                           END OF YOUR CODE                              #
            ###########################################################################

            if verbose and i % 100 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history


    def loss(self, X, y):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass



class LinearPerceptron(LinearClassifier):
    # Classifier that uses Perceptron loss

    def __init__(self, X, y):
        self.W = None
        ###########################################################################
        # TODO:                                                                   #
        # Initiate the parameters of your model.                                  #
        ###########################################################################
        num_of_features = X.shape[1]
        self.W = np.random.randn(num_of_features, 2) * 0.0001
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################


    def predict(self, X):
        y_pred = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        # Predict according to weights
        scores = X.dot(self.W)
        # Activations
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred


    def loss(self, X_batch, y_batch):
        return perceptron_loss_vectorized(self.W, X_batch, y_batch)


class LogisticRegression(LinearClassifier):
    # Classifer that uses sigmoid and binary cross entropy loss

    def __init__(self, X, y):
        self.W = None
        ###########################################################################
        # TODO:                                                                   #
        # Initiate the parameters of your model.                                  #
        ###########################################################################
        num_of_features = X.shape[1]
        self.W = np.random.randn(num_of_features, 1) * 0.0001
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

    def predict(self, X):
        y_pred = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        # Predict according to weights
        scores = X.dot(self.W)
        # Activation
        y_pred = np.round(
            sigmoid(scores).reshape((scores.shape[0],))
        ).astype(int)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred


    def loss(self, X_batch, y_batch):
        return binary_cross_entropy(self.W, X_batch, y_batch)






