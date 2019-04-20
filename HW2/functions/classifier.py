import numpy as np
from .losses import *

class LogisticRegression():
    # Classifer that uses softmax + cross entropy loss

    def __init__(self, X, y):
        num_features = X.shape[1]
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        self.W = 0.0001 * np.random.randn(num_features, num_classes)

    def predict(self, X):
        """
        Use the weight of the classifier to predict a label. 
        Input: 2D array of size (num_instances, num_features).
        Output: 1-dimensional array of length num_instances, and 
        each element is an integer giving the predicted class.        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def calc_accuracy(self, X, y):

        accuracy = 0.0
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return accuracy

    def train(self, X, y, learning_rate=1e-3, reg=0, num_iters=100, batch_size=200, verbose=False):
        
        num_instances, num_features = X.shape
        num_classes = np.max(y) + 1

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def loss(self, X_batch, y_batch, reg=0):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)





