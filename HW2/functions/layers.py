import numpy as np

def fc_forward(X, W, b):
	"""
	Computes the forward pass for an fully connected layer.
	The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	examples, where each example x[i] has shape (d_1, ..., d_k). We will
	reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	then transform it to an output vector of dimension M.
	Inputs:
	- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
	- w: A numpy array of weights, of shape (D, M)
	- b: A numpy array of biases, of shape (M,)

	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	out = None
	row_dim = X.shape[0]
	col_dim = np.prod(X.shape[1:])
	x_reshape = X.reshape(row_dim, col_dim)
	out = np.dot(x_reshape, W) + b
	cache = (X, W, b)
	return out, cache


def fc_backward(dout, cache):
	"""
	Computes the backward pass for an fully connected layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: Tuple of:
	  - X: Input data
	  - W: Weights
	  - b: Biases

	Returns a tuple of:
	- dx: Gradient with respect to X
	- dw: Gradient with respect to W
	- db: Gradient with respect to b
	"""
	x, w, b = cache
	dx, dw, db = 0, 0, 0
	x = cache[0]
	w = cache[1]
	b = cache[2]

	row_dim = x.shape[0]
	col_dim = np.prod(x.shape[1:])
	x_reshape = x.reshape(row_dim, col_dim)

	dw = x_reshape.T.dot(dout)
	dx = dout.dot(w.T).reshape(x.shape)
	db = np.sum(dout, axis=0)

	return dx, dw, db

def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = None
	out = np.maximum(0, x)
	cache = x
	return out, cache

def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	out = np.maximum(0, x)  # ReLU performed again
	out[out > 0] = 1
	dx = out * dout
	return dx

def fc_relu_forward(X, W, b):
	"""
	Forward pass for a fully connected layer followed by a ReLU

	Inputs:
	- X: Input to the fc layer
	- W, b: Weights for the fc layer

	Returns:
	- out: Output from the ReLU same shape as `a` which is the same shape as `X.shape[0]`
	- cache: Object to give to the backward pass
	"""
	#############################################################################
	# TODO: Implement the function.                                             #
	#############################################################################
	a, fc_cache = fc_forward(X, W, b)
	out, relu_cache = relu_forward(a)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (fc_cache, relu_cache)
	return out, cache


def fc_relu_backward(dout, cache):
	"""
    Backward pass for a fully connected layer followed by a ReLU
    Inputs:
    - dout: upstream derivatives
    - cache: parameters calculated during the forward pass

    Returns:
    - dX: derivative w.r.t X
    - dW: derivative w.r.t W
    - db: derivative w.r.t b
    """
	#############################################################################
	# TODO: Implement the function.                                             #
	#############################################################################
	fc_cache, relu_cache = cache
	da = relu_backward(dout, relu_cache)
	dx, dw, db = fc_backward(da, fc_cache)
	return dx, dw, db
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################


def eval_numerical_gradient_array(f, x, df, h=1e-5):
	"""
	Evaluate a numeric gradient for a function that accepts a numpy
	array and returns a numpy array.
	"""
	grad = np.zeros_like(x)
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		ix = it.multi_index

		oldval = x[ix]
		x[ix] = oldval + h
		pos = f(x).copy()
		x[ix] = oldval - h
		neg = f(x).copy()
		x[ix] = oldval

		grad[ix] = np.sum((pos - neg) * df) / (2 * h)
		it.iternext()
	return grad
