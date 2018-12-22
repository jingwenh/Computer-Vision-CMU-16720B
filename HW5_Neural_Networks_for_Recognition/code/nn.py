import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    high = np.sqrt(6/(in_size+out_size))
    low = -high

    W = np.random.uniform(low, high, (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    max_xi = np.max(x, axis=1)
    shift_x = x - np.expand_dims(max_xi, axis=1)
    res = np.exp(shift_x)
    res = res/np.expand_dims(np.sum(res, axis=1), axis=1)
    
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y*np.log(probs))

    y = y.astype(int)
    pred_y = (probs == np.expand_dims(np.max(probs, axis=1), axis=1))
    with_same_prob = np.where(np.sum(pred_y, axis=1) > 1)[0]
    for i in range(with_same_prob.shape[0]):
        pred_y[i, np.where(pred_y[i, :]==np.max(pred_y[i, :]))[0][0]+1:] = False
    error = np.sum(np.abs(pred_y-y))//2
    acc = (y.shape[0]-error)/y.shape[0]
    
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    grad_W, grad_X, grad_b = np.zeros(W.shape), np.zeros(X.shape), np.zeros(b.shape)
    res = delta * activation_deriv(post_act)

    for i in range(X.shape[0]):
        grad_W += np.dot(np.expand_dims(X[i, :], axis=1), np.expand_dims(res[i, :], axis=0))
        grad_b += res[i, :]
        grad_X[i, :] = np.dot(W, np.expand_dims(res[i, :], axis=1)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    inds = range(x.shape[0])
    while len(inds) > 0:
        rand_inds = np.random.randint(0, len(inds), batch_size)
        selected = [inds[i] for i in rand_inds]
        batch_x = [x[i] for i in selected]
        batch_y = [y[i] for i in selected]
        batches.append((np.array(batch_x), np.array(batch_y)))
        inds = list(set(inds) - set(selected))
    
    return batches
