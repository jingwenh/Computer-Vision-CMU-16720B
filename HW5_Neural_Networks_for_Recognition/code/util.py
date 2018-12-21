import numpy as np

# use for a "no activation" layer
def linear(x):
    return x

def linear_deriv(post_act):
    return np.ones_like(post_act)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(post_act):
    return 1-post_act**2

def relu(x):
    return np.maximum(x,0)

def relu_deriv(x):
    return (x > 0).astype(np.float)