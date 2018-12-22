import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')
keys = [key for key in params.keys()]
for k in keys:
    params['m_'+k] = np.zeros(params[k].shape)

# should look like your previous training loops
train_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        # loss
        total_loss += np.sum((xb - out)**2)

        # backward
        delta1 = -2*(xb-out)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hidden2', relu_deriv)
        delta4 = backwards(delta3, params, 'hidden', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)

        # update weights
        for k in params.keys():
            if '_' in k:
                continue
            params['m_'+k] = 0.9*params['m_'+k] - learning_rate*params['grad_'+k]
            params[k] += params['m_'+k]

    train_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot the training loss
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(max_iters), train_loss, color='g')
plt.show()
# save parameters
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# import pickle
# params = pickle.load(open('q5_weights.pickle', 'rb'))

# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
selected_inds = [4, 50, 500, 520, 1740, 1720, 2000, 2010, 3400, 3420]
for i in selected_inds:
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)
psnr_noisy = 0
for i in range(out.shape[0]):
    psnr_noisy += psnr(valid_x[i], out[i])

psnr_noisy /= out.shape[0]
print(psnr_noisy)
