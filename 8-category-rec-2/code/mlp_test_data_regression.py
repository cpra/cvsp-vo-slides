
# Perform regression using learned pylearn2 MLP
# ~ Christopher Pramerdorfer

import numpy as np
import matplotlib.pyplot as plt
from pylearn2.utils import serial
from theano import tensor, function

# load the model
model = serial.load('mlp_regression.pkl')

# init predictor
X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)
f = function([X], Y)

# generate some test data and apply the model
x = np.linspace(0, 2*np.pi)
x = np.matrix(x).T
y = f(x)

# show and save result

plt.plot(x, np.sin(x), '--g')
plt.plot(x, y, '-b')
plt.xlim([0, 2*np.pi])
plt.ylim([-1.1,1.1])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(('training', 'test'))

plt.gcf().set_size_inches(5, 4)
plt.savefig('mlp-result.pdf', bbox_inches='tight', pad_inches=0.05)

plt.show()
