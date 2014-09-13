
# Generate data for MLP training
# ~ Christopher Pramerdorfer

import numpy as np
import matplotlib.pyplot as plt
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

# generate data
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
x = np.matrix(x).T
y = np.matrix(y).T

# show data

plt.plot(x, y, '--g')
plt.plot(x, y, '.b')
plt.xlim([0, 2*np.pi])
plt.ylim([-1.1,1.1])
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.gcf().set_size_inches(5, 4)
plt.savefig('mlp-input.pdf', bbox_inches='tight', pad_inches=0.05)

plt.show()

# save in pylearn2 format
dm = DenseDesignMatrix(X = x, y = y)
serial.save('mlp_data_regression.pkl', dm)
