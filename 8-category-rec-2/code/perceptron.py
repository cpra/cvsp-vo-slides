
# Perceptron example
# ~ Christopher Pramerdorfer

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# xor problem ... ouch!
x = np.array([[-1,-1,+1,+1],[-1,+1,-1,+1]], dtype=float).T
y = np.array([-1,+1,+1,-1], dtype=int)

# greate two-cluster data
x, y = datasets.make_classification(n_features=2, class_sep=1, n_redundant=0, n_informative=2, n_clusters_per_class=1)
y[y==0] = -1

# fit perceptron
perc = linear_model.Perceptron(n_iter=500)
perc.fit(x, y)

# print accuracy on training data (<1 for non-separable data)
score = perc.score(x, y)
print('Accuracy on training data: {}'.format(score))

# params as in slides
w = perc.coef_[0]
b = perc.intercept_
print('Fitted params: w={}, b={}'.format(w, b))

# draw the decision boundary
if w[1] != 0:
    xl = np.linspace(np.min(x[:,0]), np.max(x[:,0]))
    yl = -(w[0]*xl-b)/w[1]
    plt.plot(xl, yl)

# draw the training data
plt.scatter(x[y==-1,0], x[y==-1,1], c='r', s=50, marker='o')
plt.scatter(x[y==+1,0], x[y==+1,1], c='b', s=50, marker='^')

# annotate plot
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([np.min(x[:,0]), np.max(x[:,0])])
plt.ylim([np.min(x[:,1]), np.max(x[:,1])])

# save to disk and show
plt.savefig('perceptron-result.pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()
