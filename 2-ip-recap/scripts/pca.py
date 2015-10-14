
# generate PCA-related figures for the slides

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.decomposition

np.random.seed(12)

# nice figures

s = 10
matplotlib.rc('font', family='sans', serif=['Computer Modern'], size=s)
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(4, 4))
matplotlib.rc('legend', fontsize=s, frameon=True, fancybox=True, loc='best')
matplotlib.rc('xtick', labelsize=s)
matplotlib.rc('ytick', labelsize=s)

def savefig(fig, filename, legend=None):
    l = matplotlib.pyplot.gca().get_legend()
    if l:
        l.get_frame().set_alpha(0.5)
    fig.savefig(filename.replace(' ', '_'), format='pdf', bbox_inches='tight')

# generate correlated data

n = 30
x = np.sort(np.random.uniform(-1, 1, size=n))
y = 2 * x + np.random.uniform(-0.5, 0.5, size=n)

X = np.vstack((x, y)).T
color = np.linspace(0, 1, x.size)

# show

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=color, marker='x', cmap=plt.get_cmap('cool'))
plt.axis('equal')

savefig(fig, 'pca-original.pdf')

# standardize

X = sklearn.preprocessing.scale(X)  # zero-mean and std of 1

# perform pca

pca = sklearn.decomposition.PCA()
pca.fit(X)

evecs = pca.components_  # eigenvectors
evals = pca.explained_variance_ratio_  # normalized eigenvalues

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=color, marker='x', cmap=plt.get_cmap('cool'))
ax = plt.axes()
ds = 2
ax.arrow(0, 0, evecs[0, 0], evecs[0, 1], head_width=0.1, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, evecs[1, 0]/3, evecs[1, 1]/3, head_width=0.1, head_length=0.1, fc='k', ec='k')
plt.axis('equal')

savefig(fig, 'pca-eigenvectors.pdf')

# rotate

Xr = np.dot(evecs, X.T).T
evecsr = np.dot(evecs, evecs.T)

fig = plt.figure()
plt.scatter(Xr[:, 0], Xr[:, 1], c=color, marker='x', cmap=plt.get_cmap('cool'))
ax = plt.axes()
ds = 2
ax.arrow(0, 0, evecsr[0, 0], evecsr[0, 1], head_width=0.1, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, evecsr[1, 0]/3, evecsr[1, 1]/3, head_width=0.1, head_length=0.1, fc='k', ec='k')
plt.axis('equal')

savefig(fig, 'pca-rotated.pdf')

# drop most uninformative axis

Xd = Xr[:, 0]

fig = plt.figure()
ax = plt.axes()
plt.scatter(Xd, np.zeros((Xd.size,)), c=color, marker='x', cmap=plt.get_cmap('cool'))
plt.axis('equal')

savefig(fig, 'pca-reduced.pdf')

# reconstruct

Xx = np.dot(evecs.T, np.vstack((Xd, np.zeros((Xd.size,))))).T

fig = plt.figure()
ax = plt.axes()
plt.scatter(Xx[:, 0], Xx[:, 1], c=color, marker='x', cmap=plt.get_cmap('cool'))
plt.axis('equal')

savefig(fig, 'pca-reconstructed.pdf')

plt.show()
