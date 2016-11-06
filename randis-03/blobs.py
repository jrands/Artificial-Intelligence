import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#generates random centers
ctrs = 3 * np.random.normal(0, 1, (2, 2))
#generates random data following normal distributions
X, y = make_blobs(n_samples=100, centers=ctrs, n_features=2,
	cluster_std=1.0, shuffle=False, random_state=0)
y[y==0] = -1 #makes sure we have +1/-1 labels
#plots data
c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')
#displays legend
plt.legend((c0, c1), ('Class_-1', 'Class_+1'), loc='upper_right',
	scatterpoints=1, fontsize=11)
#displays axis legends and title
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Two_simple_clusters_of_random_data')
#saves the figure into a .pdf file (desired!)
plt.savefig('hw3.plot.pdf', bbox_inches='tight')
plt.show()
