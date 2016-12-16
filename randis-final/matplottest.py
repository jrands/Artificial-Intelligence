import finalGetDigits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# get digits data X (training input) and y (target output)
X, y, X_te, y_te = finalGetDigits.getDataSet()

#penC  <- Penalty parameter C of the error term
#tubEpsilon  <- the epsilon-tube within which no penalty is associated

bestC=0
bestEpsilon=0
bestGamma=0
bestScore=float('-inf')
score=0

# here we create the final SVR
svr =  SVR(C=bestC, epsilon=bestEpsilon, gamma=bestGamma, kernel='rbf', verbose=True)
# here we train the final SVR
svr.fit(X, y)
# E_out in training
print("Training set score: %f" % svr.score(X, y)) 
# here test the final SVR and get E_out for testing set
ypred=svr.predict(X_te)
score=svr.score(X_te, y_te)
print("Testing set score: %f" % score)

x_min, x_max = np.min(X_te, axis=0), np.max(X_te, axis=0)
X_te = (X_te - x_min) / (x_max - x_min)

plt.figure(figsize=(6, 4))

plt.text(X_te[i, 0], X_te[0, 1], str(y_te[i]), color=plt.cm.spectral(round(ypred[i]) / 10.), fontdict={'weight': 'bold', 'size': 9})

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.tight_layout()

plt.show()
