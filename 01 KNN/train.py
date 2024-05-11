import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
## courbe roc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target
#This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
# sepal length in cm
# sepal width in cm
# petal length in cm
# petal width in cm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions,"prediction")

acc = np.sum(predictions == y_test) / len(y_test)
print(acc,'acc')


error=np.mean(predictions!=y_test)
print(error,'erro')


# ## courbe roc 

# d=np.array(predictions == y_test)
# print(d)
# fpr, tpr, threshold = roc_curve(y_test, )
# print(threshold)
# roc_auc = auc(fpr, tpr)
# print(roc_auc)