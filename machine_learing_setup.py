from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
iris = datasets.load_iris()
#print(iris)
# print(iris.target)
# print(iris.feature_names)
model = KNeighborsClassifier()
print(model.fit(iris.data, iris.target))
pre = model.predict((iris.data))
print(pre)
mat_pre_matri = metrics.confusion_matrix(iris.target, pre)
print(mat_pre_matri)
print(metrics.classification_report(iris.target, pre))
