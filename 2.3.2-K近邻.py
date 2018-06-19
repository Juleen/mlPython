import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
##1-K近邻分类
##mglearn.plots.plot_knn_classification(n_neighbors=1)
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
##导入类并将其实例化,可以设定参数，如邻居个数
clf = KNeighborsClassifier(n_neighbors=3)
##利用训练集对分类器进行拟合，即保存数据集，以便预测时计算与邻居间的距离
clf.fit(X_train, y_train)
##调用predict方法对测试数据进行预测。
print("Test set predictions: {}".format(clf.predict(X_test)))
##评估模型的泛化能力，对测试数据及测试标签调用score方法
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

##2-分析kNeighborsClassifier
##将1、3、9个邻居的三种情况的决策边界可视化
#fig, axes = plt.subplots(1, 3, figsize=(10, 3))

#for n_neighbors, ax in zip([1, 3, 9], axes):
#    ##fit方法返回对象本身，所以将实例化与拟合放在一行中
#    clf1 = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#    mglearn.plots.plot_2d_separator(clf1, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{} neighbor(s)".format(n_neighbors))
#    ax.set_xlabel("feature 0")
#    ax.set_ylabel("feature 1")
#axes[0].legend(loc=3)


##在乳腺癌数据集上验证模型复杂度与泛化能力间的关系
##将数据集分成训练集和测试集，然后使用不同的邻居个数对训练集和测试集的性能进行评估
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
#n_neighbors取值范围[1,10]间的整数
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    #构建模型
    clf_cancer = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf_cancer.fit(X_train, y_train)
    #记录训练集精度
    training_accuracy.append(clf_cancer.score(X_train, y_train))
    #记录泛化精度
    test_accuracy.append(clf_cancer.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#plt.show()

##3-K近邻回归
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
#将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#模型实例化，并将邻居个数设置为3
reg = KNeighborsRegressor(n_neighbors=3)
#利用训练集数据和训练目标值来拟合模型
reg.fit(X_train, y_train)
#对测试集进行预测
print("test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
