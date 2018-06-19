import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#DESCR键对应的是数据集的简要说明
print(iris_dataset['DESCR'][:193]+'\n...')
#target_names键对应的值是一个字符串数组，其中包含要预测的花的品种
print("Target_names: {}".format(iris_dataset['target_names']))
#feature_names 键对应的值是一个字符串列表，对每一个特征进行了说明
print("Feature names: \n{}".format(iris_dataset['feature_names']))
#数据包含在target和data字段中。data里面是花萼长度、花萼宽度、花瓣长度、花瓣宽度的测量数据，格式为Numpy数组：
print("Type of data: {}".format(type(iris_dataset['data'])))
#data数组的每一行对应一朵花，列代表每朵花的四个测量数据
print("Shape of data: {}\n".format(iris_dataset['data'].shape))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

#利用X_train中的数据创建DataFrame
#利用iris_dataset.feature_name中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#利用DataFrame创建散点图矩阵，按照y_train着色
#grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
#                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#plt.show()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#2.3.1
#生成数据集
X, y = mglearn.datasets.make_forge()
#数据集绘图
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("second feature")
print("X.shape: {}".format(X.shape))
plt.show()
