from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
iris=load_iris()#sklearn自带的iris数据集
x=iris.data
y=iris.target
test_size=0.20
seed=1#随机数的编号，seed=1，每次都一样。seed为0或不赋值，每次划分的都不一样
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=test_size,random_state=seed)#划分数据集，train_test_split随机划分
knn=KNeighborsClassifier()#直接调用KNN
knn.fit(xtrain,ytrain)
predictions=knn.predict(xtest)
print("accuracy_score",accuracy_score(ytest,predictions))
print("confusion_matrix",confusion_matrix(ytest,predictions))
print("classification_report",classification_report(ytest,predictions))

print(test_size)