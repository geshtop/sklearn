# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

from sklearn import tree

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

# import some data to play with


iris = datasets.load_iris()
rf_clf = RandomForestClassifier()


preds = cross_val_predict(estimator=rf_clf, X=iris["data"], y=iris["target"], cv=15)

print(classification_report(iris["target"], preds))

#mylist = []
#do loop

clf = tree.DecisionTreeClassifier()

clf.max_depth = 10

clf.criterion = 'entropy'

clf = clf.fit(iris.data, iris.target)

print("Decision Tree: ")

accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10)

print("Average Accuracy of  DT with depth ", clf.max_depth, " is: ", round(accuracy.mean(),3))

#mylist.append(accuracy.mean())  loop, can be used to plot later…

precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted', cv=10)

print("Average precision_weighted of  DT with depth ", clf.max_depth, " is: ", round(precision.mean(),3))


#y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
#y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
#my_f_macro = f1_score(y_true, y_pred, average='macro')

#my_f_micro = f1_score(y_true, y_pred, average='micro')

#print('my f macro {}'.format(my_f_macro))

#print('my f micro {}'.format(my_f_micro))



#X = range(10)
#plt.plot(X, [x * x for x in X])
#plt.xlabel("This is the X axis")
#plt.ylabel("This is the Y axis")
#plt.show()

