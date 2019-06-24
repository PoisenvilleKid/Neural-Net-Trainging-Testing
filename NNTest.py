import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RepeatedKFold 
from sklearn.model_selection import cross_val_score


def printPerdiction(prediction,X,y):
    print(cross_val_score(prediction, X, y, cv=10))
    print('\n')  

#Load Dataset
dataset = pd.read_csv('wdbc.csv',sep=',')

#Format Diagnosis to 1 or 0
dataset.drop(columns=['ID'], inplace=True)
np_dataset = dataset['Diagnosis'].values
np_dataset = [({'M': 0, 'B': 1})[d] for d in np_dataset]
dataset['Diagnosis'] = np_dataset

X = dataset.drop('Diagnosis', axis = 1)
Y = dataset['Diagnosis']

#Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

#Scale & Divide Test and Train Sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Fitting SVM Linear Model
print("Printing Score of SVM Linear")
svml = svm.SVC(kernel = 'linear')
svml.fit(X_train,y_train)
pred_svml = svml.predict(X_test)
printPerdiction(svml,X,Y)

#Fitting SVM Auto Model
print("Printing Score of SVM Auto")
svmp = svm.SVC(gamma='auto')
svmp.fit(X_train,y_train)
pred_svmp = svmp.predict(X_test)
printPerdiction(svmp,X,Y)

#Neural Network With tanh
print("Printing Score of NN tanh")
mlpc = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1200, activation='tanh')
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
printPerdiction(mlpc,X,Y)

#Neural Network With Softplus
print("Printing Score of NN relu")
mlpcsp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1200, activation='relu')
mlpcsp.fit(X_train, y_train)
pred_mlpcsp= mlpcsp.predict(X_test)
printPerdiction(mlpcsp,X,Y)
