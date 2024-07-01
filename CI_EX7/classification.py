import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB  

###### Load and process data ######
data = load_digits()

# Show example of 10 images from the dataset
plt.figure(figsize=(10,5))
plt.gray()
I = data.images[0]
for i in range(1,9):
    I = np.concatenate((I, data.images[i]), axis = 1)
plt.imshow(I)
plt.show()

# Flatten the 8x8 image to a vector of lenght 64
X, y = [], np.array(data.target)
for x in data.images:
    x = x.reshape((-1,))
    X.append(x)
X = np.array(X)

# print('y is', y)
##########################################

# Two options to normalize data
if 1:
    # Standardization with mean and std
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X) # X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)
else:
    # Normalize with min and max
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

names = ['Nearest Neighbors', 
         'Linear SVM', 
         'RBF SVM', 
         'Gaussian Naive-Bayes', 
         'Logistic Regression', 
         'Linear Regression']

# Define the classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability = True),
    SVC(gamma=.2, C=1, probability = True), # RBF is the default
    GaussianNB(),
    LogisticRegression(solver='liblinear', random_state=0),
    LinearRegression()]

# Iterate over classifiers and evaluate score over all test data
C = []
print()
for name, clf in zip(names, classifiers):
    clf.fit(list(X_train), list(y_train))
    
    score = clf.score(X_test, y_test) # Evaluate on test data
    C.append(clf)
    print (name, score)
C = dict( zip( names, C) )

# Iterate over classifiers and predict class for a single image
i = np.random.randint(len(y_test))
for name in C.keys():
    print ("-------------------------")
    print ('** Classifier ' + name + ': ')
    clf = C[name]

    x, y_real = X_test[i], y_test[i]

    y_predict = clf.predict(x.reshape(1,-1))[0]

    print ('Real class: ' + str(y_real) + ', predicted class: ' + str(y_predict))
    if name != 'Linear Regression': 
        dist = clf.predict_proba(x.reshape(1,-1))[0] # Get probability vector for each class (not possible for linear regression)
        dist = [np.round(d, 3) for d in dist]
        print ('Distribution: ')
        [print(i, d) for i, d in enumerate(dist)]

x = scaler.inverse_transform(X_test[i]).reshape(8,8)
plt.figure(figsize=(10,5))
plt.gray()
plt.imshow(x)
plt.show()
    