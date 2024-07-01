'''in this exersize we first read the data from the pkl files, then organized the train and the test,
use the scaler tranform to normalized the data, import the necessary classifiers, and after that fit the
scaler data. now we get the success rate of each classifier, and we got the best result in RBF SVM.
the worst result is the Nearest Neighbors, but all the classifiers showed a great result with approximate 97-98%.
now, we take a jump of 500, and draw a plot that showed us how much the amount of the data in the train
affect the success rate. we saw that the bigger the data, the bigger the success rate, as predicted.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from matplotlib.ticker import FormatStrFormatter
# Load and process data
data_path_1 = r'C:\Users\binya\PycharmProjects\oron_bina/Ex_3_bina/occupancy_train.pkl'
with open(data_path_1, 'rb') as file:
    data_train = pickle.load(file)

data_path_2 = r'C:\Users\binya\PycharmProjects\oron_bina/Ex_3_bina/occupancy_test.pkl'
with open(data_path_2, 'rb') as file:
    data_test = pickle.load(file)

X_train, y_train = [], np.array(data_train['label'])
for x in data_train['features']:
    x = x.reshape((-1,))
    X_train.append(x)
X_train = np.array(X_train)

X_test, y_test = [], np.array(data_test['label'])
for x in data_test['features']:
    x = x.reshape((-1,))
    X_test.append(x)
X_test = np.array(X_test)

# Standardization with mean and std
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

names = ['Logistic Regression',
         'Nearest Neighbors',
         'Linear SVM',
         'RBF SVM',
         'Gaussian Naive-Bayes']

classifiers = [
    LogisticRegression(solver='liblinear', random_state=0),
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=.3, C=1, probability=True),
    GaussianNB()]

jump = 500
num_samples = np.arange(jump, len(X_train) + jump, jump)
if num_samples[-1] != len(X_train):
    num_samples = np.append(num_samples, len(X_train))
scores = [[] for _ in range(len(names))]

for num in num_samples:
    X_train_subset = X_train[:num]
    y_train_subset = y_train[:num]

    for i, clf in enumerate(classifiers):
        clf.fit(X_train_subset, y_train_subset)
        score = clf.score(X_test, y_test) * 100
        scores[i].append(score)

        if num == num_samples[-1]:
            print(f'{names[i]}: {score}%')



# Plotting
for i, name in enumerate(names):
    plt.plot(num_samples, scores[i], marker='o', label=name)

plt.xlabel('Number of Samples')
plt.ylabel('Score (%)')
plt.title('Comparison of Classifier Scores')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend()
plt.show()