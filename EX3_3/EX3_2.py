'''This exersize was really similar to the previous one, but the difference here that we used only with 3
classifiers, and the user number has a direct impact of the success rate.
we draw a plot that showed us how the number of the users affect the data accuracy, and because of the
every user took a different nuber of samples, we got sometimes overfit to the data, and not a sharp rise in every
grow in the users number.
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
data_path_1 = r'C:\Users\binya\PycharmProjects\oron_bina/Ex_3_bina/emg_train.pkl'
with open(data_path_1, 'rb') as file:
    data_train = pickle.load(file)

data_path_2 = r'C:\Users\binya\PycharmProjects\oron_bina/Ex_3_bina/emg_test.pkl'
with open(data_path_2, 'rb') as file:
    data_test = pickle.load(file)

# print(data_train)
# print(data_train['users_train'])

X_train, y_train = [], np.array(data_train['label'])
for x in data_train['features']:
    x = x.reshape((-1,))
    X_train.append(x)
X_train = np.array(X_train)
# print(X_train[0])

X_test, y_test = [], np.array(data_test['label'])
for x in data_test['features']:
    x = x.reshape((-1,))
    X_test.append(x)
X_test = np.array(X_test)
# # print(X_test)
# print(len(y_test))
# Standardization with mean and std
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#
names = ['Logistic Regression',
         'Nearest Neighbors',
         'Gaussian Naive-Bayes']
# #
classifiers = [
    LogisticRegression(solver='liblinear', random_state=0),
    KNeighborsClassifier(5),
    # SVC(kernel="linear", C=0.025, probability=True),
    # SVC(gamma=.3, C=1, probability=True),
    GaussianNB()]
# #
C = []
success_rates = []
# print()
for name, clf in zip(names, classifiers):
    clf.fit(list(X_train), list(y_train))
    score = clf.score(X_test, y_test)
    success_rates.append(score)# Evaluate on test data
    C.append(clf)
    print(name, score)
C = dict(zip(names, C))

counter = 0
num_dic = {}
num_users = 32
# data_train_user = dict(data_train['users_train'])
# print(data_train_user)
for idx ,item in enumerate(data_train['users_train']):
    indexes = num_dic.setdefault(item, [])
    indexes.append(idx)

X_train_new = []
y_train_new = []

# print(sorted(num_dic.keys()))
scores = [[] for _ in range(len(names))]
for key in sorted(num_dic.keys()):
    idx = num_dic[key]
    for i in idx:
        X_train_new.append(X_train[i])
        y_train_new.append(y_train[i])

    for j, clf in enumerate(classifiers):
        clf.fit(X_train_new, y_train_new)
        score = clf.score(X_test, y_test) * 100
        scores[j].append(score)


plt.figure()
for j, name in enumerate(names):
    plt.plot(range(1, num_users + 1), scores[j], label=name)
plt.xlabel('Number of Users')
plt.ylabel('Success Rate (%)')
plt.title('Classifier Performance')
plt.legend()
plt.show()



