import csv
import numpy as np
import pickle

with open('winequality-red.csv', newline='') as csvfile:
    reader  = csv.reader(csvfile, delimiter=';', quoting = csv.QUOTE_NONNUMERIC)

    output = []
    for i, row in enumerate(reader):
        if i == 0:
            K = row
        else:
            output.append(row[:])

output = np.array(output)

data = dict()
data['features'] = output[:,:-1]
data['quality'] = output[:,-1]
data['feature_names'] = K[:-1]

with open('../wine_red_dataset.pkl', 'wb') as H:
    pickle.dump(data, H)


