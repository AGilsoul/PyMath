# K-Nearest Neighbors Lab
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class kClassifier:
    # constructor
    def __init__(self, k=1):
        if k < 1:
            k = 1
        self.k = k
        self.train_data = None
        self.train_labels = None

    # setter and getter for k
    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k

    # min-max normalization
    @staticmethod
    def data_normalize(cur_data):
        # transpose the array
        temp_list = cur_data.T
        new_list = []
        for i in range(len(temp_list)):
            minimum = min(temp_list[i])
            maximum = max(temp_list[i])
            normalized = [(x - minimum) / (maximum - minimum) for x in temp_list[i]]
            new_list.append(normalized)

        return np.array(new_list).T

    # distance between points
    @staticmethod
    def distance(point1, point2):
        total = 0
        for i in range(len(point1)):
            total += (point1[i] - point2[i])**2

        return total**0.5

    # assigns training data
    def train(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    # predicts class of a point
    def predict(self, point):
        point_distances = [(kClassifier.distance(point, self.train_data[x]), self.train_labels[x]) for x in range(len(self.train_data))]
        point_distances.sort(key=lambda y: y[0])
        point_distances = point_distances[:self.k]
        unique_labels = []
        label_counts = []
        for x in range(len(point_distances)):
            if point_distances[x][1] not in unique_labels:
                unique_labels.append(point_distances[x][1])
                label_counts.append(1)
            else:
                label_counts[unique_labels.index(point_distances[x][1])] += 1

        return unique_labels[label_counts.index(max(label_counts))]

    # tests accuracy of model
    def test(self, validation_data, validation_labels):
        correct_count = 0
        for i in range(len(validation_data)):
            label = self.predict(validation_data[i])
            if label == validation_labels[i]:
                correct_count += 1

        return correct_count / len(validation_data)


# load iris data
iris_data = load_iris()
iris_data['data'] = np.array(iris_data['data'])
iris_data['target'] = np.array(iris_data['target'])

# create classifier
classifier = kClassifier()

# normalize iris data
iris_data['data'] = classifier.data_normalize(iris_data['data'])
# shuffle iris data with labels
iris_data['data'], iris_data['target'] = shuffle(iris_data['data'], iris_data['target'])

# numpy array with values [1, 150]
k_values = np.arange(start=1, stop=151)
# empty numpy array for results
results = np.empty(150)
# for all k within 1 and 150
for i in range(1, 151):
    classifier.set_k(i)
    # splits data
    data_train, data_test, label_train, label_test = train_test_split(iris_data['data'], iris_data['target'], test_size=0.33, random_state=42)
    classifier.train(data_train, label_train)
    # gets prediction results
    results[i - 1] = classifier.test(data_test, label_test)

# plots results for k
plt.plot(k_values, results)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
