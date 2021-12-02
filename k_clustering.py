from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


class kClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.centroid_points = None

    def get_centroids(self):
        return self.centroids

    def get_clusters(self):
        return self.centroid_points

    @staticmethod
    def normalize(data):
        data = data.T
        new_array = np.empty([len(data), len(data[0])], float)
        for d in range(len(data)):
            minimum = min(data[d])
            maximum = max(data[d])
            normalized = [(x - minimum) / (maximum - minimum) for x in data[d]]
            new_array[d] = normalized

        return new_array.T

    @staticmethod
    def distance(point1, point2):
        total = 0
        for p in range(len(point1)):
            total += (point1[p] - point2[p])**2
        return total**0.5

    @staticmethod
    def array_average(data):
        data = np.array(data).T
        new_array = []
        for d in range(len(data)):
            total = 0
            for x in range(len(data[d])):
                total += data[d][x]
            new_array.append(total / len(data[d]))
        return np.array(new_array).T

    def closest_centroid(self, point):
        centroid_distance = [self.distance(point, x) for x in self.centroids]
        return centroid_distance.index(min(centroid_distance))

    def assign_points(self, data):
        self.centroid_points = [[] for _ in self.centroids]
        # assign points to centroids
        for d in range(len(data)):
            close_centroid_index = self.closest_centroid(data[d])
            self.centroid_points[close_centroid_index].append(data[d])

    def train(self, data):
        self.centroids = np.random.rand(self.k, len(data[0]))
        prev_centroids = np.empty([self.k, len(data[0])], float)
        changing = True

        self.assign_points(data)

        while changing:
            centroid_changes = 0
            print("changed")
            for c in range(len(self.centroids)):
                if len(self.centroid_points[c]) != 0:
                    self.centroids[c] = self.array_average(self.centroid_points[c])
                centroid_changes += self.distance(self.centroids[c], prev_centroids[c])
                prev_centroids[c] = self.centroids[c]

            self.assign_points(data)

            if centroid_changes == 0:
                changing = False


blob_dataset = make_blobs(n_samples=1000, n_features=2, centers=3)
# blob_dataset = make_circles(n_samples=1000)
# blob_dataset = make_moons(n_samples=1000)
point_data = np.array(blob_dataset[0])

classifier = kClustering(3)
point_data = classifier.normalize(point_data)

classifier.train(point_data)

centroid_clusters = classifier.get_clusters()
centroids = classifier.get_centroids()

for i in centroid_clusters:
    x_data = [x[0] for x in i]
    y_data = [x[1] for x in i]
    plt.scatter(x_data, y_data)

for i in centroids:
    x_data = [i[0] for x in centroids]
    y_data = [i[1] for x in centroids]
    plt.scatter(x_data, y_data, marker="x", color='black', s=80)


plt.show()
