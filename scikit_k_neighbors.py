from random import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

k = 5
iris_data = load_iris()
# iris_data['data'], iris_data['target'] = shuffle(iris_data['data'], iris_data['target'])
data_train, data_test, label_train, label_test = train_test_split(iris_data['data'], iris_data['target'], test_size=0.33, random_state=42)

k_classifier = KNeighborsClassifier(k)
k_classifier.fit(data_train, label_train)
print(k_classifier.score(data_test, label_test))