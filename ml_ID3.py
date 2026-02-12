import math
import random
import urllib.request
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = []
for line in urllib.request.urlopen(url):
    line = line.decode("utf-8").strip()
    if line:
        row = line.split(",")
        features = list(map(float, row[:4]))
        label = row[4]
        data.append(features + [label])
for row in data:
    for i in range(4):
        if row[i] < 3:
            row[i] = "Low"
        elif row[i] < 6:
            row[i] = "Medium"
        else:
            row[i] = "High"
random.shuffle(data)
split = int(0.7 * len(data))
train = data[:split]
test = data[split:]
def entropy(dataset):
    labels = [row[-1] for row in dataset]
    total = len(labels)
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent
def info_gain(dataset, feature):
    total_entropy = entropy(dataset)
    total = len(dataset)
    values = {}
    for row in dataset:
        values.setdefault(row[feature], []).append(row)
    weighted = 0
    for subset in values.values():
        weighted += (len(subset)/total) * entropy(subset)
    return total_entropy - weighted
def build_tree(dataset, features):
    labels = [row[-1] for row in dataset]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if not features:
        return max(set(labels), key=labels.count)
    best = max(features, key=lambda f: info_gain(dataset, f))
    tree = {best: {}}
    values = set(row[best] for row in dataset)
    for v in values:
        subset = [row for row in dataset if row[best] == v]
        remaining = [f for f in features if f != best]
        tree[best][v] = build_tree(subset, remaining)
    return tree
def predict(tree, row):
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]
    value = row[feature]

    if value in tree[feature]:
        return predict(tree[feature][value], row)
    else:
        return "Iris-setosa"
features = [0,1,2,3]
tree = build_tree(train, features)
correct = 0
for row in test:
    if predict(tree, row) == row[-1]:
        correct += 1

accuracy = correct / len(test)

print("Decision Tree:", tree)
print("Accuracy:", accuracy)
