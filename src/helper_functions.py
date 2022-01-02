import itertools
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics import accuracy_score as acc_score
from sklearn import preprocessing 


def all_combos(a, b):
    res = []

    permutations = itertools.permutations(a, len(b))
    for permutation in permutations:
        zipped = zip(permutation, b)
        res.append(list(zipped))

    return res


def get_mse_combo(a, b):
    max_mse = 0
    max_mse_index = 0
    i = 0
    p = []

    combos = all_combos(a, b)
    for combo in combos:
        mse = 0
        for pair in combo:
            pair = np.array(pair)
            mse -= cs([pair[0]], [pair[1]])
            #mse += np.linalg.norm(pair[0]-pair[1])
        if mse < max_mse or not max_mse:
            max_mse = mse
            max_mse_index = i
            res = combo
        i += 1

    return res


def get_mse_index(a, b):
    pairs = get_mse_combo(a, b)
    res = []
    for pair in pairs:
        first = a.index(pair[0])
        second = b.index(pair[1])
        res.append((first, second))
    return res


def kmeans_pred(X, y, label_vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init=label_vectors).fit(X)

    if not (type(label_vectors) is np.ndarray):
        label_vectors = label_vectors.to_numpy()
    indexes = get_mse_index(kmeans.cluster_centers_.tolist(), label_vectors.tolist())

    reps = dict(indexes)
    kmeans_pred = [reps.get(x, x) for x in kmeans.labels_]
    print(acc_score(y, kmeans_pred))

    return kmeans.labels_

def cos_kmeans_pred(X, y, label_vectors, n_clusters):
    X_norm = preprocessing.normalize(X)
    kmeans = KMeans(n_clusters=n_clusters, init=label_vectors).fit(X_norm)

    if not (type(label_vectors) is np.ndarray):
        label_vectors = label_vectors.to_numpy()
    indexes = get_mse_index(kmeans.cluster_centers_.tolist(), label_vectors.tolist())

    reps = dict(indexes)
    kmeans_pred = [reps.get(x, x) for x in kmeans.labels_]
    print(acc_score(y, kmeans_pred))

    return kmeans.labels_

def get_single_sample_by_index(data, i):
    all_data = data.__iter__().get_next()
    samples = all_data[0].numpy()
    labels = all_data[1].numpy()

    label_names = ['World', 'Sports', 'Business', 'Science and Technology']
    return samples[i], labels[i], label_names[labels[i]]
