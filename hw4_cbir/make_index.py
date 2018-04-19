import numpy as np
import pickle
import argparse
import os
import time
from sklearn.cluster import KMeans

from descriptor import ColorDescriptor
from lsh import LSHash
from common import read_test_set, read_train_set


def index_train_features(bin_counts):
    print("Initializing color descriptor...")
    cd = ColorDescriptor(bin_counts)
    pickle.dump(cd, open("%s/color_descriptor.p" % args.dir, "wb"))
    print("Color descriptor is saved successfully\n")

    y_test = read_test_set()
    y_train = read_train_set(y_test)

    print("Counting train features...")
    X_train = np.array(list(map(cd.describe, y_train)))
    train_set_index = list(zip(X_train, y_train))
    pickle.dump(train_set_index, open("%s/train_set_index.p" % args.dir, "wb"))
    print("Train features are saved successfully\n")

    return X_train, y_train


def index_kmean(X_train, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    print("Training K-means model...")
    kmeans.fit(X_train)
    pickle.dump(kmeans, open("%s/kmeans.p" % args.dir, 'wb'))
    print("K-means model is saved successfully\n")
    return kmeans


def index_lhs(X_train, y_train, kmeans, n_bits=8):
    train_set_index = list(zip(X_train, y_train))
    lshs = []
    for i in range(args.n_clusters):
        lshs.append(LSHash(n_bits, len(X_train[0])))

    print("Building LSHash...")
    for item, cluster in zip(train_set_index, kmeans.labels_):
        lshs[cluster].index(item[0], extra_data=item[1])

    pickle.dump(lshs, open("%s/lshs.p" % args.dir, 'wb'))
    print("LSHash is saved successfully\n")


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Get an index of train images and save it to the directory")
    parser.add_argument('--bins', nargs=3, type=int, metavar="H",
                        help="Number of bins for each of HSV diagrams (default [8, 12, 3])",
                        default=[8, 12, 3])
    parser.add_argument('-n', '--n_clusters', type=int, metavar="N",
                        help="Number of clusters for K-means (default 30)",
                        default=30)
    parser.add_argument('-b', '--n_bits', type=int, metavar="B",
                        help="Number of bits for LSH (default 8)",
                        default=8)
    parser.add_argument('-d', '--dir', type=str,
                        help="Directory to save index (default 'index')",
                        default="index")
    args = parser.parse_args()

    print(args.dir)
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    X_train, y_train = index_train_features(args.bins)

    kmeans = index_kmean(X_train, args.n_clusters)

    index_lhs(X_train, y_train, kmeans, args.n_bits)

    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    print("%.0f minutes %.0f seconds to build index" % (m, s))

