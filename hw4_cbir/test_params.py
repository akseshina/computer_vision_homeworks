
import numpy as np
import pickle
import argparse

from common import read_test_set
from cbir import APk
import time

start_time = time.time()
parser = argparse.ArgumentParser(description="Calculate MAPk and time for the whole test set")
parser.add_argument('-i', '--ind_dir', type=str,
                    help="Index directory (default 'index')",
                    default="index")
args = parser.parse_args()

index_start_time = time.time()
print("Index loading...")
cd = pickle.load(open("%s/color_descriptor.p" % args.ind_dir, "rb"))
train_set_index = pickle.load(open("%s/train_set_index.p" % args.ind_dir, "rb"))
kmeans = pickle.load(open("%s/kmeans.p" % args.ind_dir, "rb"))
lshs = pickle.load(open("%s/lshs.p" % args.ind_dir, "rb"))
elapsed_time = time.time() - index_start_time
print("Time for index loading:", elapsed_time)
print()

# Считываем тестовый датасет
y_test = read_test_set()
X_test = list(map(cd.describe, y_test))

# Ищем все картинки из теста
clusters_predicted = kmeans.predict(X_test)
APks = []
for i in range(len(X_test)):
    cluster = clusters_predicted[i]
    query = X_test[i]
    results = lshs[cluster].query(query)
    query_img_path = y_test[i]
    APks.append(APk(query_img_path, results))

# Итоговый MAPk
MAPk = np.mean(APks)
print("MAPk:", MAPk)

# Время
elapsed_time = time.time() - start_time
print("Time:", elapsed_time)
