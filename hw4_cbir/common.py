import pandas as pd
import glob


def read_test_set():
    test_set = pd.read_csv("test.dat", header=None, names=["file"])
    test_set["file"] = test_set["file"].astype(str).str[:-4]
    test_set = test_set["file"]
    test_set = "Corel/" + test_set + "jpg"
    return test_set


def read_train_set(test_set):
    train_set = glob.glob('Corel/*')
    train_set = pd.Series(train_set)
    train_set = train_set[~train_set.isin(test_set)]
    return train_set
