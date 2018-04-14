
# coding: utf-8

# ### Поиск изображений по содержанию (30 баллов)
# 
# Вам предстоит построить систему, которая позволит выполнить поиск изображений по содержанию в базе Corel-10k, которая приложена к домашнему заданию. База состоит из 100 классов изображений по 100 изображений в каждом классе. Изображения из одного класса лежат подряд в промежутке 1-100, 101-200, и тд. В файле **test.dat** лежит список картинок, которые не должны участвовать в построении поисковой базы, но по которым надо будет сделать 1000 поисковых запросов. Ограничение по памяти -- 2Гб.
# 
# 1. По изображениям не попавшим в **test.dat** посчитайте дескриптор с помощью алгоритма GIST.
#    
# 2. Разбейте дескрипторы всех изображений на 100 кластеров с помощью K-Means.
# 
# 3. Для каждого кластера постройте функцию хэширования (LSH), с помощью которой закодируйте каждое изображение в тренировочной выборке.
# 
# 4. Релизуйте функцию `retrieve` и сделайте 1000 запросов изображениями из файла **test.dat**.
# 
# 5. Продемонстрируйте работу `retrieve` на 5-ти случайных изображениях из **test.dat**. Замерьте время исполнения поискового запроса. Сделайте возможность вызвать `retrieve` без построения индекса, то есть приложите в решение индексированную базу, если только время построения индекса не укладывается в 1 минуту.  
#     
# 6. Для каждого запроса нужно оценить APk, где k=10 и посчитать среднее значение этой величины по всем запросам (MAPk).
# 
# Так же за это задание можно получить еще до 20-ти дополнительных баллов.
# Качество поиска вы посчитаете сами, а вот со скоростью есть некоторые трудности,
#
# реализуйте ваше решение таким образом,
# чтобы я мог из командной строки вызвать:
# 
# `python cbir.py --retrieve /path/to/image`
# 
# 1. Вы можете сделать быстрый поиск. Если качество вашего поиска окажется выше, чем медиана качества по всем поисковым движкам, то вы принимаете участие в борьбе за 10 призовых баллов.
# 
# 2. Вы можете сделать хороший поиск. Если ваш поиск окажется быстрее, чем медиана среднего времени работы других участников, то вы принимаете участие в борьбе за точный поиск.
# 
# Для достижения лучших результатов вам придется регулировать несколько параметров алгоритма:
# - Дескриптор. Можете взять что угодно вместо GIST
# - Количество кластеров k-means
# - Длину кода LSH
# - etc 
# 
# Помните о том, что время поиска очень важный параметр.

import numpy as np
import pickle
import argparse

from common import read_test_set, read_train_set


def class_by_path(img_path):
    img_name = int(img_path[6:-4])
    if img_name == 1:
        return 0
    img_class = (img_name - 1) // 100
    return img_class


def APk(query_img_path, results):
    query_class = class_by_path(query_img_path)
    results_classes = np.array(list(map(class_by_path, results))) 
    return np.mean(np.cumsum(results_classes == query_class) / np.arange(1, 11))


def retrieve(img_path):
    print("Searching for", img_path)
    query = cd.describe(img_path)
    cluster = kmeans.predict([query])[0]
    results = lshs[cluster].query(query)
    print("\nResults:")
    for r in results:
        print(r)
    print("\nAPk: %0.2f" % APk(img_path, results))
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Content-based (actually, color-based) simple image retrieval")
    parser.add_argument('-r', '--retrieve', type=str, required=True,
                        help="Path to image")
    parser.add_argument('-i', '--ind_dir', type=str,
                        help="Index directory (default 'index')",
                        default="index")
    args = parser.parse_args()

    print("Index loading...")
    cd = pickle.load(open("%s/color_descriptor.p" % args.ind_dir, "rb"))
    train_set_index = pickle.load(open("%s/train_set_index.p" % args.ind_dir, "rb"))
    kmeans = pickle.load(open("%s/kmeans.p" % args.ind_dir, "rb"))
    lshs = pickle.load(open("%s/lshs.p" % args.ind_dir, "rb"))
    print("Index loaded!\n")

    y_test = read_test_set()

    retrieve(args.retrieve)
