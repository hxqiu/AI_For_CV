import random
import numpy as np

data = np.random.rand(100, 2)

def get_dis(data1, data2):
    return np.sqrt(np.sum(np.square(data1 - data2)))


def kmeans(k, data_list):
    # 1 get k random centers
    centroids = []
    idx_list = random.sample(range(0, len(data_list)), k) # print(range(1,n)) [1, 2, 3, ... n-1]
    for idx in idx_list:
        centroids.append(data_list[idx])

    is_coverage = True
    while is_coverage:
        # 2 for each data ,calculate which class it should be in
        classes = [[] for i in range(k)]
        for i in range(len(data_list)):
            data_class = 0
            min_dis = 10000.0
            for j in range(len(centroids)):
                dis = get_dis(data_list[i], centroids[j])
                if(dis < min_dis):
                    min_dis = dis
                    data_class = j
            classes[data_class].append(data_list[i])
        # 3 recal center of every k classes
        new_centroids = []
        update_vol = 0.
        for i in range(k):
            new_centroids.append(np.sum(classes[i], 0)/ len(classes[i]))
            update_vol += np.sum(np.abs(new_centroids[i] - centroids[i]))
        centroids = new_centroids
        # 4 repeat 2~3 until coverage to required range
        print('total:{:.8f}'.format(update_vol))
        if update_vol < 0.01:
            is_coverage = False

kmeans(10, data)
