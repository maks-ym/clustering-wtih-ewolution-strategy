# clustering functionality

from numpy.linalg import norm
import numpy as np


class Distance:
    @staticmethod
    def manhattan(a, b):
        a = np.array(a)
        b = np.array(b)
        return norm(a-b, ord=1)

    @staticmethod
    def euclidean(a, b):
        a = np.array(a)
        b = np.array(b)
        return norm(a-b, ord=2)

#TODO: tests
class Centroids:
    @staticmethod
    def cluster(data, centers, dist_func="euclid"):
        ''' dist_func=["euclid","manhat"] '''
        dist_func = Distance.manhattan if dist_func == "manhat" else Distance.euclidean
        samples_num = len(data)
        dist_array = np.full(samples_num, float("inf"))
        labels_array = np.zeros(samples_num)
        for i in range(len(centers)):
            for j in range(samples_num):
                cur_dist = dist_func(centers[i],data[j])
                if cur_dist < dist_array[j]:
                    dist_array[j] = cur_dist
                    labels_array[j] = i
        return labels_array


#TODO: tests
class Evaluate:

    @staticmethod
    def informationGain():
        pass

    @staticmethod
    def silhouette():
        pass
