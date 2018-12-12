import cluster
import numpy as np
import os

# to remove (maybe)
import time
from sklearn.metrics import silhouette_score

class TestDistance:
    def test_manhattan_array(self):
        a = [1,3,5,7,9]
        b = [2,4,6,8,0]
        c = [1,3,5,7,9]
        d1 = cluster.Distance.manhattan(a, b)
        assert d1 == 13
        d2 = cluster.Distance.manhattan(a, c)
        assert d2 == 0
    
    def test_euclidean_array(self):
        a = [1,3,5,7,9]
        b = [2,4,6,8,0]
        c = [1,3,5,7,9]
        d1 = cluster.Distance.euclidean(a, b)
        assert str(d1)[:11] == "9.219544457"
        d2 = cluster.Distance.euclidean(a, c)
        assert d2 == 0


class TestCentroids:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    samples_path = os.path.join(cur_path, "test_data/x_test.txt")
    labels_path = os.path.join(cur_path, "test_data/y_test.txt")
    def test_centroids_cluster(self):
        # read
        samples = np.loadtxt(self.samples_path)
        true_labels = np.loadtxt(self.labels_path)
        # count clusters
        unique = np.unique(true_labels, return_counts=False)
        # create centroids
        centers = []
        for i in range(len(unique)):
            cur_samples = samples[true_labels==unique[i]]
            centers.append(np.mean(cur_samples, axis=0))
        centers = np.array(centers)
        # (manhattan dist)
        labels_1 = cluster.Centroids.cluster(samples, centers) + 1
        recall_1 = len(labels_1[labels_1==true_labels])/len(true_labels)
        assert recall_1 > 0.8
        # (euclidean dist)
        labels_2 = cluster.Centroids.cluster(samples, centers) + 1
        recall_2 = len(labels_2[labels_2==true_labels])/len(true_labels)
        assert recall_2 > 0.8


class TestEvaluate:
    def test_informatinon_gain(self):
        true_labs = [1,1,2,2,1,2,1,2,2,3,3,3,1,3,2,1,2,3,1,3]
        group_labs =[1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,1,1]
        h = cluster.Evaluate.informationGain(true_labs, group_labs)
        assert str(h)[:7] == "0.05913"
        true_labs  = np.array(true_labs)
        group_labs = np.array(group_labs)
        h = cluster.Evaluate.informationGain(true_labs, group_labs)
        assert str(h)[:7] == "0.05913"

# TODO: clean from unnecessary test
    def test_silhouette(self):
        total_test_time_start = time.time()

        cur_path = os.path.dirname(os.path.abspath(__file__))
        test_samples_path = os.path.join(cur_path, "test_data/x_test.txt")
        test_labels_path = os.path.join(cur_path, "test_data/y_test.txt")
        train_samples_path = os.path.join(cur_path, "test_data/x_train.txt")
        train_labels_path = os.path.join(cur_path, "test_data/y_train.txt")
        
        test_samples = np.loadtxt(test_samples_path)
        test_labels = np.loadtxt(test_labels_path)
        train_samples = np.loadtxt(train_samples_path)
        train_labels = np.loadtxt(train_labels_path)
        
        distances = ["cosine", "braycurtis", "canberra", "chebyshev", "correlation", "hamming", "sqeuclidean", "euclidean", "manhattan"]
        # train, test
        # labels = [train_labels, test_labels]
        # samples = [train_samples, test_samples]
        labels = [test_labels]
        samples = [test_samples]


        scores = []
        print("distance\ttest\ttime")
        for dist_f in distances:
            cur_scores = []
            cur_times = []
            for labs, smpls in zip(labels, samples):
                cur_start = time.time()
                # s = cluster.Evaluate.silhouette(smpls, labs, dist_func=dist_f)
                s = silhouette_score(smpls, labs, metric=dist_f)
                cur_scores.append(s)
                cur_times.append(time.time() - cur_start)
            scores.append(cur_scores)
            print("{}\t{:.3f}\t{:.2f}".format(
                dist_f, cur_scores[0], cur_times[0]))
        ## manhattan
        # s = cluster.Evaluate.silhouette(samples, true_labels, dist_func="manhattan")
        # assert s == 0.9
        ## euclidean
        # s = cluster.Evaluate.silhouette(samples, true_labels, dist_func="euclidean")
        # assert s == 0.9
        print("total time of silhouette test: {}".format(time.time() - total_test_time_start))
        assert False
