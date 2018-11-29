import time

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("{}({},{}) {} sec".format(method.__name__, args, kw, te-ts))
        return result

    return timed

def plot_clusters(data, labels):
    pass
    # PCA

    # plot each cluster
