import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

plot_colors = ["xkcd:blue",
          "xkcd:green",
          "xkcd:brown",
          "xkcd:red",
          "xkcd:cyan",
          "xkcd:yellow",
          "xkcd:orange",
          "xkcd:darkgreen",
          "xkcd:indigo",
          "xkcd:teal",
          "xkcd:lavender",
          "xkcd:yellowgreen"
          ]

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("{}({},{}) {} sec".format(method.__name__, args, kw, te-ts))
        return result

    return timed


def plot_scores(iters, scores, adapt_function, params_tuple):
    """
    x_axis : iters
    y_axis : scores_max, scores_min, scores_avg, scores_median
    adapt_function : {"silhouette"|"info_gain"}
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Generations (total {})'.format(len(iters-1)), fontsize=15)
    ax.set_ylabel('{} score (max, min, avg, median)'.format(adapt_function), fontsize=15)
    ax.set_title('Convergence plot\n{}'.format(params_tuple), fontsize=20)

    colors = plot_colors

    scores_names = ["max", "min", "avg", "median"]

    max_scores = [sc_list.max() for sc_list in scores]
    min_scores = [sc_list.min() for sc_list in scores]
    avg_scores = [sc_list.mean() for sc_list in scores]
    med_scores = [sc_list.median() for sc_list in scores]
    scores = [max_scores, min_scores, avg_scores, med_scores]

    line_styles = ['-.', ':', '-', '--']

    for c_name, c_scores, c_style in zip(scores_names, scores, line_styles):
        ax.plot(iters, c_scores, linestyle=c_style)

    ax.legend(scores_names)
    # ax.grid()
    plt.show()



def plot_clusters(data, labels, labels_map=None):
    # PCA
    data = np.array(data)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=4)
    principal_components = pca.fit_transform(data)
    # plot each cluster
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # chosen components from analysis
    comp1, comp2 = 0, 2
    ax.set_xlabel('Principal Component {}'.format(comp1+1), fontsize=15)
    ax.set_ylabel('Principal Component {}'.format(comp2+1), fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    labels_uniq = set(labels)
    colors = plot_colors

    for label, color in zip(labels_uniq, colors[:len(labels_uniq)]):
        cur_cluster = principal_components[labels == label, :]
        ax.scatter(cur_cluster[:, comp1],
                   cur_cluster[:, comp2],
                   c=np.full(cur_cluster.shape[0], color))

    if labels_map:
        ax.legend([labels_map[l] for l in labels_uniq])
    else:
        ax.legend(labels_uniq)
    # ax.grid()
    plt.show()


def plot_clusters_3d(data, labels, labels_map=None):
    # PCA
    data = np.array(data)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=4)
    principal_components = pca.fit_transform(data)

    # plot each cluster
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # chosen components from analysis
    comp1, comp2, comp3 = 0, 1, 2
    ax.set_xlabel('Principal Component {}'.format(comp1+1), fontsize=15)
    ax.set_ylabel('Principal Component {}'.format(comp2+1), fontsize=15)
    ax.set_zlabel('Principal Component {}'.format(comp3+1), fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    labels_uniq = set(labels)
    colors = plot_colors

    for label, color in zip(labels_uniq, colors[:len(labels_uniq)]):

        cur_cluster = principal_components[labels == label, :]
        ax.scatter(cur_cluster[:, comp1],
                   cur_cluster[:, comp2],
                   cur_cluster[:, comp3],
                   c=np.full(cur_cluster.shape[0], color))

    if labels_map:
        ax.legend([labels_map[l] for l in labels_uniq])
    else:
        ax.legend(labels_uniq)

    # ax.grid()
    plt.show()
