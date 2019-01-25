import os
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


def _plot_scores_core(iters, scores, adapt_function,_subplot=(1,1,1), title="Convergence plot"):
    plt.subplot(_subplot[0], _subplot[1], _subplot[2])

    plt.xlabel('Generations (total {}, with initial)'.format(len(iters)), fontsize=11, labelpad=-1)
    plt.ylabel('{} score (max, min, avg, median)'.format(adapt_function), fontsize=11, labelpad=-2)
    plt.title(title, fontsize=12)

    scores_names = ["max", "min", "avg", "median"]
    line_styles = ['-.', ':', '-', '--']

    max_scores = [sc_list.max() for sc_list in scores]
    min_scores = [sc_list.min() for sc_list in scores]
    avg_scores = [sc_list.mean() for sc_list in scores]
    med_scores = [np.median(sc_list) for sc_list in scores]
    scores = [max_scores, min_scores, avg_scores, med_scores]

    for c_scores, c_style in zip( scores, line_styles):
        plt.plot(iters, c_scores, linestyle=c_style)

    plt.legend(scores_names)
    plt.grid()

def plot_scores(iters, scores, adapt_function, params_tuple, to_file=False, out_dir="logs"):
    """
    x_axis : iters
    y_axis : scores_max, scores_min, scores_avg, scores_median
    adapt_function : {"silhouette"|"info_gain"}
    """

    fig = plt.figure(figsize=(8, 8))
    title = "Convergence plot\n" + "(" +", ".join([str(p) for p in params_tuple]) + ")"
    _plot_scores_core(iters, scores, adapt_function, _subplot=(1,1,1), title=title)

    if to_file:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        stamp = str(time.time())
        plt.savefig(os.path.join(out_dir, stamp + '_plot.png'), bbox_inches='tight')
    else:
        plt.show()


def plot_avg_scores(iterations_list, scores_list, adapt_function, best_indices, params_tuple, to_file=False, out_dirs=[]):
    """
    create plot averaged from several experiments
    """

    exp_num = len(iterations_list)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Convergence plot\n" + "(" +", ".join([str(p) for p in params_tuple]) + ")", fontsize=12)
    
    # first 4 experiments (as a max)

    for exp_i, sub_place in zip([i for i in range(min(exp_num, 4))], [1,2,5,6]):
        exp_i_best = scores_list[exp_i][best_indices[exp_i][0]][best_indices[exp_i][1]]
        _plot_scores_core(iterations_list[exp_i], scores_list[exp_i], adapt_function, _subplot=(2,4,sub_place), 
                            title="Experiment {} (best {})".format(exp_i, exp_i_best))

    # averaged
    plt.subplot(1,2,2)
    scores_list = np.array(scores_list)
    # avg best
    avg_best = 0
    for sc_list, best_ind in zip(scores_list, best_indices):
        avg_best += sc_list[best_ind[0], best_ind[1]]
    avg_best /= len(best_indices)

    plt.title("Average of {} experimants (avg best {})".format(len(iterations_list), avg_best))
    plt.xlabel('Generations (total {}, with initial)'.format(len(iterations_list[0])), fontsize=11)
    plt.ylabel('{} score (max, min, avg, median)'.format(adapt_function), fontsize=11, labelpad=-1)

    maxima = scores_list.max(axis=2).mean(axis=0)
    minima = scores_list.min(axis=2).mean(axis=0)
    means = scores_list.mean(axis=2).mean(axis=0)
    medians = np.median(scores_list, axis=2).mean(axis=0)
    assert len(maxima) == len(iterations_list[0])
    assert len(minima) == len(iterations_list[0])
    assert len(means) == len(iterations_list[0])
    assert len(medians) == len(iterations_list[0])

    lists = [maxima, minima, means, medians]
    scores_names = ["max", "min", "avg", "median"]
    line_styles = ['-.', ':', '-', '--']

    for c_scores, c_style in zip(lists, line_styles):
        plt.plot(iterations_list[0], c_scores, linestyle=c_style)

    plt.legend(scores_names)
    plt.grid()
    if to_file:
        out_dir = out_dirs[-1].rstrip("/") + "_averaged"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        stamp = str(time.time())
        plt.savefig(os.path.join(out_dir, stamp + '_plot_avg.png'), bbox_inches='tight')
    else:
        plt.show()


def get_principal_components(data, principal_comp_num=3):
    # PCA
    data = np.array(data)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=principal_comp_num)
    return pca.fit_transform(data)


def _clusters_2d_core_plot(pr_components, labels, chosen_comp = [0,1], subplt_=(1,1,1), labels_map=None, grid=False):
    plt.subplot(subplt_[0], subplt_[1], subplt_[2])
    # chosen components from analysis
    comp1, comp2 = chosen_comp[0], chosen_comp[1]
    plt.xlabel('Principal Component {}'.format(comp1+1), fontsize=12)
    plt.ylabel('Principal Component {}'.format(comp2+1), fontsize=12)
    # plt.title('2 component PCA', fontsize=13)

    labels_uniq = set(labels)
    colors = plot_colors

    for label, color in zip(labels_uniq, colors[:len(labels_uniq)]):
        cur_cluster = pr_components[labels == label, :]
        plt.scatter(cur_cluster[:, comp1],
                   cur_cluster[:, comp2],
                   s=5,
                   c=np.full(cur_cluster.shape[0], color))
    
    if labels_map:
        plt.legend([labels_map[l] for l in labels_uniq])
    else:
        plt.legend(labels_uniq)

    if grid:
        plt.grid()


def plot_clusters_2d(data, labels, labels_map=None, chosen_comp=[0,1], grid=False):
    #PCA
    principal_components = get_principal_components(data, 3)
    # plot each cluster
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("PCA", fontsize=12)
    _clusters_2d_core_plot(principal_components, labels, chosen_comp, (1,1,1), 
                           labels_map=labels_map, grid=grid)

    plt.show()


def plot_clusters_3d(data, labels, labels_map=None):
    # PCA
    principal_components = get_principal_components(data, 3)

    # plot each cluster
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("PCA", fontsize=12)
    ax = fig.add_subplot(111, projection='3d')
    # chosen components from analysis
    comp1, comp2, comp3 = 0, 1, 2
    ax.set_xlabel('Principal Component {}'.format(comp1+1), fontsize=12)
    ax.set_ylabel('Principal Component {}'.format(comp2+1), fontsize=12)
    ax.set_zlabel('Principal Component {}'.format(comp3+1), fontsize=12)
    # ax.set_title('3 component PCA', fontsize=13)

    labels_uniq = set(labels)
    colors = plot_colors

    for label, color in zip(labels_uniq, colors[:len(labels_uniq)]):

        cur_cluster = principal_components[labels == label, :]
        ax.scatter(cur_cluster[:, comp1],
                   cur_cluster[:, comp2],
                   cur_cluster[:, comp3],
                   s=5,
                   c=np.full(cur_cluster.shape[0], color))

    if labels_map:
        ax.legend([labels_map[l] for l in labels_uniq])
    else:
        ax.legend(labels_uniq)

    # ax.grid()
    plt.show()

def plot_clusters(data, labels, labels_map=None, grid=False):
    """
    Build 3 2D PCA plots and 1 3D PCA plot
    """
    # init setting
    labels_uniq = set(labels)
    colors = plot_colors
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PCA | data shape: {}; {} clusters".format(data.shape, len(labels_uniq)), fontsize=13)

    #PCA
    principal_components = get_principal_components(data, 3)

    # 2D
    _clusters_2d_core_plot(principal_components, labels, [0,1], (3,2,1), labels_map, grid)
    _clusters_2d_core_plot(principal_components, labels, [0,2], (3,2,3), labels_map, grid)
    _clusters_2d_core_plot(principal_components, labels, [1,2], (3,2,5), labels_map, grid)

    # 3D
    ax = fig.add_subplot(122, projection='3d')
    comp1, comp2, comp3 = 0, 1, 2
    ax.set_xlabel('Principal Component {}'.format(comp1+1), fontsize=12)
    ax.set_ylabel('Principal Component {}'.format(comp2+1), fontsize=12)
    ax.set_zlabel('Principal Component {}'.format(comp3+1), fontsize=12)
    
    for label, color in zip(labels_uniq, colors[:len(labels_uniq)]):
        cur_cluster = principal_components[labels == label, :]
        ax.scatter(cur_cluster[:, comp1],
                   cur_cluster[:, comp2],
                   cur_cluster[:, comp3],
                   s=5,
                   c=np.full(cur_cluster.shape[0], color))
    if labels_map:
        ax.legend([labels_map[l] for l in labels_uniq])
    else:
        ax.legend(labels_uniq)
    plt.show()
