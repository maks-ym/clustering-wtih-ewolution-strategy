"""
Simple Genetic Algorithm
"""
import cluster
import numpy as np

def initialize_population(data, centers_num=12, n=20):
    """
    Choose randomly genes for chromosomes between minima and maxima of corresponding genes in data
    data : data to get minima and maxima for generation
    centers_num : number of centroids per individual to generate
    n : number of indiwiduals

    return : initial population pop_0
    """
    data = np.array(data)
    maxima = data.max(axis=0)
    minima = data.min(axis=0)
    return (maxima - minima) * np.random.random_sample((n, centers_num, len(data[0]))) + minima


def get_adapt_scores(pop_t, data, true_labs, adapt_function="silhouette", dist_measure="euclidian"):
    """
    Cluster data for eacn individual (centroids set) in population and get scores.

    pop_t : population to score
    true_labs : for data provided
    adapt_function : {"silhouette"|"info_gain"}; these values to be returned
    dist_measure : {"euclidian"|"manhattan"|"cosine"} used for clustering and silhouette score

    return : scores for each individual; 1-D array of length n (pop_t size)
    """
    dist_func = cluster.Distance.euclidean
    if dist_measure == "manhattan":
        dist_func = cluster.Distance.manhattan
    if dist_measure == "cosine":
        dist_func = cluster.Distance.cosine

    # TODO: vectorize
    scores = np.zeros(len(pop_t))
    for i, individual in enumerate(pop_t):
        labs = cluster.Centroids.cluster(data, individual, dist_func=dist_func)
        if adapt_function == "silhouette":
            scores[i] = cluster.Evaluate.silhouette(data, labs, dist_func=dist_func)
        elif adapt_function == "info_gain":
            labs = cluster.Utils.adjust_labels(labs, true_labs)
            scores[i] = cluster.Evaluate.information_gain(true_labs, labs)
    
    return scores


def reproduction(pop_t, adapt_scores):
    """
    Randomly copy individuals from P_t to T_t, but based on adapt_scores:
    the higher the score the greater propability to be copied (reproduced)

    return : new temporary population T_t
    """
    pass


def crossover(temp_pop_t, prob_cross = 0.2):
    """
    Simple one point crossover of chromosomes (in temporary population T_t).

    Steps:
    - split population into pairs
    - crossover with probability prob_cross, choose randomly the place to split with uniform distribution

    return : modified temporary population T_t
    """
    pass

# TODO : check how mutation works
def mutation(pop_o_t, prob_mutation, min=-1, max=1):
    """
    Mutation of each gene with probability prob_mutation.
    If mutate, choose new gene value from corresponding range between min and max

    return : new child population O_t
    """
    pass


def run_SGA():
    """
    Run the whole Simple Genetic Algotithm.
    # TODO : check what to hold (best last etc)
    Log to file the process ; holds the best indiwidual (?????)
    """
    pass
