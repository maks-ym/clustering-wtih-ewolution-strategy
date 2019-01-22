"""
Simple Genetic Algorithm
"""

def initialize_population(data, individual_shape, n):
    """
    Choose randomly genes for chromosomes between minima and maxima of corresponding genes in data
    data : data to get minima and maxima for generation
    individual_shape : shzpe of one indiwidual
    n : number of indiwiduals
    """
    pass


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
def mutation(pop_o_t, prob_mutation, mins, maxs):
    """
    Mutation of each gene with probability prob_mutation.
    If mutate, choose new gene value from corresponding range between min and max

    return : new child population O_t
    """
    pass
