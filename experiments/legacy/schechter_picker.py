import numpy as np

def schechter_picker(N,alpha=-1.3, M_star=3.e10, M_min=1.e5):
    """
    Draw masses at random from a Schechter (1976) function
    N(M) ~ (M/M*)**alpha * exp(M/M*)   
    Adapted from https://gist.github.com/joezuntz/5056136
    Based on algorithm in http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf

    Parameters
    ----------
    N : integer
        The number of random masses to return

    alpha : float
        The slope of the powerlaw portion of the luminosity function (-1.3 default)

    M_star : float
        The mass of the break of the mass function (3.e10 solar masses defaut)

    M_min : float
        The minimum mass (1.e5 default)
    """
    n=0
    output = []
    while n<N:
        M = np.random.gamma(scale=M_star, shape=alpha+2, size=N)
        M = M[M>M_min]
        u = np.random.uniform(size=M.size)
        M = M[u<M_min/M]
        output.append(M)
        n+=M.size
    return np.concatenate(output)[:N]
