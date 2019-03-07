"""
Monte Carlo Integration
Jake Callahan

Many important integrals cannot be evaluated symbolically because the integrand
has no antiderivative. Traditional numerical integration techniques like
Newton-Cotes formulas and Gaussian quadrature usually work well for one-dimensional
integrals, but rapidly become inefficient in higher dimensions. Monte Carlo
integration is an integration strategy that has relatively slow convergence, but
that does extremely well in high-dimensional settings compared to other techniques.
In this program I implement Monte Carlo integration and apply it to a classic
problem in statistics.
"""
import numpy as np
from scipy import stats
import scipy.linalg as la
from matplotlib import pyplot as plt

def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    points = np.random.uniform(-1,1,(n,N))
    lengths = la.norm(points, axis = 0)
    estimate = 2**n*np.count_nonzero(lengths < 1) / N
    return estimate

def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #Get sampled points
    points = np.random.uniform(a,b,(1,N))
    #Get y vals for f
    yk = f(points)
    #Sum f(xi) over all i
    sum = np.sum(yk)
    #Get volume of interval
    V = np.abs(b-a)

    return V/N * sum

def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #Cast mins and maxes to arrays
    mins = np.array(mins)
    maxs = np.array(maxs)

    #Get n, sample points, and volume
    n = len(mins)
    x = np.random.uniform(0,1,(n,N))
    V = np.prod(maxs-mins)

    #Put sample points in correct interval
    for i in range(n):
        x[i] = x[i]*(maxs[i] - mins[i]) + mins[i]

    #Evaluate sample points
    y = [f(x[:, i]) for i in range(N)]
    #Return volume
    return V * np.sum(y) / N

def integrate_normal():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #Define omega
    mins = [-3/2, 0, 0, 0]
    maxs = [3/4, 1, 1/2, 1]
    #Define f
    f = lambda x: (1/(4*np.pi**2)) * np.exp(-1 * x.T @ x/2)

    #Get exact value
    means, cov = np.zeros(4), np.eye(4)
    exact_val = stats.mvn.mvnun(mins, maxs, means, cov)[0]

    #Define items for loop
    domain = np.logspace(1,5,20)
    rel_error = []

    for N in domain:
        #Get approximation of F
        approx_val = mc_integrate(f,mins,maxs,int(N))
        #Get relative error
        rel_error.append(np.abs(exact_val - approx_val) / np.abs(exact_val))

    plt.loglog(domain, rel_error, marker = 'o', label = 'Relative Error')
    plt.loglog(domain, 1/np.sqrt(domain), marker = 'o', label = '1/$\sqrt{N}$')
    plt.title("Relative error of montecarlo integration")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
