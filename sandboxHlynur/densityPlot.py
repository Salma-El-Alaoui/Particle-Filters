'''
Created on 8. jan. 2016

@author: Hlynur
'''

'''
Created on 8. jan. 2016

@author: Hlynur
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
def plotPdf(data, x, t):
    density = gaussian_kde(data)
    xs = np.linspace(min(data),max(data),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()   
    plt.plot(xs,density(xs))
    plt.plot((x[t], x[t]), (0, max(density(xs))), 'k-')
    plt.show()
    
def lericsonPlot(X, x, T):
    __import__('mpl_toolkits.mplot3d')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(55, 45)
    #bins = np.linspace(np.min(X), np.max(X), 100)
    bins = np.linspace(-12, 12, 100)
    ax.set_xlim3d(0, T)
    ax.set_ylim3d(bins[0], bins[-1])
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x_T$')
    ax.set_zlabel('$p(x_T | y_{1:T})$')

    for t in range(0, len(x)):
        if t%8 is not 0 or t<20:
            continue
        density = gaussian_kde(X[t])
        xs = np.linspace(min(X[t]),max(X[t]),200)
        eggs = xs
        density.covariance_factor = lambda : .25
        density._compute_covariance()   
        #plt.plot(xs,density(xs))
        ax.plot([t]*len(eggs),eggs, density(xs))
        index = min(range(len(eggs)), key=lambda i: abs(eggs[i]-x[t]))
        print density(xs)[index]
        ax.scatter(t, x[t], density(xs)[index])
    plt.tight_layout()
    plt.show()