"particle filtering algorithms"

import sys
import numpy as np
import mpl_toolkits.mplot3d
from matplotlib import pyplot as plt

from smc import models


info = lambda *a, **k: print(*a, file=sys.stderr, **k)

def sis(model, observations, N):
    "sequential important sampling"
    T = len(observations)
    X = np.zeros((T,N))
    W = np.zeros((T,N))

    for t, y in enumerate(observations):
        if t == 0:
            X[t, :] = model.p_initial.rvs(N)
        else:
            X[t, :] = model.sample_transitions(t, X[t-1, :])
        W[t, :] = model.p_emission(t, X[t, :]).pdf(y)
        W[t, :] = W[t, :] / W[t, :].sum()
    return X, W



def sir(model, observations, N):
    "sequential importance resampling"
    T = len(observations)
    X = np.zeros((T, N))
    W = np.zeros((T, N))

    for t, y in enumerate(observations):
        if t == 0:
            X[t, :] = model.p_initial.rvs(N)
        else:
            X[t, :] = model.sample_transitions(t, X[t-1, :])
        W[t, :] = model.p_emission(t, X[t, :]).pdf(y)
        W[t, :] = W[t, :] / W[t, :].sum()
        resampled = np.random.choice(N, N, p=W[t, :])
        info('t {}, eliminated {:.2f}% of particles'.format(t, 100*(1 - len(set(resampled))/N)))
        X[t, :] = X[t, resampled]

    return X, W


def plot_estimate(mean, sd):
    "Reproduction of Figure 2/5 (filtering estimates for SIR/SIS)"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(T), [x for x, y in gen], label= "True Volatility")
    ax.plot(np.arange(T), mean, label="Filter Mean", color='r')
    ax.plot(np.arange(T), mean + sd, label="+/- 1 S.D", linestyle='--', color='g')
    ax.plot(np.arange(T), mean - sd, linestyle='--', color='g')
    title = "SV Model: " + method.upper() + " Filtering Estimates"
    plt.legend()
    plt.title(title)


def plot_particle_distribution(X, W, iterations = [2, 10, 50]):

    N = W[0,:].size
    for n in iterations:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(W[n,:], np.sqrt(N))
        plt.title("n= " + str(n))
        plt.xlabel("Normalized Weights")
        plt.ylabel("Particle Count")
        plt.show()


def plot_distribution(X):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    bins = np.linspace(np.min(X)*0.9, np.max(X)*1.1, 100)
    ax.set_xlim3d(0, T)
    ax.set_ylim3d(bins[0], bins[-1])
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x_t$')
    ax.set_zlabel('$p(x_t | y_{1:t})$')
    step = max(1, T//25)
    for t, (x_t, y_t) in list(enumerate(gen))[::step]:
        counts, bins = np.histogram(X[t, :], bins=bins)
        counts = counts / counts.sum()
        binpoints = (bins[1:] + bins[:-1])/2
        ax.plot(t*np.ones(len(binpoints)), binpoints, counts)
        ax.scatter(t, x_t, counts[np.argmin((binpoints - x_t)**2)])
        #ml_est = np.argmax(counts)
        #ax.scatter(t, binpoints[ml_est], counts[ml_est], color='red')
    return ax


if __name__ == "__main__":
    # generate some data and filter it
    import os
    model = eval(os.environ.get('MODEL', 'models.stochastic_volatility.doucet_example_model'))()
    method = os.environ.get('METHOD', 'sir')
    T = int(os.environ.get('T', '100'))
    N = int(os.environ.get('N', '500'))
    outname = os.environ.get('OUTPUT')
    save_3d = bool(int(os.environ.get('SAVE_3D', '0')))

    info('(model, method, T, N) =', (model, method, T, N))

    gen = list(model.generate(T))
    X, W = eval(method)(model, [y for x, y in gen], N=N)

    # plot the result
    if method == "sis":
        filter_mean = np.average(X, weights=W, axis=1)
        filter_sd = np.sqrt(np.average((X.T - filter_mean).T**2, weights=W, axis=1))
    else:
        filter_mean = X.mean(axis=1)
        filter_sd = X.std(axis=1)

    plot_estimate(filter_mean, filter_sd)
    plot_particle_distribution(X, W)
    ax = plot_distribution(X)
    plt.tight_layout()

    if outname and save_3d:
        ax.view_init(25, 45-1.2)
        plt.savefig('r{}'.format(outname), dpi=300)
        ax.view_init(25, 45+1.2)
        plt.savefig('l{}'.format(outname), dpi=300)
    elif outname:
        ax.view_init(25, 45)
        plt.savefig(outname, dpi=96)
    else:
        ax.view_init(25, 45)
        plt.show()
