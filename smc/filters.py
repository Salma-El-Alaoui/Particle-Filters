"particle filtering algorithms"

import sys
import numpy as np

from smc import models


info = lambda *a, **k: print(*a, file=sys.stderr, **k)


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


if __name__ == "__main__":
    # generate some data and filter it
    import os
    model = eval(os.environ.get('MODEL', 'models.stochastic_volatility.doucet_example_model'))()
    method = os.environ.get('METHOD', 'sir')
    T = int(os.environ.get('T', '10'))
    N = int(os.environ.get('N', '100'))
    info('(model, method, T, N) =', (model, method, T, N))
    model = models.trig_toy_model()
    gen = list(model.generate(T))
    X, W = eval(method)(model, [y for x, y in gen], N=N)

    # plot the result

    __import__('mpl_toolkits.mplot3d')
    from matplotlib import pyplot as plt
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

    plt.tight_layout()
    ax.view_init(25, 45-1.2)
    plt.savefig('count-r.png', dpi=300)
    ax.view_init(25, 45+1.2)
    plt.savefig('count-l.png', dpi=300)
    #plt.plot(bins, counts)
