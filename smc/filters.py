"particle filtering algorithms"

import sys
import numpy as np
from scipy import stats
from scipy.stats import entropy
import mpl_toolkits.mplot3d
from matplotlib import pyplot as plt

from smc import models


info = lambda *a, **k: print(*a, file=sys.stderr, **k)

def smoothingDis(filtering_func, model, observations, N):
    T = len(observations)
    X, W = filtering_func(model, observations, N)
    P = np.zeros((T, N))
    P[T-1, :] = W[T-1, :]
    for t in range(T-2, -1 ,-1):
        P[t, :] = W[t, :]*model.p_transition(t, X[t,:]).pdf(X[t,:])
    return X,P

def sis(model, observations, N):
    "sequential important sampling"
    T = len(observations)
    X = np.zeros((T,N))
    W = np.zeros((T,N))

    for t, y in enumerate(observations):
        if t == 0:
            X[t, :] = model.p_initial.rvs(N)
            W[t, :] = model.p_emission(t, X[t, :]).pdf(y)
        else:
            X[t, :] = model.sample_transitions(t, X[t-1, :])
            W[t, :] = W[t-1, :]*model.p_emission(t, X[t, :]).pdf(y)

    for t, y in enumerate(observations):
        W[t, :] = W[t, :] / W[t, :].sum()
    return X, W



def sir(model, observations, N, resampling_criterion=lambda X, W, t: True):
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

        if resampling_criterion(X, W, t):
            resampled = np.random.choice(N, N, p=W[t, :])
            eliminated = 1 - len(set(resampled))/N
            info('t {}, eliminated {:.2f}% of particles'.format(t, 100*eliminated))
            X[t, :] = X[t, resampled]

    return X, W


def sir_adaptive_ess(model, observations, N, threshold=2):
    ess = lambda vals: 1/((vals**2).sum())
    ess_crit = lambda X, W, t: ess(W[t, :]) < N/threshold
    return sir(model, observations, N, resampling_criterion=ess_crit)


def sir_adaptive_entropy(model, observations, N, threshold=1.055):
    ent_crit = lambda X, W, t: entropy(W[t, :]) < np.log(N)/threshold
    return sir(model, observations, N, resampling_criterion=ent_crit)


def apf(model, observations, N):
    "auxiliary particle filtering"
    # Assumptions: q = f, and as stated in reference 23,
    # "A note on auxiliary particle filters", the function
    # p(y_t|x_t-1) is approximated as g(y_t|mu(x_t-1)), where
    # mu is the mode, mean, or median of the function f(x_t|x_t-1). As
    # f is a gaussian distribution in our case, the mean = median = mode.
    # This assumption could lead to an estimator with a very large or infinite
    # variance. If one attempts to calculate the integral that gives p(y_t|x_t-1), one
    # will arrive with an integrand that is of the form exp(exp(x)), i.e. intractable. An approximation
    # of this integral should be possible to calculate using numerical approaches.
    T = len(observations)
    X = np.zeros((T,N))
    W = np.zeros((T,N))

    for t, y in enumerate(observations):
        if t == 0: #Init, T = 0: w_1 = g(y_1|x_1) * g(y_2|alpha*x_1):
            X[t, :] = model.p_initial.rvs(N)
            W[t, :] = model.p_emission(t, X[t, :]).pdf(y)*model.p_emission(t, X[t, :]*model.alpha).pdf(y)
            #print("In t == 0, y -->" + str(y))
        else: #T > 0: a_n = g(y_n|x_n) * g(y_n+1|alpha*x_n)/g(y_n|alpha*x_n-1)
            X[t, :] = model.sample_transitions(t, X[t-1, :])
            W[t, :] = model.p_emission(t, X[t, :]).pdf(y)*model.p_emission(t, X[t, :]*model.alpha).pdf(y)

            denominator = model.p_emission(t, X[t-1, :]*model.alpha).pdf(y)
            #print("min value denominator " + str(min(denominator)))
            #print(str(len(denominator)))
            denominator = [10**(-300) if (x==0) else x for x in denominator]
            #print(str(min(W[t,:])))
            W[t, :] = np.divide(W[t, :],denominator)
            #print(str(min(W[t,:])))
            #print("----")
        W[t, :] = W[t, :] / W[t, :].sum()
        resampled = np.random.choice(N, N, p=W[t, :])
        info('t {}, eliminated {:.2f}% of particles'.format(t, 100*(1 - len(set(resampled))/N)))
        X[t, :] = X[t, resampled]

    print(str(X.shape) + " " + str(W.shape))
    return X, W

def mcmc(model, observations, N):
    "smc filtering with mcmc Moves"
    T = len(observations)
    X = np.zeros((T,N))
    W = np.zeros((T,N))
    L = 10 #Fixed lag

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

        #MCMC Kernels. We sample N samples of X from the kernel at each iteration and use accepted X to replace the
        #corresponding from the q-function.

        #Candidate sampling according to Metropolis-Hastings (MH) on page 27 with some minor modifications.
        if t == 1 or t < T-1:
            candidate = stats.norm(0.5*(model.alpha*X[t-1, :]+(1/model.alpha)*X[t+1, :]), model.sigma**2).pdf(X[t, :]) #Essentially, this is the self-engineered proposal distribution

            acceptance_probability_nominator = model.p_emission(t, candidate).pdf(y) * \
                                               model.p_transition(t, candidate).pdf(X[t+1, :]) * \
                                               model.p_transition(t, X[t-1, :]).pdf(X[t, :]) #What should q in the nominator  be?

            acceptance_probability_denominator = model.p_emission(t, X[t, :]).pdf(y) * model.p_transition(t, X[t, :]).pdf(X[t+1, :]) * \
                                                 model.p_transition(t, X[t-1, :]).pdf(X[t, :]) * \
                                                 stats.norm(0.5*(model.alpha*X[t-1, :]+(1/model.alpha)*X[t+1, :]), model.sigma**2).pdf(X[t, :])
                                                 # The transition functions f(x_k|x'_k-1) and f(x'_k|x'_k-1) should be the same because
                                                 # we do not store the previous candidates x'_k-1, i.e. they are in the same list as
                                                 # x_k-1. We should therefore let them cancel out according to mathematical
                                                 # procedure.

            acceptance_probability = acceptance_probability_nominator/acceptance_probability_denominator
            for i in range(N):
                if min(1, acceptance_probability[i]) >= 1:
                    X[t+1, i] = candidate[i]
                else:
                    sample = np.random.multinomial(1, [acceptance_probability[i], 1-acceptance_probability[i]], size=1)
                    if sample[0][0] == 1:
                        X[t+1, i] = candidate[i]
                    else:
                        X[t+1, i] = X[t, i]


    return X, W

def block(model, observations, N, L=10, resampling_criterion=lambda X, W, t: True):
    "block sampling"
    T = len(observations)
    X = np.zeros((T, N))
    Xnew = np.zeros((T, N))
    W = np.zeros((T, N))

    for t in range(0, T):
        if t == 0:
            X[t, :] = model.p_initial.rvs(N)
            W[t, :] = model.p_emission(t, X[t, :]).pdf(observations[t])
        elif t < L:
            Xnew[0, :] = model.p_initial.rvs(N)
            W[t, :] = np.ones(N)
            for tt in range (t-1, t+1):
              Xnew[tt, :] = model.sample_transitions(tt, Xnew[tt-1, :])
              W[t, :] *= model.p_emission(tt, Xnew[tt, :]).pdf(observations[tt])            
        else:
            Xnew[t-L, :] = np.array(X[t-L, :])
            Wtop = np.ones(N)
            Wbot = np.ones(N)
            for tt in range (t-L+1, t+1):
              Xnew[tt, :] = model.sample_transitions(tt, Xnew[tt-1, :])
              Wtop *= model.p_emission(tt, Xnew[tt, :]).pdf(observations[tt])
              if tt <= t:
                Wbot *= model.p_emission(tt, X[tt, :]).pdf(observations[tt])
            Wbot = [10**(-300) if (x==0) else x for x in Wbot]
            W[t, :] = Wtop / Wbot

        W[t, :] = W[t, :] / W[t, :].sum()

        if resampling_criterion(Xnew, W, t):
            resampled = np.random.choice(N, N, p=W[t, :])
            eliminated = 1 - len(set(resampled))/N
            info('t {}, eliminated {:.2f}% of particles'.format(t, 100*eliminated))
            if t < L:
                X[:, :] = Xnew[:, resampled]
            else:
                #X[0:t-L, :] = X[0:t-L, resampled]
                X[t-L:t, :] = Xnew[t-L:t, resampled]
        else:
            X[t, :] = Xnew[t, :]

    return X, W

def block_adaptive_ess(model, observations, N, L=10, threshold=2):
    ess = lambda vals: 1/((vals**2).sum())
    ess_crit = lambda X, W, t: ess(W[t, :]) < N/threshold
    return block(model, observations, N, L, resampling_criterion=ess_crit)


def plot_estimate(mean, sd, filtering = 1, smoothing_method = None):
    "Reproduction of Figure 2/5 (filtering estimates for SIR/SIS)"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(T), [x for x, y in gen], label= "True Volatility")
    ax.plot(np.arange(T), mean, label="Filter Mean", color='r')
    ax.plot(np.arange(T), mean + sd, label="+/- 1 S.D", linestyle='--', color='g')
    ax.plot(np.arange(T), mean - sd, linestyle='--', color='g')
    if filtering:
        title = "SV Model: " + method.upper() +" Filtering Estimates"
    else:
        title = "SV Model: " + smoothing_method.upper() +" Smoothing Estimates"
    plt.legend()
    plt.title(title)
    return ax

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

def compute_mean_sd(X, W=None):
    # plot the result
    if W != None:
        filter_mean = np.average(X, weights=W, axis=1)
        filter_sd = np.sqrt(np.average((X.T - filter_mean).T**2, weights=W, axis=1))
    else:
        filter_mean = X.mean(axis=1)
        filter_sd = X.std(axis=1)

    return filter_mean, filter_sd


def get_avg_err(model, N, method, gen):
    X, W = eval(method)(model, [y for x, y in gen], N=N)
    if method == "sis":
        filter_mean = np.average(X, weights=W, axis=1)
        filter_sd = np.sqrt(np.average((X.T - filter_mean).T**2, weights=W, axis=1))
    else:
        filter_mean = X.mean(axis=1)
        filter_sd = X.std(axis=1)
    avg_err = np.sum(np.abs(filter_mean - gen[:, 0]))/100
    avg_sqe = np.sum(np.square(filter_mean - gen[:, 0]))/100
    avg_sd = np.sum(filter_sd)/100
    return np.array([avg_err, avg_sqe, avg_sd])

def performance_test(model, N, methods):
    for method in methods:
        avgs = np.zeros(3)
        for i in range (0, 20):
            gen = np.genfromtxt('batches/sv{}.csv'.format(i), delimiter=',')
            avgs += get_avg_err(model, N, method, gen)
        avgs /= 20
        info('(method, avg_err, avg_sqe, avg_sd) =', (method, avgs[0], avgs[1], avgs[2]))

if __name__ == "__main__":
    # generate some data and filter it
    import os
    model = eval(os.environ.get('MODEL', 'models.stochastic_volatility.doucet_example_model'))()
    method = os.environ.get('METHOD', 'sir')
    T = int(os.environ.get('T', '100'))
    N = int(os.environ.get('N', '500'))
    outname = os.environ.get('OUTPUT')
    save_3d = bool(int(os.environ.get('SAVE_3D', '0')))
    sv_csv = os.environ.get('SVCSV', '')
    create_csv = bool(int(os.environ.get('CREATECSV', '0')))
    perf_test = bool(int(os.environ.get('PERFTEST', '0')))

    info('(model, method, T, N) =', (model, method, T, N))

    if perf_test:
      performance_test(model, N, ['sis', 'sir', 'sir_adaptive_ess', 'sir_adaptive_entropy', \
                                  'apf', 'mcmc', 'block', 'block_adaptive_ess'])

    if sv_csv == '':
      gen = list(model.generate(T))
      if create_csv:
        np.savetxt("sv.csv", gen, delimiter=',')
    else:
      gen = np.genfromtxt(sv_csv, delimiter=',')
      T = gen.shape[0]

    X, W = eval(method)(model, [y for x, y in gen], N=N)
    Xs, Ws = smoothingDis(sis, model, [y for x, y in gen], N=N)
    mean_sis, sd_sis = compute_mean_sd(Xs, Ws)
    Xsr, Wsr = smoothingDis(sir, model, [y for x, y in gen], N=N)
    mean_sir, sd_sir = compute_mean_sd(Xsr, Wsr)

    # plot the result
    if method == "sis":
        filter_mean, filter_sd = compute_mean_sd(X, W)
    else:
        filter_mean, filter_sd = compute_mean_sd(X)

    #estimates plots
    ax = plot_estimate(filter_mean, filter_sd)
    plt.show()
    ax = plot_estimate(mean_sis, sd_sis, filtering = 0, smoothing_method = "sis")
    plt.show()
    ax = plot_estimate(mean_sir, sd_sir, filtering = 0, smoothing_method = "sir")
    plt.show()

    # particle histograms
    plot_particle_distribution(X, W)


    ax = plot_distribution(X)
    plt.title('$\mathcal{{M}}$ = {}, $T$ = {}, $N$ = {}'.format(method, T, N))
    plt.tight_layout()

    if outname and save_3d:
        ax.view_init(25, 45-1.2)
        plt.savefig('r{}'.format(outname), dpi=300)
        ax.view_init(25, 45+1.2)
        plt.savefig('l{}'.format(outname), dpi=300)
    elif outname:
        ax.view_init(25, 55)
        plt.savefig(outname, dpi=96)
    else:
        ax.view_init(25, 45)
        plt.show()
