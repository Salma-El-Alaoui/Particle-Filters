"stochastic volatility model"

import numpy as np
from scipy import stats


class markov_model():
    def generate(self, T):
        x = self.p_initial.rvs()
        for t in range(T):
            y = self.p_emission(t, x).rvs()
            yield x, y
            x = self.p_transition(t, x).rvs()

    def sample_transitions(self, t, X):
        return [self.p_transition(t, x).rvs() for x in X]



class quick_transition_sampling():
    """mixin class that adds a really fast sample_transitions method, requires
    the mean function to be isolated as p_transition_mean"""

    def sample_transitions(self, t, X):
        p = self.p_transition(t, X[0])
        return p.rvs(X.shape) - p.mean() + self.p_transition_mean(t, X)


class stochastic_volatility(quick_transition_sampling, markov_model):
    """stochastic volatility is a model in which the next state X_n is given by
    the previous state X_n-1 and some volatility variable V_n, that is to say

      X_n = alpha * X_n + sigma * V_n

    and the emission Y_n is an exponential function of the state with some
    wiener process W_n ~ N(0, 1),

      Y_n = beta * exp(X_n / 2) * W_n

    long story short, we get the following

      transition pdf:    f(x' | x) = N(x'; alpha*x, sigma2)
      emission pdf:       g(y | x) = N(y; 0, beta2*exp(x))

    lastly the initial state is chosen so the marginal distribution of X_n is
    equal to the initial pdf,

      initial state pdf:      µ(x) = N(x; 0, sigma2 / (1.0 - alpha2))
    """

    def __init__(self, alpha, beta, sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.p_transition_mean = lambda t, x: alpha*x
        self.p_transition = lambda t, x: stats.norm(alpha*x, sigma)
        self.p_emission = lambda t, x: stats.norm(0, beta*np.exp(x/2.0))
        self.p_initial = stats.norm(0, sigma/np.sqrt(1.0 - alpha**2))

    @classmethod
    def doucet_example_model(cls):
        return cls(alpha=0.91, sigma=1.0, beta=0.5)


class trig_toy_model(quick_transition_sampling, markov_model):
    def __init__(self, sigma0=np.sqrt(10), sigmav=np.sqrt(10), sigmaw=1):
        self.p_initial = stats.norm(0, sigma0)
        self.p_transition_mean = lambda t, x: x/2 + 25*x/(1 + x**2) + 8*np.cos(1.2*t)
        self.p_transition = lambda t, x: stats.norm(self.p_transition_mean(t, x), sigmav)
        self.p_emission = lambda t, x: stats.norm(x**2/20, sigmaw)


if __name__ == "__main__":
    model = stochastic_volatility.doucet_example_model()
    print('t,x_t,y_t')
    for t, (x, y) in enumerate(model.generate(500)):
        print(','.join(map(str, (t, x, y))))
