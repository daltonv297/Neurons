import numpy as np

class NeuralNetwork:
    def __init__(self, num_neurons, prob_con=1, g=1.5, tau=30, alpha_modul=16, prob_modul=0.003):
        self.num_neurons = num_neurons
        self.prob_con = prob_con
        self.g = g
        self.tau = tau
        self.alpha_modul = alpha_modul
        self.prob_modul = prob_modul
        self.rng = np.random.default_rng()
        self.connection_map = np.zeros((num_neurons, num_neurons))
        self.excitation = self.rng.uniform(-0.1, 0.1, num_neurons)
        self.initialize(g)
        
        
    def initialize(self, g):
        self.connection_map = g * self.rng.standard_normal(self.connection_map.shape) \
            / np.sqrt(self.prob_con * self.num_neurons)
        self.connection_map[self.rng.uniform() < 1 - self.prob_con] = 0

    
    def update(self, dt):
        activation = sigmoid(self.excitation)
        random_exc = self.rng.uniform(-1, 1, self.num_neurons) * self.alpha_modul
        random_exc[self.rng.uniform() < 1 - self.prob_modul] = 0
        delta_exc = self.connection_map @ activation + random_exc
        self.excitation += dt / self.tau  * (-self.excitation + delta_exc)

    
    def perturb(self, mag=1):
        self.excitation += self.rng.uniform(-1, 1, self.num_neurons) * mag


def sigmoid(x):
    return 1 / (1 + np.exp(-100*x))