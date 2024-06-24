import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    num_neurons = 200
    dt = 1
    NN = NeuralNetwork(num_neurons, g=1.5, prob_modul=0.003)
    num_iter = 1000
    num_to_plot = 10
    rng = np.random.default_rng()
    neurons_to_plot = rng.choice(num_neurons, num_to_plot, replace=False)
    excitation_hist = []
    for i in range(num_iter):
        excitation_hist.append(NN.excitation[neurons_to_plot])
        NN.update(dt)
        
    plt.plot(excitation_hist)
    plt.show()