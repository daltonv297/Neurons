import numpy as np
import matplotlib.pyplot as plt 
from pynput import keyboard
from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    num_neurons = 200          
    dt = 1                    
    NN = NeuralNetwork(num_neurons, g=1.5, prob_modul=0, alpha_modul=1)
    num_iter = 1000
    num_to_plot = 10
    plot_window = 200
    animate = False
    rng = np.random.default_rng()
    neurons_to_plot = rng.choice(num_neurons, num_to_plot, replace=False)
    excitation_hist = []

    def on_press(key):
        global run
        if key == keyboard.Key.space:
            NN.perturb()
        if key == keyboard.Key.esc:
            run = False
        
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    run = True
    if animate:
        i = 0
        while run:
            plt.cla()
            excitation_hist.append(NN.excitation[neurons_to_plot])
            NN.update(dt)
            if len(excitation_hist) < plot_window:
                to_plot_x = list(range(i+1))
                to_plot_y = excitation_hist
            else:
                to_plot_x = list(range(i-plot_window+1, i+1))
                to_plot_y = excitation_hist[-plot_window:]

            plt.plot(to_plot_x, to_plot_y)
            plt.pause(0.01)
            i += 1
        
    else:
        for i in range(num_iter):
            plt.cla()
            excitation_hist.append(NN.excitation[neurons_to_plot])
            NN.update(dt)
            if animate:
                if len(excitation_hist) < plot_window:
                    to_plot_x = list(range(i+1))
                    to_plot_y = excitation_hist
                else:
                    to_plot_x = list(range(i-plot_window+1, i+1))
                    to_plot_y = excitation_hist[-plot_window:]

                plt.plot(to_plot_x, to_plot_y)
                plt.pause(0.01)

        plt.plot(excitation_hist)
        plt.show()