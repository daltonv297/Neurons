import numpy as np
import keyboard
import time
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_linear_network(num_neurons, default_potential, default_impulse):
    neuron_list = [Neuron(default_potential) for _ in range(num_neurons)]
    for i in range(num_neurons-1):
        connection_map = {neuron_list[j]: default_impulse if j == i+1 else 0 for j in range(num_neurons)}
        neuron_list[i].set_connection_map(connection_map)

    connection_map = {neuron_list[j]: default_impulse if j == 0 else 0 for j in range(num_neurons)}
    neuron_list[num_neurons-1].set_connection_map(connection_map)

    return neuron_list

def create_random_network(num_neurons, default_potential, refractory_ticks, stdev):
    neuron_list = [Neuron(default_potential, refractory_ticks) for _ in range(num_neurons)]
    for i in range(num_neurons):
        connection_map = {neuron_list[j]: np.abs(np.random.randn()) * stdev for j in range(num_neurons)}
        neuron_list[i].set_connection_map(connection_map)

    return neuron_list

def create_structured_network(default_potential, refractory_ticks):
    num_neurons = 2
    neuron_list = [Neuron(default_potential, refractory_ticks) for _ in range(num_neurons)]
    neuron_list[0].set_connection_map({neuron_list[0]: 0, neuron_list[1]: -0.6})
    neuron_list[1].set_connection_map({neuron_list[0]: 0, neuron_list[1]: 1})
    neuron_list[1].next_potential = 1

    return neuron_list

class NeuralNetwork:
    def __init__(self, num_neurons, default_potential, default_impulse, refractory_ticks):
        #self.net = create_linear_network(num_neurons, default_potential, default_impulse)
        #self.net = create_random_network(num_neurons, default_potential, refractory_ticks, 0.2)
        self.net = create_structured_network(default_potential, refractory_ticks)

    def tick(self):
        for neuron in self.net:
            neuron.compute_activation()
        for neuron in self.net:
            neuron.move_potentials()

    def input_node_impulse(self, magnitude):
        self.net[0].add_impulse(magnitude)

    def get_output_node(self):
        pass

    def get_all_neurons(self):
        return self.net


class Neuron(NeuralNetwork):
    def __init__(self, action_potential, refractory_ticks):
        self.connection_map = None
        self.action_potential = action_potential
        self.potential = 0
        self.next_potential = 0
        self.activating = False
        self.refractory_ticks = refractory_ticks
        self.tick_count = 0

    def compute_activation(self):
        if (self.potential > self.action_potential) & (self.tick_count == 0):
            self.activating = True
            self.tick_count = self.refractory_ticks
            for neuron in self.connection_map:
                neuron.add_impulse(self.connection_map[neuron] + np.random.randn()*0.1)
        else:
            self.activating = False
            if self.tick_count > 0:
                self.tick_count -= 1

    def move_potentials(self):
        self.potential = self.next_potential
        self.next_potential = 0

    def add_impulse(self, magnitude):
        self.next_potential += magnitude

    def add_connection(self, neuron, out_potential):
        pass

    def set_connection_map(self, connection_map):
        self.connection_map = connection_map

    def get_potential(self):
        return self.potential

    def is_activating(self):
        return self.activating


def main():
    num_neurons = 200
    nn = NeuralNetwork(num_neurons, 0.5, 0.6, 0)
    #nn.input_node_impulse(0.6)

    while True:
        if keyboard.is_pressed('control'):
            nn.input_node_impulse(0.6)

        nn.tick()

        out_str = ' '.join([bcolors.FAIL + '1' + bcolors.ENDC if n.is_activating() else '0' for n in nn.get_all_neurons()])
        sys.stdout.write('\r ' + out_str)

        sys.stdout.flush()

        time.sleep(0.4)


if __name__ == '__main__':
    main()


