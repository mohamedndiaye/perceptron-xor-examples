from random import random 

class Neuron(object): 
    bias = 0 
    output = 0 
    weights = [] 
    def __init__(self, prev_layer_size): 
        self.weights = [random() for i in range(prev_layer_size)] 
    def __repr__(self): 
        return '{"bias":"'+str(self.bias)+'" , "output":"'+str(self.output)+'" , "weights":'+str(self.weights)+'}' 

class Layer(object): 
    activation = '' 
    neurons = [] 
    outputs = [] 
    def __init__(self, nb_neurons, prev_nb_neurons, activation): 
        self.activation = activation 
        self.neurons = [Neuron(prev_nb_neurons) for i in range(nb_neurons)] 
    def __repr__(self): 
        return '{"activation":"'+str(self.activation)+'" , "neurons":'+str(self.neurons)+' , "outputs":'+str(self.outputs)+'}' 

class Network(object): 
    layers = [] 
    def add(self, nb_neurons, activation='identity'): 
        layer = Layer(nb_neurons, 
                      len(self.layers[-1:][0].neurons) if len(self.layers)>0 else 0, activation) 
        self.layers.append(layer) 
    def __repr__(self): 
        return '{"Network":{"layers":'+str(self.layers)+'}}'