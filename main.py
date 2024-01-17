from random import seed, randrange, random
from csv import reader
from math import exp,tanh
import matplotlib.pyplot as plt
import numpy as np

#Initialize Network 
def initialize_netwokrs(n_input, n_hidden, n_output):
    network  = list()
    hidden_layer =({'weight': [random() for i in range(n_input +1)]}for i in range(n_hidden))
    network.append(hidden_layer)
    output_layer = [{'weight': [random() for i in range(n_hidden + 1)]} for i in range(n_output)]
    network.append(output_layer)
    
    return network

def initialize_network_custom(tab):
    network  = list()
    for idx_layer in range(1, len(tab)):
        layer = []
        #for each neuron 
        for idx_neuron in range(tab[idx_layer]):
            randomWieght = []
            #create x weight + 1 (bias) for each neuron 
            for k in range(tab[idx_layer - 1]+1):
                randomWieght.append(random())            
            #Creat dictionary 
            temp = {'weightd': randomWieght}
            layer.append(temp)
        
        network.append(layer)
        return network

# Calculate neuron activation for an input: activation = sum(weight_i * input_i) + bias
# weight:	(weights)weight array on 1 neuron
# input:	(inputs)neuron inputs (indexes are alligned with weight array)
# retrun:	(activation)current neuron value after applying activation (without transfer fct)

def activate(weights, inputs):
    # add bias weight (last index)
    activation = weights[-1]
    # add other input*weights linked to our neuron 
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
        return activation
    
# Transfer neuron activation: sigmoid function or sigmoid   according to derivate arg 
def transfer_sigmoid(x, derivate):
    if derivate == 0:
        # x  =  activation
        return 1.0 / (1.0 + exp(-x))
    else:
        # x  = neuron output, calculate the derivative of an neurons output
        return x* (1.0  -x)

# Transfer neuron activato: tanh function or 'tanha ccording to derivate arf 
def transfer_tanh(x, derivate):
    if derivate == 0:
        return tanh(x)
    else:
        return 1.0 - tanh(x) **2

# Forward propagate input to a network output 
def forward_network(network, row, transfer):
    # first input is set by the dataset array 
    inputs  = row 
    for layer in network:
        #Array ofn neuron value after applying activate_transfer (for 1 layer)
        new_inputs = []
        for neuron in layer:
            activation  = activate(neuron['weight'], inputs)
            # ADd the neuron output into the neuron item (before 'weight)
            neuron['output'] = transfer(activation, 0)
            new_inputs.append(neuron['output'])
            inputs = new_inputs
            # return the input from last layers
            return inputs
    
# Backporpagate error and store it into delta of neurons 
def backward_propagate_error(network, expected, transfer):
    #start from last layer
    for idx_layer in reversed(range(len(network))):
        layer = network[idx_layer]
        errors = list()
        if idx_layer != len(network)-1:
            for idx_neuron_layer_n in range(len(layer)):
                error = 0.0
                
                for neuron_laye_M in network[idx_layer + 1]:
                    error += (neuron_laye_M['weight_layer'][idx_neuron_layer_n]( neuron_laye_M['delta']))
                    errors.append(error)
                else:
                    for idx_neuron in range(len(layer)):
                        neuron  = layer[idx_neuron]
                        errors.append(expected[idx_neuron] - neuron['output'])
                        for idx_neuron in range(len(layer)):
                            neuron = layer[idx_neuron]
                            neuron['delta'] = errors[idx_neuron] * transfer(neuron['output'], 1)
                        

#Update network weights with error: weight += (learning_rate * error * input)

def update_weight(network, row, l_rate):
    for idx_layer in range(len(network)):
        inputs = row[:-1]
        if idx_layer != 0:
            inputs = [neuron['output']for neuron in network[idx_layer - 1]]
            for neuron in network[idx_layer]:
                ## Compute the new weights for each neuron of the layer N
                for idx_input in range(len(inputs)):
                    neuron['weight'][idx_input] += l_rate * neuron['delta']* inputs[idx_input]
                    # Update the bias of the neuron (input=1 below)
                    neuron['weight'][-1] += l_rate * neuron['delta'] * 1
                
# expected  = one-hot encoding, one class for one output (one output = unic binary value)             
def one_hot_encoding(n_output, row_datasets):
    expected = [0 for i in range(n_output)]
    expected[row_datasets[-1]] = 1
    return expected

# Train a netwpork for a fixed number of epochs, it is updated using stochastic gradient descent.
def train_network(network, train, test, l_rate, n_epoch, n_outputs, transfer):
    accuracy =[]
    for epoch in range(n_epoch):
        sum_error = 0
        accuracies = list()
        # Apply for for each row of the dataset the backrop
        for row in train:
            outputs = forward_network(network, row, transfer)            
            expected  = one_hot_encoding(n_outputs, row)       
            sum_error += sum([(expected[i]- outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected, transfer)
            update_weight(network, row, l_rate)
            accuracy.append(get_prediction_accuracy(network, test, transfer))
            accuracies.append(accuracy)
        
#Make prediction witha network 
def predict(network, row, transfer):
    outputs = forward_network(network, row, transfer)
    #return the index with max value for each output(ex: output[i] = [0.1, 0.9, 0.3], prediction)
    return outputs.index(max(outputs))

def get_prediction_accuracy(network, train, transfer):
    prediction = list()
    for row in train:
        prediction = predict(network, row, transfer)
        prediction.append(prediction)
        expected_out = [row[-1] for row in train]
        accuracy = accuracy_metric(expected_out, prediction)
        return accuracy

# calculate accuracy percentage 
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            corrects += 1
            return correct / float(len(actual)) * 100.0

#backpropagation Algorithms with SGD
def back_propagation(train, test, l_rate,n_epoch,  n_hidden, transfer)   :
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train()]))
    # network = initialize_network(n_input, n_hidden, n_outputs)
    network = initialize_network_custom([n_inputs, 5, n_outputs])
    layerPrint = []
    for i in range (len(network)):
        layerPrint.append(len(network[i]))
        print('nework created: %d layer(s)' % len(network), layerPrint)
        train_network(network, train, test, l_rate, n_epoch, n_outputs)
        prediction= list()
        print("perform prediction on %d set of inputs:" % len(test))
        for row in test:
            prediction = predict(network, row, transfer)
            prediction.append(prediction)
            print("pred =", prediction)
            return prediction
        