import neuralnetwork as nn 
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = (784,24,10)

net = nn.NeuralNetwork(layer_sizes)

trainingrounds = 10
#for e in range(trainingrounds):
    #net.train(training_images, training_labels)
    #net.print_accuracy(training_images, training_labels)

