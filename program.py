import neuralnetwork as nn 
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = (784,16,10)
trainingrounds = 10
mini_batch_size = 20
learning_rate = 0.1

training_data = list(zip(training_images, training_labels))

net = nn.NeuralNetwork(layer_sizes)

for e in range(trainingrounds):
    net.SGD(training_data, mini_batch_size, learning_rate)
    net.print_accuracy(training_images, training_labels)

