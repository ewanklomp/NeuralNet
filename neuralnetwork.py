import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(w,a) + b)
        return a

    def SGD(self, images, mini_batch_size, learningrate): #stochastic gradient descent, split everything up in minibatches for efficiency
        
        imagecount = len(images)
        random.shuffle(images)
        mini_batches = [images[k:k+mini_batch_size] for k in range(0, imagecount, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learningrate)

        print("SGD finished")

    def update_mini_batch(self, mini_batch, learningrate):
        del_b = [np.zeros(b.shape) for b in self.biases] 
        del_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_b, delta_w = self.backprop(x,y)
            del_b = [nb+dnb for nb, dnb in zip(del_b, delta_b)]
            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_w)]
        
        self.weights = [w-(learningrate/len(mini_batch))*delw for w, delw in zip(self.weights, del_w)]
        self.biases = [b-(learningrate/len(mini_batch))*delb for b, delb in zip(self.biases, del_b)]


    def backprop(self, x, y): #to make
        del_b = [np.zeros(b.shape) for b in self.biases] 
        del_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        delta = (activations[-1] - y) * (self.sigmoid(z)*(1-self.sigmoid(z))) # cost_derivative * sigmoid_prime
        del_b[-1] = delta
        del_w[-1] = np.matmul(delta, activations[-2].transpose())

        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            sp = self.sigmoid(z)*(1-self.sigmoid(z)) #sigmoid_prime of z
            delta = np.matmul(self.weights[-l+1].transpose(), delta) * sp
            del_b[-l] = delta
            del_w[-l] = np.matmul(delta, activations[-l-1].transpose())
        return (del_b, del_w)


    def print_accuracy(self, images, labels):
        predictions = self.feedforward(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a,b in zip(predictions,labels)])
        print ('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), (num_correct/len(images))*100))

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

