import numpy as np
import pandas as pd

import pickle
import os
import shutil

WEIGHTS_DIR = 'nn_weights/'

# simple feed-forward neural network
class NeuralNetwork(object):
    def __init__(self, layers=[], activations=[], learning_rate=0.01, epochs=500, name='noname'):

        #hyperparameter
        self.layers = layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        #initialize the model
        self.weights, self.biases = self._init_weight()

        self.name = name

    def fit(self, X_train, Y_train, X_valid, Y_valid, minibatch_size=50, verbose=True, save_step=100):
        valid_costs = np.zeros(self.epochs)
        train_costs = np.zeros(self.epochs)
        num_minibatch = int(np.ceil(X_train.shape[0] / minibatch_size))

        for i in range(self.epochs):
            for j in range(num_minibatch):
                start_idx = j * minibatch_size
                end_idx = start_idx + minibatch_size
                if j == num_minibatch - 1:
                    end_idx = X_train.shape[0]

                batch_X = X_train[start_idx:end_idx]
                batch_Y = Y_train[start_idx:end_idx]

                activations = self._forward_prop(batch_X)
                dW, db = self._back_prop(batch_Y, activations)
                self._sgd_optimizer(dW, db)

                if verbose:
                    print('Batch {:5d} / {:5d}\r'.format(j+1, num_minibatch), end='')
        
            
            train_costs[i] = self.cost(X_train, Y_train)
            valid_costs[i] = self.cost(X_valid, Y_valid)
            
            valid_acc = self.accuracy(X_valid, Y_valid)

            if verbose:
                print('\nEpoch {:5d} / {:5d}: Loss->{:.6f}(validation) {:.6f}(training) Accuracy->{:.6f}'.format(i+1, self.epochs, valid_costs[i], train_costs[i], valid_acc))
            
            if i % save_step == 0:
                self.save()

        return train_costs, valid_costs

    def predict(self, X):
        Y_hat = self._forward_prop(X)[-1][0]
        return (Y_hat == Y_hat.max(axis=1)[:, None]).astype(int)

    def accuracy(self, X, Y):
        preds = self.predict(X)
        return (preds == Y).mean()

    def cost(self, X, Y):
        Y_hat = self._forward_prop(X)[-1][0]
        return - np.multiply(Y, np.log(Y_hat)).sum() / Y.shape[0]

    def _forward_prop(self, X):
        A = X
        activations = []
        activations.append((X, None))

        #hidden layers
        for l in range(len(self.layers)-2):
            Z = np.dot(A, self.weights[l]) + self.biases[l]
            A = activation_dict[self.activations[l]][0](Z)
            A_deriv = activation_dict[self.activations[l]][1](Z)
            activations.append((A[:, :], A_deriv[:, :]))
        
        #output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        Y_hat = softmax(Z)
        activations.append((Y_hat, None))
        return activations

    def _back_prop(self, Y, activations):
        grads_w = []
        grads_b = []
        
        # output gradients
        Y_hat = activations[-1][0]
        dA = (Y_hat - Y) / Y.shape[0]
        
        for l in range(len(self.layers)-1)[::-1]:
            dW = np.dot(activations[l][0].T, dA)
            db = np.sum(dA, axis=0)
            grads_w.append(dW)
            grads_b.append(db)
            
            if l != 0:
                dA = activations[l][1] * np.dot(dA, self.weights[l].T)

        return grads_w[::-1], grads_b[::-1]
    
    def _sgd_optimizer(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def _init_weight(self):
        weights = []
        biases = []
        for l in range(len(self.layers)-1):
            w = 0.1*np.random.randn(self.layers[l], self.layers[l+1])
            b = np.random.randn(self.layers[l+1])
            weights.append(w)
            biases.append(b)

        return weights, biases
    
    def save(self):
        model_path = WEIGHTS_DIR + self.name + '/'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)

        os.mkdir(model_path)
        
        # save parameters
        for i in range(len(self.weights)):
            pd.DataFrame(self.weights[i]).to_csv(model_path + 'weights{}.csv'.format(i), header=None, index=False)
            pd.DataFrame(self.biases[i]).to_csv(model_path + 'biases{}.csv'.format(i), header=None, index=False)
        
        # save hyperparameter
        hp_dict = {
            'layers': self.layers,
            'activations': self.activations,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'name': self.name
        }
        
        with open(model_path+'hp.pickle', 'wb') as fh:
            pickle.dump(hp_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(name):
        # check the model exists
        model_path = WEIGHTS_DIR + name + '/'
        if not os.path.isdir(model_path):
            return None
        
        hp_dict = {}
        with open(model_path+'hp.pickle', 'rb') as fh:
            hp_dict = pickle.load(fh)

        nn = NeuralNetwork(
            layers=hp_dict['layers'],
            activations=hp_dict['activations'],
            learning_rate=hp_dict['learning_rate'],
            epochs=hp_dict['epochs'],
            name=hp_dict['name'])

        for i in range(len(nn.layers)-1):
            w = pd.read_csv(model_path+'weights{}.csv'.format(i), header=None).values
            b = pd.read_csv(model_path+'biases{}.csv'.format(i), header=None).values
            nn.weights[i] = w[:]
            nn.biases[i] = b.reshape(b.shape[0])

        return nn

    def print(self):
        print('Layers: {}'.format(self.layers))
        print('Activations: {}'.format(self.activations))
        print('Learning_rate: {}'.format(self.learning_rate))
        print('Epochs: {}'.format(self.epochs))
        print('Weight + Biases')
        for i in range(len(self.weights)):
            print('w{}: {}'.format(i, self.weights[i].shape))
            print('b{}: {}'.format(i, self.biases[i].shape))

# ======= output =====================
def softmax(z):
    ex = np.exp(z - np.max(z))
    return ex / np.sum(ex, axis=1, keepdims=True)


# ======= activation functions =======
def sigmoid(z):
    return 1. / (1. + np.exp(-1.*z))
def sigmoid_derivative(z):
    return sigmoid(z) * (1.-sigmoid(z))

def tanh(z):
    return np.tanh(z)
def tanh_derivative(z):
    return 1. - np.power(tanh(z), 2)

def relu(z):
    return np.maximum(z, 0.)
def relu_derivative(z):
    return (z>0.) * 1.0


activation_dict = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative)
}
