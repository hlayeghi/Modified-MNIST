import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from data import load_dataset, one_hot, load_array, save_array

# for testing purposes
def xor_data():
    X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
    Y = np.array([[1.,0.], [0.,1.], [0.,1.], [1.,0.]])
    return X, Y, X[:], Y[:]

if __name__ == '__main__':

    #x_train, y_train, x_valid, y_valid = xor_data() 
    
    print('Loading dataset')
    x_train, y_train, x_valid, y_valid = load_dataset('big')
    print('Done loading')
    
    print('One hotting data')
    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)


    #nn = NeuralNetwork.load('big_350')
    #preds = nn.predict(x_valid)
    #save_array(preds, 'big_350_preds')
    #print(nn.accuracy(x_valid, y_valid))

    training_errors = []
    validation_errors = []
    training_acc = []
    validation_acc = []

    for n_hidden in [100, 200, 300, 400]:

        nn = NeuralNetwork(
            layers=[x_train.shape[1], n_hidden, y_train.shape[1]], 
            activations=['tanh'], 
            learning_rate=0.03,
            epochs=15,
            name='big_{}'.format(n_hidden))

        training_costs, validation_costs = nn.fit(
            x_train, 
            y_train, 
            x_valid, 
            y_valid, 
            minibatch_size=100, 
            verbose=True,
            save_step=1)
    
        # plot learning curve
        plt.plot(training_costs, color='red', label='training error')
        plt.plot(validation_costs, color='blue', label='validation error')
        plt.title('Learning curve for one hidden layer with {} neurons'.format(n_hidden))
        plt.legend()
        plt.show()
    
        print(training_costs)
        print(validation_costs)
        
        training_errors.append(training_costs[-1])
        validation_errors.append(validation_costs[-1])
    
        training_acc.append(nn.accuracy(x_train, y_train))
        validation_acc.append(nn.accuracy(x_valid, y_valid))


    # plot training-validation errors 
    plt.plot([100, 200, 300, 400], training_errors, color='red', label='training error')
    plt.plot([100, 200, 300, 400], validation_errors, color='blue', label='validation error')
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

     
    # plot training-validation accuracy
    plt.plot([100, 200, 300, 400], training_acc, color='red', label='training')
    plt.plot([100, 200, 300, 400], validation_acc, color='blue', label='validation')
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
