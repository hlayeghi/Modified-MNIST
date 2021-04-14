import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data import load_dataset, save_array

FIG_PATH = '../report/figures/'
npoints =5

def showCM(cm,methodname):
    # Show confusion matrix in a separate window
    fig1 = plt.figure(1)
    plt.matshow(cm)
    plt.title('Confusion matrix for '+methodname)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig1.savefig(FIG_PATH+methodname+'Q1CM.pdf')

if __name__ == '__main__':
    
    print('Loading...')
    x_train, y_train, x_valid, y_valid = load_dataset('big')
    print('Done')

    y_train = np.reshape(y_train, y_train.shape[0])
    y_valid = np.reshape(y_valid, y_valid.shape[0])
        
    x_train = x_train[:15000]
    y_train = y_train[:15000]
    
    
    
    clf = LinearSVC(
        C=1, 
        verbose=0)

    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    
    save_array(preds, 'svm_preds')
    
    
    #results = np.zeros((10, 3))
    #score = np.zeros(npoints)
    #i = 0

    #for dataset in ['big']:
    #    print('Loading...')
    #    x_train, y_train, x_valid, y_valid = load_dataset(dataset)
    #    print('Done')

    #    y_train = np.reshape(y_train, y_train.shape[0])
    #    y_valid = np.reshape(y_valid, y_valid.shape[0])
        
    #    x_train = x_train[:15000]
    #    y_train = y_train[:15000]

    #    Cspan = np.logspace(-3,0,npoints)
    #    for i in range(npoints):
    #        C = Cspan[i]
    #        print('Dataset: {}, C: {}'.format(dataset, C))

    #        clf = LinearSVC(
    #            C=C, 
    #            verbose=0)

    #        clf.fit(x_train, y_train)
        
    #        train_acc = clf.score(x_train, y_train)
    #        valid_acc = clf.score(x_valid, y_valid)
            
    #        print('C: {}'.format(C))
    #        print('training accuracy: {}'.format(train_acc))
    #        print('validation accuracy: {}\n'.format(valid_acc))

    #        results[i,0] = C
    #        results[i,1] = train_acc
    #        results[i,2] = valid_acc
    #        i += 1
            
    #        score[i] = valid_acc
            #cm[i,:,:] = confusion_matrix(y_valid, clf.predict(x_valid))
        
        #imax = np.argmax(score)
        #Cmax = Cspan[imax]
            
    #    print('\n') 
    #save_array(results, 'Q1_res.csv')
    
    
    #showCM(cm[imax,:,:].reshape(10,10),'LinearSVM')
    #np.savetxt(FIG_PATH+'LinearSVMCM.csv', np.array(score)[None,:], delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")
    #np.savetxt(FIG_PATH+'LinearSVMCMbest.csv', np.array(cm[imax,:,:]).reshape(10,10), delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")
    #np.savetxt(FIG_PATH+'LinearSVMCspan.csv', np.array(Cspan)[None,:], delimiter=' & ' ,newline=' \\\\\n',fmt = "%s")

