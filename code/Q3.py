# library
# standard library

import matplotlib.pyplot as plt
import numpy   as np
import pandas as pd
# third-party library
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# load data

# THRESHOLDED DATA

# trainXPath = "../data/treshold/train_x.csv"
# trainYPath = "../data/treshold/train_y.csv"
trainXPath = "../data/thresholded/train_xog.csv"
trainYPath = "../data/thresholded/train_yog.csv"
validXPath = "../data/thresholded/valid_x.csv"
validYPath = "../data/thresholded/valid_y.csv"
testXPath = "../data/thresholded/test_x.csv"

# BIGGEST NUMBER DATA
# trainXPath = "../data/aug_big_mnist/train_x.csv"
# trainYPath = "../data/aug_big_mnist/train_y.csv"
# validXPath = "../data/aug_big_mnist/valid_x.csv"
# validYPath = "../data/aug_big_mnist/valid_y.csv"
# testXPath = "../data/aug_big_mnist/test_x.csv"

# ORIGINAL

# trainXPath = "../data/train_valid/train_x.csv"
# trainYPath = "../data/train_valid/train_y.csv"
# validXPath = "../data/train_valid/valid_x.csv"
# validYPath = "../data/train_valid/valid_y.csv"
# testXPath = "../data/test_x.csv"


dtype = torch.cuda.FloatTensor


# dtype =  torch.FloatTensor
#dataset for the training dataset
class kaggleDataset(Dataset):
    def __init__(self, csv_pathX, csv_pathY, transforms=None):
        self.x_data = pd.read_csv(csv_pathX, header=None)
        self.y_data = pd.read_csv(csv_pathY, header=None).as_matrix()
        self.transforms = transforms

    def __getitem__(self, index):
        # label = np.zeros((10))
        # label[self.y_data[index][0]] = 1
        # singleLable = torch.from_numpy(label).type(dtype)

        singleLable = torch.from_numpy(self.y_data[index]).type(torch.FloatTensor)
        singleX = np.asarray(self.x_data.iloc[index]).reshape(1, 64, 64)
        x_tensor = torch.from_numpy(singleX).type(dtype)
        return x_tensor, singleLable

    def __len__(self):
        return len(self.x_data.index)

#dataset for the valid dataset
class kaggleDatasetNoReshape(Dataset):
    def __init__(self, csv_pathX, csv_pathY, transforms=None):
        self.x_data = pd.read_csv(csv_pathX, header=None)
        self.y_data = pd.read_csv(csv_pathY, header=None).as_matrix()
        self.transforms = transforms

    def __getitem__(self, index):
        # label = np.zeros((10))
        # label[self.y_data[index][0]] = 1
        # singleLable = torch.from_numpy(label).type(dtype)

        singleLable = torch.from_numpy(self.y_data[index]).type(torch.FloatTensor)
        singleX = np.asarray(self.x_data.iloc[index])
        x_tensor = torch.from_numpy(singleX).type(torch.FloatTensor)
        return x_tensor, singleLable

    def __len__(self):
        return len(self.x_data.index)

#dataset for test set
class testDataset(Dataset):
    def __init__(self, csv_pathX, transforms=None):
        self.x_data = pd.read_csv(csv_pathX, header=None)
        self.transforms = transforms

    def __getitem__(self, index):
        singleX = np.asarray(self.x_data.iloc[index]).reshape(1, 64, 64)
        x_tensor = torch.from_numpy(singleX).type(dtype)
        return x_tensor

    def __len__(self):
        return len(self.x_data.index)


# Hyper Parameters
EPOCH = 60
BATCH_SIZE = 300
LR = 0.0001  # learning rate

#the network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                
            ),  

            nn.ReLU(),  # activation
            nn.BatchNorm2d(64),
            
        )
        self.conv2 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(64),
            nn.MaxPool2d(  # reduce the size
                kernel_size=2,  # F
                stride=2  # W = (W-F)/S+1
            ),  
           
        )
        self.conv3 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=128,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.25),
        )
        self.conv4 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=128,  # input height
                out_channels=128,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(128),
            nn.MaxPool2d(  # reduce the size
                kernel_size=2,  # F
                stride=2  # W = (W-F)/S+1
            ),  
        
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # input height
                out_channels=256,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  
            nn.BatchNorm2d(256),
            nn.Dropout(0.25)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,  # input height
                out_channels=256,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(256),
            nn.MaxPool2d(  # reduce the size
                kernel_size=2,  # F
                stride=2  # W = (W-F)/S+1
            ), 

        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Dropout2d(p=0.25)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(p=0.25)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.fullConnect = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )


        self.out = nn.Sequential(
            nn.Linear(1024*2*2, 10),
        )

    def forward(self, x):
        x = x.type(dtype).double()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 16 * 16)
        x = self.fullConnect(x)
        #x = self.linear2(x)
        output = self.out(x)
        return output


#function to visualize the confusion matrix
def plot_confusionMatrix(y_tar, y_pred, title='Confusion matrix'):
    cm = confusion_matrix(y_tar, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#function to show image
def imgShower(data, target, numberOfExample):
    data = data.numpy()
    target = target.numpy()
    print(data.shape, target.shape)
    for i in range(numberOfExample):
        plt.title('Label is {label}'.format(label=target[i]))
        plt.imshow(data[i], cmap='gray')
        plt.show()

#function to train cnn with a changing learn rate, it will train as many epoch as the patience (number) then divide the learning rate by 10 and train again
#for every epoch the trained model will be saved
def trainCNN(EPOCH, trainXPath, trainYPath, patience=8):
    print('Loading dataset')
    trainData = kaggleDataset(trainXPath, trainYPath)
    train_loader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE,
                              shuffle=True)  # , num_workers=1,pin_memory=True)
    cnn = CNN().cuda()
    print(cnn)
    cnn.double()
    cnn.train()
    LR = 0.03
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    for epoch in range(EPOCH):
        # load saved model
        # model = torch.load('cnnModelF5Pool2F5Pool2')
        if (epoch % patience == 0 and epoch != 0):
            LR = LR / 10
            print('LR changed to ' + str(LR))
        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
        for batch_idx, (data, target) in enumerate(train_loader):

            # imgShower(data,target)
            target = target.numpy()
            target = np.transpose(target)[0]
            data, target = Variable(data.type(dtype)), Variable(torch.from_numpy(target).type(dtype).long())
            output = cnn(data)  # cnn output
            # print(output.shape,target.shape)
            loss = loss_func(output, target)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           (batch_idx + 1) / len(train_loader), loss.data[0]))

        if epoch % 1 == 0:
            torch.save(cnn, 'models/cnnModelGrantFinal')
            testCNNResult('models/cnnModelVINCENT_8L_augthresh', validXPath, validYPath)
    state = {
        'epoch': EPOCH,
        'state_dict': cnn.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'models/GrantFinal')
    torch.save(cnn, 'models/cnnModelGrantFinal')

#continue to train a saved cnn model with a changing learn rate, it will train as many epoch as the patience (number) then divide the learning rate by 10 and train again
#for every epoch the trained model will be saved 
def continueTrainCNN(EPOCH, trainXPath, trainYPath, modelpath,patience = 8):
    trainData = kaggleDataset(trainXPath, trainYPath)
    train_loader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE,
                              shuffle=True)  # , num_workers=1,pin_memory=True)
    model = torch.load(modelpath)
    model.cuda()
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MultiLabelSoftMarginLoss()
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    LR = 0.0001
    for epoch in range(EPOCH):
        if (epoch % patience == 0 and epoch != 0):
            LR = LR / 10
	    print('LR changed to '+str(LR))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # load saved model
        # model = torch.load('cnnModelF5Pool2F5Pool2')

        for batch_idx, (data, target) in enumerate(train_loader):

            # imgShower(data,target)
            target = target.numpy()
            target = np.transpose(target)[0]
            data, target = Variable(data.type(dtype)), Variable(torch.from_numpy(target).type(dtype).long())
            output = model(data)  # cnn output
            # print(output.shape,target.shape)
            loss = loss_func(output, target)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           (batch_idx + 1) / len(train_loader), loss.data[0]))
        if epoch % 1 == 0:
            torch.save(model, modelpath)
            testCNNResult(modelpath, validXPath, validYPath)

    torch.save(model, modelpath)


def separateTrainValid():
    trainData = kaggleDatasetNoReshape(trainXPath, trainYPath)
    train_loader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.numpy()
        print(data.shape)
        target = target.numpy()
        print(target.shape)
        dfx = pd.DataFrame(data)
        dfy = pd.DataFrame(target)
        if (batch_idx < 20):
            with open('fastTrain_x_1.csv', 'a') as f:
                dfx.to_csv(f, index=False, header=False)
            dfy = pd.DataFrame(target)
            with open('fastTrain_y_1.csv', 'a') as f:
                dfy.to_csv(f, index=False, header=False)
        else:
            with open('train_x_1.csv', 'a') as f:
                dfx.to_csv(f, index=False, header=False)

            with open('train_y_1.csv', 'a') as f:
                dfy.to_csv(f, index=False, header=False)

#this function take into a saved trained cnn model, evaluate the test set and save the result in a csv file
def testCNN(modelName):
    testData = testDataset(testXPath)
    test_loader = DataLoader(dataset=testData, batch_size=50, shuffle=False)  # , num_workers=1,pin_memory=True)
    result = 0
    model = torch.load(modelName)
	model.cuda()
    model.eval()
    for batch_idx, data in enumerate(test_loader):
        data = Variable(data.type(dtype))

        output = model(data)
        pred = torch.max(output.cpu(), 1)[1].data.numpy()
        if batch_idx < 1:
            result = pred
        else:
            result = np.append(result, pred)
        print(len(result))
    df = pd.DataFrame(np.transpose(result.reshape(1, -1)))
    df.to_csv("submissions/test_y_vincent_8L_augthresh.csv", index_label='Id', header=['Label'])

#this function evaluate the validation set and produce the accuracy score and plot the confusian matrix
def testCNNResult(modelName, ValidX, ValidY, plotCM = False):
    testData = kaggleDataset(ValidX, ValidY)
    test_loader = DataLoader(dataset=testData, batch_size=15, shuffle=False)  # , num_workers=1,pin_memory=True)
    result = 0
    trueRes = 0
    model = torch.load(modelName)
    model.cuda()
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.type(dtype))
        target = np.transpose(target.cpu().numpy())[0]

        output = model(data)
        pred = torch.max(output.cpu(), 1)[1].data.numpy()
        if batch_idx < 1:
            result = pred
            trueRes = target
        else:
            result = np.append(result, pred)
            trueRes = np.append(trueRes, target)
        # if batch_idx%20 == 0:
        #    print(len(result))
        #    print(len(trueRes))
        #    print(accuracy_score(trueRes, result))

    print('final accuracy')
    print(accuracy_score(trueRes, result))
    if plotCM == True:
	confusion_matrix(trueRes,result) 


if __name__ == '__main__':
    # testCNN('cnnModelF3F3F5new1')
    # trainCNN(EPOCH, trainXPath, trainYPath,8)
    # testCNN('models/cnnModelVINCENT_8L_augthresh')
    # separateTrainValid()
    # testCNNResult('cnnModelGrant256',validXPath, validYPath)
    continueTrainCNN(EPOCH,trainXPath,trainYPath,'models/cnnModelGrantFinal')
    
    plot_confusionMatrix(y1, y2)
