import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil

from sklearn.preprocessing import OneHotEncoder

from skimage.measure import label, regionprops
from skimage.morphology import disk,closing,square,erosion
from skimage.filters.rank import median
from skimage.transform import rotate

# paths
DATA_PATH = '../data/'
DATASET_DICT = {
    'download': DATA_PATH + 'download/',
    'og': DATA_PATH + 'og/',
    'threshold': DATA_PATH + 'threshold/',
    'big': DATA_PATH + 'big/',
    'aug_threshold': DATA_PATH + 'aug_threshold/',
    'aug_big': DATA_PATH + 'aug_big/'
}
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# toggle this flag to save or not the preprocessed files
SAVE_ALL = False

# file names
TRAIN_X = 'train_x.csv'
TRAIN_Y = 'train_y.csv'
VALID_X = 'valid_x.csv'
VALID_Y = 'valid_y.csv'
TEST_X = 'test_x.csv'

TRAIN_PERCENT = 0.95

# threshold value
THRESHOLD = 230

# Biggest Number
MERGED = 0
LENGTHMAX = 30 # Expected MNIST image sizes
DISKSIZE = 1 # for median filter
OUTPUTSIZE = 32 # Size of data

# Figures Params
FIG_PATH = '../report/figures/'
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)

#==============================
# Save and Load arrays
def load_array(fname):
    a = pd.read_csv(fname, header=None).values
    return a

def save_array(array, fname):
    pd.DataFrame(array).to_csv(fname, header=None, index=False)

def load_dataset(name):
    dataset_path = DATASET_DICT[name]
    
    # check if the dataset already exists
    if not os.path.isdir(dataset_path):
        if name == 'og':
            return download_og()
        else:
            return generate_dataset(name)
    
    else:
        train_x = load_array(dataset_path + TRAIN_X)
        train_y = load_array(dataset_path + TRAIN_Y)
        valid_x = load_array(dataset_path + VALID_X)
        valid_y = load_array(dataset_path + VALID_Y)
        return train_x, train_y, valid_x, valid_y


#==============================
# Generating datasets
def generate_dataset(name):
    if name == 'threshold':
        return generate_threshold_dataset()
    elif name == 'big':
        return generate_big_dataset()
    
    elif name == 'aug_threshold':
        tx,ty,vx,vy = load_dataset('threshold')
        return augment(tx,ty,vx,vy)
    elif name == 'aug_big':
        tx,ty,vx,vy = load_dataset('big')
        return augment(tx,ty,vx,vy)


#==============================
# Download original dataset
def download_og():
    print('Downloading original dataset')
    
    #TODO
    # change to download
    x = load_array(DATASET_DICT['download'] + TRAIN_X)
    y = load_array(DATASET_DICT['download'] + TRAIN_Y)
    test_x = load_array(DATASET_DICT['download'] + TEST_X)

    train_x, train_y, valid_x, valid_y = train_valid_split(x,y)
    
    # save them locally
    dataset_path = DATASET_DICT['og']
    os.mkdir(dataset_path)

    save_array(train_x, dataset_path + TRAIN_X)
    save_array(train_y, dataset_path + TRAIN_Y)
    save_array(valid_x, dataset_path + VALID_X)
    save_array(valid_y, dataset_path + VALID_Y)
    save_array(test_x, dataset_path + TEST_X)
    
    print('Finished downloading')

    return train_x, train_y, valid_x, valid_y

# Split the og dataset into TRAIN/VALIDATION
def train_valid_split(x, y):
    # shuffle data
    rdm_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rdm_state)
    np.random.shuffle(y)

    num_train = int(TRAIN_PERCENT*x.shape[0])
    
    # split
    train_x = x[:num_train, :]
    train_y = y[:num_train, :]
    valid_x = x[num_train:, :]
    valid_y = y[num_train:, :]
    
    return train_x, train_y, valid_x, valid_y



#==============================
# Thresholding
def threshold_filter(images):
    return (images > THRESHOLD).astype(float)

def medianfilter(image,r):
        image = image.reshape((64,64))
        return median(image, disk(r))   

def generate_threshold_dataset():
    print('Generating the threshold dataset')
    og_train_x, og_train_y, og_valid_x, og_valid_y = load_dataset('og')
    og_test_x = load_array(DATASET_DICT['og'] + TEST_X)
    
    th_train_x = np.zeros((og_train_x.shape[0],64**2),dtype = float)
    th_valid_x = np.zeros((og_valid_x.shape[0],64**2),dtype = float)
    th_test_x = np.zeros((og_test_x.shape[0],64**2),dtype = float)
    
    # threshold first
    og_train_x = threshold_filter(og_train_x)
    og_valid_x = threshold_filter(og_valid_x)
    og_test_x = threshold_filter(og_test_x)
    
    # median second
    for i in range(og_train_x.shape[0]):
        th_train_x[i,:] = medianfilter(og_train_x[i],DISKSIZE).flatten()
    for i in range(og_valid_x.shape[0]):
        th_valid_x[i,:] = medianfilter(og_valid_x[i],DISKSIZE).flatten()
    for i in range(og_test_x.shape[0]):
        th_test_x[i,:] = medianfilter(og_test_x[i],DISKSIZE).flatten()    
    
    if SAVE_ALL:
        print('Saving dataset')
        dataset_path = DATASET_DICT['threshold']
        shutil.rmtree(dataset_path)
        os.mkdir(dataset_path)
        
        save_array(th_train_x, dataset_path + TRAIN_X)
        save_array(train_y, dataset_path + TRAIN_Y)
        save_array(th_valid_x, dataset_path + VALID_X)
        save_array(valid_y, dataset_path + VALID_Y)
        save_array(th_test_x, dataset_path + TEST_X)
    
    print('Done generating dataset')
    return th_train_x, og_train_y, th_valid_x, og_valid_y


# ============================
# Big dataset
def normalizeIm(im,size = OUTPUTSIZE):
    im = im.astype(int)
    isize = im.shape
    hs = int(size/2) #half size for centering
    w = int(isize[0]/2)
    h = int(isize[1]/2)
    if isize[0]>size:
        im = im[w-hs:w+hs,:]
        isize = im.shape
        w = int(isize[0]/2)
        
    if isize[1]>size:
        im = im[:,h-hs:h+hs]
        isize = im.shape
        h = int(isize[1]/2)
            
    rw = int(isize[0] %2)
    rh = int(isize[1] %2)
    
    out = np.zeros((size,size),dtype=int)
    out[hs-w:hs+w+rw,hs-h:hs+h+rh] = im
    
    return out  

def bigSegment(imageref,imageseg,ignore = False):
    
    bw = closing(imageseg, square(1))    
    label_image = label(bw)
    amax = -1
    wmax = -1
    for region in regionprops(label_image):
        r0, c0, r1, c1 = region.bbox
        length = np.max([r1-r0,c1-c0])
        width = np.min([r1-r0,c1-c0])
        if length<=LENGTHMAX or ignore:
            a = length**2
            segment = imageref[r0:r1,c0:c1]
            if a>amax or (a== amax and width>wmax):
                outIm = segment
                amax = a
                wmax = width            
        else:
            raise Exception('Merged Numbers')
    return outIm

def biggest(imth,erode = 0):
    if erode == 0:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)))
    elif erode == -1:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)),ignore = True)
    else:
        im = bigSegment(imth,erosion(imth,disk(erode)))  
    return im

def findBiggest(imth):
    #imth = imth.reshape(64,64)
    immed = medianfilter(imth,1)
    try:
        out = biggest(immed)
    except:
        try:
            out = biggest(immed,erode =1)
        except:
            #try:
                #bigNumImA,bigNumImaAS,bigNumImP,imageth= preprocess(image,erode =2)
            #except:
            out = biggest(immed,erode = -1)
    return normalizeIm(out)


def generate_big_dataset():
    print('Generating the big dataset')
    og_train_x, og_train_y, og_valid_x, og_valid_y = load_dataset('og')
    og_test_x = load_array(DATASET_DICT['og'] + TEST_X)
    
    big_train_x = np.zeros((og_train_x.shape[0],OUTPUTSIZE**2),dtype = int)
    big_valid_x = np.zeros((og_valid_x.shape[0],OUTPUTSIZE**2),dtype = int)
    big_test_x = np.zeros((og_test_x.shape[0],OUTPUTSIZE**2),dtype = int)
    
    # threshold first
    og_train_x = threshold_filter(og_train_x)
    og_valid_x = threshold_filter(og_valid_x)
    og_test_x = threshold_filter(og_test_x)
    
    # median second
    for i in range(og_train_x.shape[0]):
        big_train_x[i,:] = findBiggest(og_train_x[i].reshape((64,64))).flatten()
    for i in range(og_valid_x.shape[0]):
        big_valid_x[i,:] = findBiggest(og_valid_x[i].reshape((64,64))).flatten()
    for i in range(og_test_x.shape[0]):
        big_test_x[i,:] = findBiggest(og_test_x[i].reshape((64,64))).flatten()    
    
    if SAVE_ALL:
        print('Saving dataset')
        dataset_path = DATASET_DICT['big']
        shutil.rmtree(dataset_path)
        os.mkdir(dataset_path)
        
        save_array(big_train_x, dataset_path + TRAIN_X)
        save_array(train_y, dataset_path + TRAIN_Y)
        save_array(big_valid_x, dataset_path + VALID_X)
        save_array(valid_y, dataset_path + VALID_Y)
        save_array(big_test_x, dataset_path + TEST_X)
    
    print('Done generating dataset')
    return big_train_x, og_train_y, big_valid_x, og_valid_y


# =============================
# rotate img
def rotate_img(img):
    img_size = int(np.sqrt(img.shape[0]))
    rots = []
    rots.append(rotate(img.reshape((img_size, img_size)), 27, preserve_range=True))
    rots.append(rotate(img.reshape((img_size, img_size)), -27, preserve_range=True))

    return [(r > 0.75).astype(int).reshape((1, img_size**2)) for r in rots]

# Augment dataset
def augment(train_x, train_y, valid_x, valid_y):
    old_size = train_x.shape[0]
    new_size = old_size * 3
    
    aug_x = np.zeros((new_size, train_x.shape[1]))
    aug_y = np.zeros((new_size, 1))

    for i in range(old_size):
        start_idx = 3*i
        label = train_y[i]
        rotations = rotate_img(train_x[i])
        
        aug_x[start_idx] = train_x[i, :]
        aug_x[start_idx + 1] = rotations[0]
        aug_x[start_idx + 2] = rotations[1]

        aug_y[start_idx] = label
        aug_y[start_idx + 1] = label
        aug_y[start_idx + 2] = label

    state = np.random.get_state()
    np.random.shuffle(aug_x)
    np.random.set_state(state)
    np.random.shuffle(aug_y)

    return aug_x, aug_y, valid_x, valid_y



# =============================
# Visualization
def showDataset(fnamex,fnamey,datasetname,nrows=10,ncols =10,indices = []):
    print('Showing '+datasetname+' dataset:')
    data = load_array(fnamex)
    labels = load_array(fnamey)
    dim = int(np.sqrt(data.shape[1]))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.3, nrows*1.4),
                           sharex=True, sharey=True)
    if indices ==[]:
        indices = np.random.randint(0,high=data.shape[0],size=nrows*ncols)
    for i in range(nrows*ncols):
        index = indices[i]
        image = data[index,:].reshape(dim,dim)
        x = i//ncols
        y = i%ncols
        ax[x,y].imshow(image,cmap=plt.cm.gray)
        ax[x,y].axis('off')
        ax[x,y].set_title(format(labels[index]))
    fig.tight_layout()
    print(fnamex)
    fig.savefig(FIG_PATH+datasetname+'Dataset.pdf')
    return indices

def showallFilters(nrows=10,ncols=10,indices=[]):
    print('Loading original dataset:')
    data = load_array(DATA_PATH+'og/train_x.csv')
    labels = load_array(DATA_PATH+BIGGEST_DIR+'train_y.csv')
    dim = int(np.sqrt(data.shape[1]))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.3, nrows*1.4),
                           sharex=True, sharey=True)
    figth, axth = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.3, nrows*1.4),
                           sharex=True, sharey=True)
    figthmed, axthmed = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.3, nrows*1.4),
                           sharex=True, sharey=True)
    figbig, axbig = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.3, nrows*1.4),
                           sharex=True, sharey=True)
    if indices ==[]:
        indices = np.random.randint(0,high=data.shape[0],size=nrows*ncols)
    for i in range(nrows*ncols):
        index = indices[i]
        image = data[index,:].reshape(dim,dim)
        x = i//ncols
        y = i%ncols
        ax[x,y].imshow(image,cmap=plt.cm.gray)
        ax[x,y].axis('off')
        ax[x,y].set_title(format(labels[index]))
        
        axth[x,y].imshow(threshold_filter(image),cmap=plt.cm.gray)
        axth[x,y].axis('off')
        axth[x,y].set_title(format(labels[index]))
        
        axthmed[x,y].imshow(medianfilter(threshold_filter(image),1),cmap=plt.cm.gray)
        axthmed[x,y].axis('off')
        axthmed[x,y].set_title(format(labels[index]))
        
        axbig[x,y].imshow(findBiggest(threshold_filter(image)),cmap=plt.cm.gray)
        axbig[x,y].axis('off')
        axbig[x,y].set_title(format(labels[index]))
    fig.tight_layout()
    figth.tight_layout()
    figthmed.tight_layout()
    figbig.tight_layout()
    
    fig.savefig(FIG_PATH+'originalDataset.pdf')
    figth.savefig(FIG_PATH+'thresholdDataset.pdf')
    figthmed.savefig(FIG_PATH+'thresholdmedDataset.pdf')
    figbig.savefig(FIG_PATH+'biggestDataset.pdf')


    return indices
        


def show_image(img_as_array, dim=64):
    plt.imshow(img_as_array.reshape((dim, dim)), cmap='gray')
    plt.show()

def show_random_image(imgs_as_array, labels, dim=64):
    rdm_idx = np.random.randint(imgs_as_array.shape[0])
    print('Label: {}'.format(labels[rdm_idx]))
    show_image(imgs_as_array[rdm_idx], dim=dim)
    
    
#==============================
def one_hot(arr):
    enc = OneHotEncoder()
    return enc.fit_transform(arr).toarray()



if __name__ == '__main__':
    
    x,y,vx,vy = load_dataset('aug_threshold')
    print(x.shape)
    print(y.shape)
    print(vx.shape)
    print(vy.shape)
    
    for i in range(5):
        show_random_image(x, y, dim=64)
    for i in range(5):
        show_random_image(vx, vy, dim=64)



    #showDataset(DATA_PATH+BIGGEST_DIR+'train_x.csv',DATA_PATH+BIGGEST_DIR+'train_y.csv','biggest')
    #showDataset(DATA_PATH+THRESHOLD_DIR+'train_x.csv',DATA_PATH+THRESHOLD_DIR+'train_y.csv','threshold')
    #showDataset(DATA_PATH+THRESHOLDMED_DIR+'train_x.csv',DATA_PATH+THRESHOLDMED_DIR+'train_y.csv','thresholdmed')
