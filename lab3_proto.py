import numpy as np
import os
import random as rand

import pickle

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from prondict import prondict

from lab3_tools import *
from lab2_proto import *

from lab2_proto import *
from lab1_proto import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """

    phonemelist = []

    for word in wordList:
        for phoneme in prondict[word]:
            phonemelist.append(phoneme)

        if(addShortPause):
            phonemelist.append("sp")

    if(addSilence):
        phonemelist.insert(0, "sil")
        phonemelist.append("sil")

    return phonemelist




def forcedAlignment(utteranceHMM, lmfcc, stateTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

    lmndd = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])

    viterbimax, viterbipath = viterbi(lmndd, np.log(utteranceHMM['startprob'][:-1]), np.log(utteranceHMM['transmat'][:-1, :-1]))

    symbol_sequence = [stateTrans[i] for i in viterbipath]

    transcription = frames2trans(symbol_sequence,'z43a.lab')

    return symbol_sequence


def saveFiles(datatype, discnr, phoneHMMs, stateList):
    data = []
    for root, dirs, files in os.walk('tidigits/disc_4.' + str(discnr) + '.1/tidigits/' + str(datatype)):
        for idx, file in enumerate(files):
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                samplelmfcc = mfcc(samples)
                samplemspec = mspec(samples)
                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict)
                utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
                stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

                symbol_sequence = forcedAlignment(utteranceHMM, samplelmfcc, stateTrans)
                targets =  np.array([stateList.index(target) for target in symbol_sequence])

                data.append({'filename': filename, 'lmfcc': samplelmfcc, 'mspec': samplemspec, 'targets': targets})

    np.savez(str(datatype) + 'data' + discnr + '.npz', data=data)



def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """

def splitSet(input):
    np.random.shuffle(input)

    trainingset = []
    validationset = []

    trainingspeakers = []
    validationspeakers = []

    # [0][0] : men training
    # [0][1] : women training
    # [1][0] : men validation
    # [1][1] : women validation
    setdistribution = [[0 for i in range(0,2)] for j in range(0,2)]

    for datapoint in input:
        characteristicarray = datapoint['filename'].split("\\")
        gender = characteristicarray[1]
        speaker = characteristicarray[2]

        if speaker in trainingspeakers:
            trainingset.append(datapoint)
            if gender == 'man' or gender == 'boy':
                setdistribution[0][0] += 1
            elif gender == 'woman' or gender == 'girl':
                setdistribution[0][1] += 1
        elif speaker in validationspeakers:
            validationset.append(datapoint)
            if gender == 'man' or gender == 'boy':
                setdistribution[1][0] += 1
            elif gender == 'woman' or gender == 'girl':
                setdistribution[1][1] += 1
        else:
            decider = rand.uniform(0, 1)

            setratio = 0

            if (gender == 'man'  or gender == 'boy') and (setdistribution[1][0] + setdistribution[0][0] != 0):
                setratio = 0.1 - setdistribution[1][0] / (setdistribution[1][0] + setdistribution[0][0])
            elif (gender == 'woman'  or gender == 'girl') and (setdistribution[1][1] + setdistribution[0][1] != 0):
                setratio = 0.1 - setdistribution[1][1] / (setdistribution[1][1] + setdistribution[0][1])

            decider += setratio

            if decider > 0.9:
                validationset.append(datapoint)

                validationspeakers.append(speaker)
                if gender == 'man' or gender == 'boy':
                    setdistribution[1][0] += 1
                elif gender == 'woman' or gender == 'girl':
                    setdistribution[1][1] += 1
            else:
                trainingset.append(datapoint)
                trainingspeakers.append(speaker)
                if gender == 'man' or gender == 'boy':
                    setdistribution[0][0] += 1
                elif gender == 'woman' or gender == 'girl':
                    setdistribution[0][1] += 1

    print("male training: " + str(setdistribution[0][0]))
    print("female training: " + str(setdistribution[0][1]))
    print("male validation: " + str(setdistribution[1][0]))
    print("female validation: " + str(setdistribution[1][1]))
    print("validation percentage: " + str(100*(setdistribution[1][0]+setdistribution[1][0])/len(input)))
    print("female percentage: " + str(100*(setdistribution[0][1]+setdistribution[1][1])/len(input)))

    np.savez('splitdata.npz', trainingdata=trainingset, validationdata=validationset)

def dynamic_features(feature):
    N = feature.shape[0]
    M = feature.shape[1] * 7
    dyn = np.zeros((N, M))

    maxidx = len(feature) - 1
    for idx, row in enumerate(feature):
        idxarray = np.abs(np.arange(idx-3, idx+4))
        idxarray = np.where(idxarray >= maxidx, maxidx - idxarray % maxidx, idxarray)
        dyn[idx, :] = feature[tuple([idxarray])].flatten()

    return dyn

def standardize(dataset, feature, type, hasDynamicFeatures, scaler = None, saveAs = None):
    print("Standardize")

    if type == 'all':
        C = 0

        sizes = np.zeros(len(dataset), dtype="int32")
        for i in range(len(dataset)):
            sizes[i] = dataset[i][feature].shape[0]

        C = np.sum(sizes, dtype="int32")
        if feature == 'targets':
            data = np.zeros((C, 1), dtype=np.float32)
        else:
            N =  dataset[0][feature].shape[1]
            data = np.zeros((C, N + (N*6*hasDynamicFeatures)), dtype=np.float32)

        start_idx = 0
        for i in range(len(dataset)):
            if hasDynamicFeatures and feature != 'targets': # If not targets and has dynamic features
                data[start_idx: start_idx + sizes[i], :] = dynamic_features(dataset[i][feature])
            elif feature == 'targets': # If targets (w/o dynamic features)
                data[start_idx: start_idx + sizes[i], :] = dataset[i][feature].reshape(sizes[i], 1)
            else: # If not targets and not has dynamic features
                data[start_idx: start_idx + sizes[i], :] = dataset[i][feature]
            start_idx += sizes[i]

    elif type == 'speaker':
        pass
    elif type == 'utterance':
        pass

    if scaler != None:
        scaler.fit_transform(data)

    if saveAs != None:
        np.savez('standardized/' + saveAs + '.npz', data=data)


    return data

def plotHistory():
    pickle_in = open("history/trainHistoryDict_3.pickle","rb")
    history = pickle.load(pickle_in)

    print(history.keys())
    # summarize history for accuracy

    plt.subplot(2,1,1)

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.subplot(2,1,2)
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def modelBuilder(stateList, shouldSave):

    hiddenLayers = 3
    epochs = 20

    outputDim = len(stateList)

    lmfcc_train_x_reg = np.load("standardized/lmfcc_train_x_reg.npz", allow_pickle=True)['data'].astype('float32')
    train_y_noncat = np.load("standardized/train_y.npz", allow_pickle=True)['data']
    train_y = np_utils.to_categorical(train_y_noncat, outputDim)

    lmfcc_val_x_reg = np.load("standardized/lmfcc_val_x_reg.npz", allow_pickle=True)['data'].astype('float32')
    val_y_noncat = np.load("standardized/val_y.npz", allow_pickle=True)['data']
    val_y = np_utils.to_categorical(val_y_noncat, outputDim)
    input_dim = lmfcc_train_x_reg.shape[1]

    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))

    for i in range(hiddenLayers - 1):
        model.add(Dense(256, activation='relu'))

    model.add(Dense(outputDim, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(lmfcc_train_x_reg, train_y, epochs=epochs, batch_size=256, validation_data = (lmfcc_val_x_reg, val_y), verbose = 1)

    if shouldSave:
        model.save('model/model1.h5')

        with open("history/trainHistoryDict_" + str(hiddenLayers) + ".pickle", 'wb') as f:
            pickle.dump(history.history, f)

        print("Learned weights are saved in model/model1.h5")


# -- Load model and create stateList -- #
print("Load model and statelist")
phoneHMMs = np.load('lab2_models_v2.npz', allow_pickle=True)['phoneHMMs'].item() # Ska egentligen använda models_all
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

'''

# -- Will create the initial file -- #
saveFiles('train', '1', phoneHMMs, stateList)
saveFiles('test', '2', phoneHMMs, stateList)


# -- Loads the initial npz file, and creates a training/validation split -- #
npzfile = np.load("traindata1.npz", allow_pickle=True)
splitSet(npzfile['data'])


# -- Loads the test, training and validation dataset file -- #
print("Open data files")
splitdata = np.load("splitdata.npz", allow_pickle=True)
testdata = np.load("testdata2.npz", allow_pickle=True)['data']
trainingdata = splitdata['trainingdata']
validationdata = splitdata['validationdata']

print("SCALE")
# -- Define the scaler -- #
scaler = StandardScaler()

# -- Create the standardized datasets with dynamic features -- #
lmfcc_train_x_dyn = standardize(trainingdata, 'lmfcc' , 'all', True, scaler, "lmfcc_train_x_dyn")
lmfcc_val_x_dyn = standardize(validationdata, 'lmfcc' , 'all', True, scaler, "lmfcc_val_x_dyn")
lmfcc_test_x_dyn = standardize(testdata, 'lmfcc' , 'all', True, scaler, "lmfcc_test_x_dyn")
mspec_train_x_dyn = standardize(trainingdata, 'mspec' , 'all', True, scaler, "mspec_train_x_dyn")
mspec_val_x_dyn = standardize(validationdata, 'mspec' , 'all', True, scaler, "mspec_val_x_dyn")
mspec_test_x_dyn = standardize(testdata, 'mspec' , 'all', True, scaler, "mspec_test_x_dyn")

# -- Create the standardized datasets with regular features -- #
lmfcc_train_x_reg = standardize(trainingdata, 'lmfcc' , 'all', False, scaler, "lmfcc_train_x_reg")
lmfcc_val_x_reg = standardize(validationdata, 'lmfcc' , 'all', False, scaler, "lmfcc_val_x_reg")
lmfcc_test_x_reg = standardize(testdata, 'lmfcc' , 'all', False, scaler, "lmfcc_test_x_reg")
mspec_train_x_reg = standardize(trainingdata, 'mspec' , 'all', False, scaler, "mspec_train_x_reg")
mspec_val_x_reg = standardize(validationdata, 'mspec' , 'all', False, scaler, "mspec_val_x_reg")
mspec_test_x_reg = standardize(testdata, 'mspec' , 'all', False, scaler, "mspec_test_x_reg")

# -- Create the target datasets -- #
train_y = standardize(trainingdata, 'targets', 'all', False, None, "train_y")
val_y = standardize(validationdata, 'targets', 'all', False, None, "val_y")
test_y = standardize(testdata, 'targets', 'all', False, None, "test_y")

'''

#modelBuilder(stateList, True)


plotHistory()
