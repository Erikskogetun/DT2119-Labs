import numpy as np

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




def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
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


phoneHMMs = np.load('lab2_models_v2.npz')['phoneHMMs'].item() # Ska egentligen använda models_all
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
#print(stateList)

filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(filename)
lmfcc = mfcc(samples)

wordTrans = list(path2info(filename)[2])
#print(wordTrans)

phoneTrans = words2phones(wordTrans, prondict, addShortPause=True)
print(phoneTrans)

utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
#print(utteranceHMM.keys())

stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]
#print(stateTrans)

print(lmfcc)
lmndd = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
print(lmndd.shape)
print(lmndd)

#print(lmndd.shape)
#print(utteranceHMM)
viterbimax, viterbipath = viterbi(lmndd, np.log(utteranceHMM['startprob'][:-1]), np.log(utteranceHMM['transmat'][:-1, :-1]))
print(viterbimax)
print(viterbipath)

#print(stateTrans)
symbol_sequence = [stateTrans[i] for i in viterbipath]
transcription = frames2trans(symbol_sequence,'z43a.lab')
print(transcription)
