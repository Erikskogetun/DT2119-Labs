import numpy as np
from lab2_tools import *
from prondict import *
import matplotlib.pyplot as plt

def compareUtterances(thefunc):
    for utteranceidx, utterance in enumerate(data):
        maxProb = None
        maxDigit = None

        for digitidx, digit in enumerate(isolated):
            concd = concatHMMs(phoneHMMs, isolated[digit])
            lmndd = log_multivariate_normal_density_diag(utterance['lmfcc'], concd['means'], concd['covars'])
            funcVal, funcB = thefunc(lmndd, np.log(concd['startprob'][:-1]), np.log(concd['transmat'][:-1, :-1]))

            if(maxProb is None or funcVal > maxProb):
                maxDigit = digit
                maxProb = funcVal

        print("FOR UTTERANCE " + str(utterance['digit']) + " BY " + str(utterance['gender']) + " WINNER HMM: " + str(maxDigit))


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """

    concatHMM = {}
    newdim = len(hmm1['transmat']) + len(hmm2['transmat']) - 1

    concatHMM['name'] = hmm1['name'] + " " + hmm2['name']
    concatHMM['startprob'] = np.zeros([newdim])
    concatHMM['transmat'] = np.zeros([newdim, newdim])
    concatHMM['means'] = np.concatenate((hmm1['means'], hmm2['means']))
    concatHMM['covars'] = np.concatenate((hmm1['covars'], hmm2['covars']))


    for idx1, element1 in enumerate(hmm1['startprob']):
    	if (idx1 != len(hmm1['startprob']) - 1): # If its not the last element
    		concatHMM['startprob'][idx1] = element1
    	else:
    		for idx2, element2 in enumerate(hmm2['startprob']):
    			concatHMM['startprob'][idx1 + idx2] = element1 * element2

    for height in range(hmm1['transmat'].shape[1] - 1):
        for idx1, element1 in enumerate(hmm1['transmat'][height]):
            if (idx1 != len(hmm1['transmat'][height]) - 1):
                concatHMM['transmat'][height][idx1] = element1
            else:

                for idx2, element2 in enumerate(hmm2['startprob']):
                    concatHMM['transmat'][height][idx1 + idx2] = element1 * element2

    concatHMM['transmat'][hmm1['transmat'].shape[1] - 1:,hmm1['transmat'].shape[0] - 1: ] = hmm2['transmat']

    return concatHMM

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name.
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
       """

    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
        """

    # TODO: Fyll i denna

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
        """

    forward_prob = np.zeros((log_emlik.shape))

    forward_prob[0] = log_startprob.T + log_emlik[0]

    for n in range(1, forward_prob.shape[0]):
        for i in range(forward_prob.shape[1]):
            forward_prob[n, i] = logsumexp(forward_prob[n - 1] + log_transmat[:, i]) + log_emlik[n, i]
    return logsumexp(forward_prob[len(forward_prob) - 1]), forward_prob


def viterbiBacktrack(B, lastIdx):
    viterbi_path = [lastIdx]
    for i in reversed(range(1, B.shape[0])):
        viterbi_path.append(B[i, viterbi_path[-1]])
    viterbi_path.reverse()
    return np.array(viterbi_path)

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
        """

    B = np.zeros(log_emlik.shape, dtype = int)
    V = np.zeros(log_emlik.shape)
    V[0] = log_startprob.flatten() + log_emlik[0]

    for n in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            V[n][j] = np.max(V[n - 1,:] + log_transmat[:,j]) + log_emlik[n, j]
            B[n][j] = np.argmax(V[n - 1,:] + log_transmat[:,j])

    # Backtrack to take viteri path
    viterbi_path = viterbiBacktrack(B, np.argmax(V[ log_emlik.shape[0] - 1]))

    return np.max(V[ log_emlik.shape[0] - 1]), viterbi_path

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
        """

    log_b = np.zeros(log_emlik.shape)
    for n in reversed(range(log_emlik.shape[0] - 1)):
        for i in range(log_emlik.shape[1]):
            log_b[n, i] = logsumexp(log_transmat[i,:] + log_emlik[n + 1, :] + log_b[n + 1,:])
    return logsumexp(log_b[0]), log_b

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
        """

    # From appendix A.4
    return log_alpha + log_beta - logsumexp(log_alpha[len(log_alpha) - 1])


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
	""" Update Gaussian parameters with diagonal covariance

	Args:
	X: NxD array of feature vectors
	log_gamma: NxM state posterior probabilities in log domain
	varianceFloor: minimum allowed variance scalar
	were N is the lenght of the observation sequence, D is the
	dimensionality of the feature vectors and M is the number of
	states in the model

	Outputs:
	means: MxD mean vectors for each state
	covars: MxD covariance (variance) vectors for each state
	"""
	gamma = np.exp(log_gamma) # Into regular domain

	print()

	means = np.zeros((log_gamma.shape[1], X.shape[1]))
	covars = np.zeros((log_gamma.shape[1], X.shape[1]))

	for i in range(log_gamma.shape[1]):
		gamma_sum = np.sum(gamma[:,i])

		means[i] = np.sum(gamma[:,i].reshape(-1, 1) * X, axis = 0) / gamma_sum

		covars[i] = np.sum(gamma[:,i].reshape(-1, 1) * (X - means[i])**2, axis = 0) / gamma_sum
		covars[covars < varianceFloor] = varianceFloor

	return (means, covars)

def ExpectationMax(X, HMM, max_iter = 20, tol = 1):
	prev_likelihood = None
	iterations = 0

	while iterations < max_iter:

		iterations += 1

		obsloglik = log_multivariate_normal_density_diag(X, HMM['means'], HMM['covars'])

		ForwardMax, ForwardProbs = forward(obsloglik, np.log(HMM['startprob'][:-1]), np.log(HMM['transmat'][:-1,:-1]))
		BackwardMax, BackwardProbs  = backward(obsloglik, np.log(HMM['startprob'][:-1]), np.log(HMM['transmat'][:-1,:-1]))
		gamma = statePosteriors(ForwardProbs, BackwardProbs)
		HMM['means'], HMM['covars'] = updateMeanAndVar(X, gamma)

		print("Iteration " + str(iterations) + ": ")
		print(ForwardMax)

		if ((prev_likelihood == None) or (ForwardMax - prev_likelihood > tol)):
			prev_likelihood = ForwardMax
		else:
			break;

	return HMM
