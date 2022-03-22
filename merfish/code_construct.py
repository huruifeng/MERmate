# Example Code Construction
# Purpose: To illustrate construction of a MHD4 code and a MHD2 code.
# -------------------------------------------------------------------------
import numpy as np

from merfish.codes.hamming_words import *

## Generate MHD4 Code
# Generate the Extended Hamming Code
numDataBits = 11
EHwords = GenerateExtendedHammingWords(numDataBits)

# Find Hamming Weight
hammingWeight = np.sum(EHwords,1)

# Cut words
MHD4words = EHwords[hammingWeight==4,:]

# print properties of code
print('-----------------------------------------------------------------')
print('Constructed',MHD4words.shape[0], 'barcodes/words')
print('Found the following hamming weights:')
print(list(np.unique(np.sum(MHD4words,1))))

# Check HD
hammingDistance = lambda x,y: sum(abs(x-y))
measuredDistances = np.empty((MHD4words.shape[0], MHD4words.shape[0]))
measuredDistances[:] = np.Inf
for i in range(MHD4words.shape[0]):
    for j in range(MHD4words.shape[0]):
        if i==j:
            continue
        else:
            measuredDistances[i,j] = hammingDistance(MHD4words[i,:],MHD4words[j,:])

minDist = np.min(measuredDistances)
print('Found the following minimum HD:')
print(list(np.unique(minDist)))

##############################################################################
## Generate MHD2 Code
numBits = 14
onBitInds = nchoosek(list(range(numBits)), 4)
MHD2words = np.zeros((onBitInds.shape[0], numBits))
for i in range(MHD2words.shape[0]):
    MHD2words[i,onBitInds[i,:]] = 1

# print Properties
print('-----------------------------------------------------------------')
print('Constructed',MHD2words.shape[0],'barcodes/words')
print('Found the following hamming weights:')
print(list(np.unique(np.sum(MHD2words,1))))

# Check HD
hammingDistance = lambda x,y: sum(abs(x-y))
measuredDistances = np.empty((MHD2words.shape[0], MHD2words.shape[0]))
measuredDistances[:] = np.Inf
for i in range(MHD2words.shape[0]):
    for j in range(MHD2words.shape[0]):
        if i==j:
            continue
        else:
            measuredDistances[i,j] = hammingDistance(MHD2words[i,:], MHD2words[j,:])

minDist = np.min(measuredDistances)
print('Found the following minimum HD:')
print(list(np.unique(minDist)))


