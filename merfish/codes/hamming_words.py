import numpy as np
from scipy.optimize import fsolve

from .hamming import *
from utils.misc import *
from utils.funcs import *

def GenerateExtendedHammingWords(numDataBits, **kwargs):
    # ------------------------------------------------------------------------
    # [words, generator, numParityBits] = GenerateExtendedHammingWords(numDataBits)
    # This function returns the minimum extended hamming code words for the
    # number of data bits specified. It also returns the generator matrix.
    #--------------------------------------------------------------------------
    # Outputs: words, gen, numParityBits
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if numDataBits < 1:
        error('[Error]:invalidArguments - A valid number of data bits is required.')

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    numOn =  0       # Returns a code with a fixed number of On bits (number of 1s in a code)
    numPar = 1   # A parallel.pool object can be provided to speed some calculations

    if "numOn" in kwargs: numOn = kwargs["numOn"]
    if "numPar" in kwargs: numPar = kwargs["numPar"]


    # -------------------------------------------------------------------------
    # Determine the number of required parity bits for the hamming code
    # -------------------------------------------------------------------------
    numParityBits = np.ceil(fsolve(lambda x: 2**x - x -1 - numDataBits, max(np.log2(numDataBits),1)))

    # -------------------------------------------------------------------------
    # Create the hamming code parity matrix
    # -------------------------------------------------------------------------
    par = hammgen(max(np.ceil(np.log2(numDataBits + numParityBits[0])), 3))

    # -------------------------------------------------------------------------
    # Create the extended hamming parity check matrix
    # -------------------------------------------------------------------------
    par = np.concatenate((par, np.zeros((par.shape[0],1))),axis=1)
    par = np.concatenate((par, np.ones((1, par.shape[1]))),axis=0)
    par = frref(par)[0] % 2

    # -------------------------------------------------------------------------
    # Create the generator
    # -------------------------------------------------------------------------
    gen = gen2par(par)

    # -------------------------------------------------------------------------
    # Create the shortened generator
    # -------------------------------------------------------------------------
    excessBits = gen.shape[0] - numDataBits
    gen = gen[0: gen.shape[0]-excessBits, 0: gen.shape[1]-excessBits]

    # -------------------------------------------------------------------------
    # Create the words
    # -------------------------------------------------------------------------
    if numOn == 0: # Return all. Memory inefficient for large codes
        wordDec = np.arange(2**numDataBits)
        words = de2bi(wordDec,gen.shape[0])
        words = (words@gen) %2
    else:
        # Determine number of data words
        numWords = 2**numDataBits

        # Decide the number of parallel workers to use
        numParToUse = min(numWords, numPar)

        words = np.zeros((0, gen.shape[1]))

        # Loop over words and keep only those that have the correct number of words
        for i in range(numWords):
            # Covert to binary
            localWord = de2bi(i,gen.shape[0])
            # Calculate extended hamming code word
            localWord = (localWord@gen)%2
            # Look for threshold
            if sum(localWord) == numOn:
                words = np.concatenate((words, localWord),axis=0)

    return words
    #
