import os
import numpy.matlib

from utils.funcs import *
from utils.misc import *

def add():
    return 10

def CodebookToMap(codebook, **kwargs): ##return [codeMap, geneNames, codewords, parameters]
    # ------------------------------------------------------------------------
    # [codeMap, geneNames, codeWords] = CodebookToMap(codebook, varargin)
    # This function generates a container.Map object from a either the path to
    #   a valid codebook or a codebook structure.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # --codebook/Either a path to a valid codebook or a codebook structure.
    #
    #--------------------------------------------------------------------------
    # Outputs
    # --codeMap/A container.Map object with keys corresponding to the valid
    #   codewords described in the codebook (and error correctable codewords)
    #   and the names of the corresponding objects (genes).
    # --geneNames/A cell array of the value entries for the codeMap
    # --codewords/A cell array of the key entries for the codeMap
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    errCorrFunc = []
    keyType = 'int'
    mapContents = 'all'

    if "errCorrFunc" in kwargs: errCorrFunc = kwargs["errCorrFunc"]
    if "keyType" in kwargs: keyType = kwargs["keyType"]
    if "mapContents" in kwargs: mapContents = kwargs["mapContents"]

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not codebook:
        error('[Error]:invalidArguments - A codebook is required')

    # -------------------------------------------------------------------------
    # Parse provided data
    # -------------------------------------------------------------------------
    if isinstance(codebook,str):
        if not os.path.exists(codebook):
            error('[Error]:invalidArguments - The provided path is not to a valid file.')
        else:
            try:
                codebook = fastaread(codebook)
            except:
                error('[Error]:invalidArguments - The provided path is not to a valid if !fasta file.')
    
    if isinstance(codebook,dict):
        if len(codebook)<=0:
            error('[Error]:invalidArguments - The provided codebook structure is empty.')

    # -------------------------------------------------------------------------
    # Determine Key Conversion Function
    # -------------------------------------------------------------------------
    removeWs = lambda x: x.replace(" ","") # Useful short hand for removing whitespace
    if keyType == 'int':
        keyConv= lambda x: bi2de(x)
    elif keyType == 'binStr':
        keyConv = removeWs
    
    # -------------------------------------------------------------------------
    # Determine values for map object
    # -------------------------------------------------------------------------
    exactMap = {}
    for header_i in codebook:
        key_i = keyConv(header_i)
        geneName = codebook[header_i].split(" ")[0]
        exactMap[key_i] = geneName
    
    # -------------------------------------------------------------------------
    # Return geneNames and codewords in the order in the codebook
    # -------------------------------------------------------------------------
    geneNames = list(exactMap.values())
    codewords = list(exactMap.keys())

    # -------------------------------------------------------------------------
    # Add correctable keys: Pass as logical arrays
    # -------------------------------------------------------------------------

    keys = []
    values = []
    if callable(errCorrFunc) or (not isinstance(errCorrFunc,list)):
        newKeyConv = lambda x: keyConv(x)
        for header_i in codebook:
            codewordLogical = [1 if x_i == "1" else 0 for x_i in removeWs(header_i)]
            newKeys = [newKeyConv("".join(map(str, x_i))) for x_i in errCorrFunc(codewordLogical)]
            if len(newKeys) > 0:
                geneName = codebook[header_i].split(" ")[0]
                newValues = np.tile([geneName], len(newKeys))
                keys += newKeys
                values += list(newValues)

        correctableMap = dict(zip(keys, values))

    else:
        codeMap = exactMap
    
    # -------------------------------------------------------------------------
    # Return codeMap
    # -------------------------------------------------------------------------
    if callable(errCorrFunc):
        if mapContents == 'exact':
            codeMap = exactMap
        elif mapContents == 'correctable':
            codeMap = correctableMap
        elif mapContents == 'all':
            codeMap = [exactMap,correctableMap]


    return codeMap
    

def SECDEDCorrectableWords(codeword, **kwargs):
    # ------------------------------------------------------------------------
    # codewords = SECDEDCorrectableWords(codeword, varargin)
    # This function generates a cell array of all codewords that would be
    # corrected to the given codeword using a SECDED code, i.e. hamming distance = 1.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # --codebook/The codeword used to find surrounding codewords.
    #
    #--------------------------------------------------------------------------
    # Outputs
    # --codewords/A cell array of 1xN logicals corresponding to the surrounding
    #   codewords

    # -------------------------------------------------------------------------
    # Generate surrounding words
    # -------------------------------------------------------------------------
    codewords = GenerateSurroundingCodewords(codeword, 1)
    return codewords

def GenerateSurroundingCodewords(codeword, hammDist, **kwargs):
    # ------------------------------------------------------------------------
    # codewords = GenerateSurroundingCodewords(codeword, hammDist, varargin)
    # This function generates a cell array of all codewords exactly the hamming
    # distance (hammDist) of the specified codeword.
    #
    # The codeword can be provided as a logical array, a string, or an integer.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # --codebook/The codeword used to find surrounding codewords.
    # --hammDist/The hamming distance between the 'central' codeword and the
    #   returned codewords.
    #
    #--------------------------------------------------------------------------
    # Outputs
    # --codewords/A cell array of 1xN logicals corresponding to the surrounding
    #   codewords
    #--------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------

    if not isinstance(hammDist,int):
        error('[Error]:invalidArguments - Incorrect hamming distance arguments')

    # -------------------------------------------------------------------------
    # Generate surrounding words
    # -------------------------------------------------------------------------
    C = nchoosek(list(range(len(codeword))), hammDist)
    codewords = np.tile(codeword, (len(C),1))

    for i in range(len(C)):
        codewords[i, C[i,:]] = np.logical_not(codewords[i, C[i,:]])

    return codewords







