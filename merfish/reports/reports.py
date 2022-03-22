import re

import matplotlib
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import uuid
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import stats, iqr

from merfish.analysis.image_data import ReadDax
from merfish.analysis.image_utils import fliptform, imtransform
from merfish.codes.codebook_process import GenerateSurroundingCodewords
from utils.funcs import *
from utils.misc import Ncolor, de2bi, bi2de, parula_map


def GenerateCompositeImage(words, imageData, **kwargs):
    # ------------------------------------------------------------------------
    # [figHandles, parameters] = GenerateCompositeImage(words, imageData, varargin)
    # This function creates a composite image
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # imageData/A structure array with an element for each image used to create
    #   the elements in words.  See CreateWordsStructure for information on
    #   field names.
    #--------------------------------------------------------------------------
    # Outputs
    # hybReport/A structure containing information from the report
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # October 2, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.


    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['printedUpdates']= True
    parameters['showNames']= False
    parameters['embedNames']= True
    parameters['saveAndClose']= True
    parameters['useSubFolderForCellReport']= True
    parameters['overwrite']= True
    parameters['figFormats'] = 'png'
    parameters['numImageColumns'] = 4
    parameters['displayHybLabel']= True
    # parameters['wordOverlayStyle'] = 'matchType'   #{'matchType', 'wordIdentity'}

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if  not isinstance(words,list) or len(imageData) ==0:
        error('[Error]:invalidArguments: Invalid words or imageData structures.')

    word_df = pd.DataFrame(words)
    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    parameters['reportsToGenerate']['cellWithWordsImage'] = True

    # -------------------------------------------------------------------------
    # Generate total report
    # -------------------------------------------------------------------------
    if parameters["reportsToGenerate"]["cellWithWordsImage"]:
        cellIDs = np.unique([iD["cellNum"] for iD in imageData],return_index=False)
        figHandles = {}
        for i in range(len(cellIDs)):
            # Index image Data and sort by hyb number
            localImageData = np.asarray([iD for iD in imageData if iD["cellNum"] == cellIDs[i]])
            hybNums = [lIData["hybNum"] for lIData in localImageData]
            sind = np.argsort(hybNums)
            localImageData = localImageData[sind]

            # Create figure handle.
            fig_name_i = 'CellWithWords_Cell_'+str(cellIDs[i])
            figHandles[i] = plt.figure(fig_name_i)
            ax = figHandles[i].add_subplot()

            # Create aligned image
            alignedIm = np.zeros((localImageData[0]["imageH"], localImageData[0]["imageW"],len(localImageData)))
            for h in range(len(localImageData)):
                print("\t-> Processing cellWithWordsImage for cell",cellIDs[i]," - image",h+1,"/",len(localImageData),"...")
                dax= ReadDax(infoFile=localImageData[h]["infFilePath"],endFrame=0, verbose=False)[0]
                if dax.ndim >= 3:
                    dax = np.amax(dax,0) # max project the first frame
                [H,W] = dax.shape
                tformInv = fliptform(localImageData[h]["tform"])
                [alignedDax,_,_] = imtransform(dax,tformInv, xyscale=[1,1],xdata=[0, W],ydata=[0,H])
                alignedIm[:,:,h] = alignedDax

            # Add color and plot image
            gray = cm.get_cmap('gray', len(localImageData))  ## color object: matplotlib.colors.LinearSegmentedColormap
            # gray = cm.get_cmap('gray')(np.linspace(0, 1, len(localImageData))) ## color array: n*4
            Io,cmap = Ncolor(alignedIm,gray)
            Io_max = np.amax(Io)
            Io = Io / Io_max
            ax.imshow(Io,extent=[0, Io.shape[0], Io.shape[1], 0], cmap=cmap)

            # Find local words
            localWords = word_df.loc[word_df["cellID"] == cellIDs[i],:]

            # Plot Exact matches
            inds = localWords["isExactMatch"]
            ax.plot(localWords["wordCentroidX"][inds], localWords["wordCentroidY"][inds], 'go',fillstyle='none',markersize=4,markeredgewidth=0.5)

            # Plot corrected matches
            inds = localWords["isCorrectedMatch"]
            ax.plot(localWords["wordCentroidX"][inds], localWords["wordCentroidY"][inds], 'rx',markersize = 2,markeredgewidth=0.5)

            # Plot unmatched words
            inds = ~ (localWords["isExactMatch"] | localWords["isCorrectedMatch"])
            ax.plot(localWords["wordCentroidX"][inds], localWords["wordCentroidY"][inds], 'b.', markersize = 2)

            if parameters["saveAndClose"]:
                if parameters["useSubFolderForCellReport"]:
                    if not os.path.exists(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[i]))):
                        os.mkdir(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[i])))
                    saveFile = os.path.join(parameters["savePath"],
                                            "Cell_" + str(cellIDs[i]),
                                            fig_name_i + "." + parameters["figFormats"])
                    print("Saved: ",saveFile)
                    figHandles[i].savefig(saveFile)
                else:
                    saveFile = os.path.join(parameters["savePath"],
                                            fig_name_i + "." + parameters["figFormats"])
                    print("Saved: ", saveFile)
                    figHandles[i].savefig(saveFile)
            plt.close(figHandles[i])

    return [figHandles, parameters]

def GenerateOnBitHistograms(words, **kwargs):
    # ------------------------------------------------------------------------
    # [figHandles, parameters] = GenerateOnBitHistograms(words, varargin)
    # This function creates on bit histograms for the provided words for all
    # words or by cell.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array with an element for each word. See
    #   CreateWordsStructure for information about fields.
    #--------------------------------------------------------------------------
    # Outputs
    # figHandles/Handles for the generated figures.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # October 3, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.


    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports['numOnBitsHistByCell'] = True
    defaultReports['numOnBitsHistAllCells'] = True

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['reportsToGenerate'] = defaultReports
    parameters['printedUpdates'] = True
    parameters['saveAndClose'] = True
    parameters['useSubFolderForCellReport'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(words, list) or len(words) == 0:
        error('[Error]:invalidArguments', 'Invalid words data.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    word_df = pd.DataFrame(words)
    figHandles = {}

    # -------------------------------------------------------------------------
    # Cell By Cell Report
    # -------------------------------------------------------------------------
    if parameters["reportsToGenerate"]["numOnBitsHistByCell"]:
        cellIDs = word_df["cellID"].unique()
        for i in range(len(cellIDs)):
            fig_name_i = 'numOnBitsHist_cell' + str(cellIDs[i])
            figHandles[i] = plt.figure(fig_name_i)
            ax = figHandles[i].add_subplot()

            localWords = word_df.loc[word_df["cellID"] == cellIDs[i]]
            ax.hist(localWords["numOnBits"], list(range(localWords["numHyb"][0] + 1)),color="blue",edgecolor='black', linewidth=1.0)
            ax.set_xlabel('Number of On Bits')
            ax.set_xlim(0,localWords["numHyb"][0] + 1)
            ax.set_ylabel('Counts')
            ax.set_title('Cell ' +str(cellIDs[i]))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if parameters["saveAndClose"]:
                if parameters["useSubFolderForCellReport"]:
                    if not os.path.exists(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[i]))):
                        os.mkdir(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[i])))
                    saveFile = os.path.join(parameters["savePath"],
                                            "Cell_" + str(cellIDs[i]),
                                            fig_name_i + "." + parameters["figFormats"])
                    print("Saved: ",saveFile)
                    figHandles[i].savefig(saveFile)
                else:
                    saveFile = os.path.join(parameters["savePath"],
                                            fig_name_i + "." + parameters["figFormats"])
                    print("Saved: ", saveFile)
                    figHandles[i].savefig(saveFile)
            plt.close(figHandles[i])

    # -------------------------------------------------------------------------
    # All Cell Reports
    # -------------------------------------------------------------------------
    if  parameters["reportsToGenerate"]["numOnBitsHistAllCells"]:
        fig_name_i = 'numOnBitsHistAllCells' + str(cellIDs[i])
        figHandles[i+1] = plt.figure(fig_name_i)
        ax = figHandles[i].add_subplot()
        ax.hist(word_df["numOnBits"], list(range(word_df["numHyb"][0] +1)),color="blue",edgecolor='black', linewidth=1.0)
        ax.set_xlabel('Number of On Bits')
        ax.set_xlim(0, word_df["numHyb"][0] + 1)
        ax.set_ylabel('Counts')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"],
                                            fig_name_i + "." + parameters["figFormats"])
            print("Saved: ", saveFile)
            figHandles[i+1].savefig(saveFile)
        plt.close(figHandles[i])

    return [figHandles, parameters]


def GenerateFPKMReport(words, **kwargs):
    # ------------------------------------------------------------------------
    # [FPKMData, parameters] = GenerateFPKMReport(words, FPKMData, varargin)
    # This function generates an FPKM report from the provided words.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array with an element for each found word.
    #   See CreateWordsStructure for information on field names
    # FPKMdata/A structure array with the following fields
    #   --geneName: A string specifying the name of the RNA isform
    #   --FPKM: The FPKM for that gene
    #--------------------------------------------------------------------------
    # Outputs
    # FPKMData/A structure containing information from the report
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # September 26, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports['totalFPKMReport']=True  # {report name, 'off'/'on' do not/do display figure}
    defaultReports['cellByCellFPKMReport'] = False  #  Not included as a default report

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['reportsToGenerate'] = defaultReports
    parameters['printedUpdates'] = True
    parameters['FPKMReportExactMatchOnly']=False
    parameters['FPKMReportEmbedNames'] = True
    parameters['showNames'] = False
    parameters['embedNames'] = False

    parameters['saveAndClose'] = True
    parameters['useSubFolderForCellReport'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(words, list) or len(words) == 0:
        error('GenerateFPKMReport:invalidArguments', 'Invalid words data.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    if "FPKMData" not in parameters:
        error('GenerateFPKMReport:invalidArguments', 'FPKM data is missing.')

    FPKMData = pd.DataFrame(parameters["FPKMData"].items(),columns=["geneName","FPKM"])
    words = pd.DataFrame(words)

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if ('geneName' not in words) or ('geneName' not in FPKMData) or ('FPKM' not in FPKMData) or len(parameters["FPKMData"]) == 0:
        error('[Error]:invalidArguments', 'Invalid words or FPKMdata structures.')

    FPKMReport = {}
    # -------------------------------------------------------------------------
    # Identify valid words
    # -------------------------------------------------------------------------
    if parameters["FPKMReportExactMatchOnly"]:
        validWords = words["isExactMatch"]
        figNameModifier = 'ExactOnly'
    else:
        validWords = words["isExactMatch"] | words["isCorrectedMatch"]
        figNameModifier = 'ExactAndCorrected'

    if parameters["printedUpdates"]:
        print('---------------------------------------------------------------')
        print('Found',int(np.sum(validWords)),'valid words in', len(words),'total words')

    FPKMReport["wordInds"] = np.nonzero(validWords.values) # Save for report

    # -------------------------------------------------------------------------
    # Determine target gene names
    # -------------------------------------------------------------------------
    targetNames = FPKMData["geneName"].values

    # -------------------------------------------------------------------------
    # Generate total report
    # -------------------------------------------------------------------------
    if 'totalFPKMReport' in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]['totalFPKMReport']:
        fig_name_i = 'FPKMReportAllCells_'+figNameModifier
        fig_i = plt.figure(fig_name_i)
        FPKMReport["totalReportFigHandle"] = fig_i
        ax = fig_i.add_subplot()

        counts = GenerateGeneCounts(words["geneName"][validWords].values, targetNames) # Record number of names not in list

        FPKMReport["geneNames"] = list(targetNames)
        FPKMReport["geneNames"].append('unknown name')
        FPKMReport["counts"] = counts
        FPKMReport["countsWOUnknown"] = counts[:-1]
        FPKMReport["FPKM"] = FPKMData["FPKM"].tolist()

        if parameters["showNames"]:
            FPKMReport["pearsonCorr"] = PlotCorr2(FPKMData["FPKM"].tolist(), counts[:-1],axesHandle=ax,pointNames=targetNames)
            fig_name_i = fig_name_i + "withNames"
        else:
            FPKMReport["pearsonCorr"] = PlotCorr2(FPKMData["FPKM"].tolist(), counts[:-1], axesHandle=ax)

        if parameters["embedNames"]:
            pass
            # ImbedNamesInFigure(FPKMReport["totalReportFigHandle"], targetNames)

        ax.set_xlabel('FPKM');
        ax.set_ylabel('Counts');

        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"],
                                    fig_name_i + "." + parameters["figFormats"])
            FPKMReport["totalReportFigHandle"].savefig(saveFile)
            print("Saved: ", saveFile)
        plt.close(FPKMReport["totalReportFigHandle"])

        if parameters["printedUpdates"]:
            print('All Cells p log10: ', FPKMReport["pearsonCorr"]["log10rho"])

    # -------------------------------------------------------------------------
    # Generate Individual Cell Report
    # -------------------------------------------------------------------------
    if 'cellByCellFPKMReport' in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["cellByCellFPKMReport"]:
        cellIDs = words["cellID"].unique()
        FPKMReport["cellReportFigHandles"] = {}
        FPKMReport["cellReportCounts"] = {}
        FPKMReport["cellByCellPearsonCorr"]= {}
        for c in range(len(cellIDs)):
            fig_name_i = 'FPKMReportCell_' + str(cellIDs[c])
            fig_i = plt.figure(fig_name_i)
            FPKMReport["cellReportFigHandles"][c] = fig_i
            ax = fig_i.add_subplot()

            localInds = validWords & (words["cellID"] == cellIDs[c])
            counts = GenerateGeneCounts(words["geneName"][localInds].values, FPKMData["geneName"].values)
            FPKMReport["cellReportCounts"][c]= counts

            if parameters["showNames"]:
                FPKMReport["cellByCellPearsonCorr"][c] = PlotCorr2(FPKMData["FPKM"].tolist(), counts[:-1],axesHandle=ax,pointNames=targetNames)
            else:
                FPKMReport["cellByCellPearsonCorr"][c] = PlotCorr2(FPKMData["FPKM"].tolist(), counts[:-1], axesHandle=ax)
            if parameters["embedNames"]:
                pass
                # ImbedNamesInFigure(FPKMReport["cellReportFigHandles"][c], targetNames)

            ax.set_xlabel('FPKM')
            ax.set_ylabel('Counts')

            if parameters["saveAndClose"]:
                if parameters["useSubFolderForCellReport"]:
                    if not os.path.exists(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[c]))):
                        os.mkdir(os.path.join(parameters["savePath"], "Cell_" + str(cellIDs[c])))
                    saveFile = os.path.join(parameters["savePath"],
                                            "Cell_" + str(cellIDs[c]),
                                            fig_name_i + "." + parameters["figFormats"])
                    FPKMReport["cellReportFigHandles"][c].savefig(saveFile)
                    print("Saved: ", saveFile)
                else:
                    saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
                    FPKMReport["cellReportFigHandles"][c].savefig(saveFile)
                    print("Saved: ", saveFile)
            plt.close(FPKMReport["cellReportFigHandles"][c])

            if parameters["printedUpdates"]:
                print('Cell-'+str(cellIDs[c]),'p log10:',str(FPKMReport["cellByCellPearsonCorr"][c]["log10rho"]))

    return [FPKMReport, parameters]

# -------------------------------------------------------------------------
# Internal Function Definitions: Simple name-based histogram
# -------------------------------------------------------------------------
def GenerateGeneCounts(geneNames, targetNames):
    counts = [0.0] * len(targetNames)
    for j in range(len(targetNames)):
        inds = geneNames == targetNames[j]
        counts[j] = np.sum(inds);
        geneNames = geneNames[~inds]
    counts.append(len(geneNames))
    return counts

def PlotCorr2(x=[],y=[],**kwargs):
    # PlotCorr2(x,y)
    # creates log-log plot of correlation betweent x and y
    # stats = PlotCorr(x,y) returns stats.log10rho, stast.log10pvalue
    #  stats.rho and stats.pvalue for the correlation in addition to the plot
    #
    #  non-zero points are removed from log-log correlation and correlation
    #  plot, but not from the linear correlations computed.
    # Second version of PlotCorr
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['MarkerSize'] = 2
    parameters['FontSize'] = 4
    parameters['colorMap'] = 'jet'
    parameters['nameBuffer'] = 0.1
    parameters['figHandle'] = []
    parameters['axesHandle'] = []
    parameters['pointNames'] = []
    parameters['plotFunction'] = plt.loglog
    parameters['includeLog10'] = True
    parameters['includeLin'] =  True

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if x == [] or y==[]:
        error('PlotCorr2:invalidArguments: requires x,y')

    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    x = np.array(x)
    y = np.array(y)
    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    non0 = (x>0) & (y>0)
    nonNaN = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    pearsonCorr = {}
    if np.size(x[non0 & nonNaN])==0:
       print('[Warning]: no nonzero data to plot')
       pearsonCorr["log10rho"] = []
       pearsonCorr["log10pvalue"] = []
       pearsonCorr["rho"] = []
       pearsonCorr["pvalue"] = []
    else:
        if len(parameters["pointNames"])>0:
            parameters["plotFunction"](x,y,'k.',markersize=parameters["MarkerSize"])
            x_new = x+parameters["nameBuffer"]*x
            for i in range(len(x_new)):
                parameters['axesHandle'].text(x_new[i],y[i],parameters["pointNames"][i],fontsize=parameters["FontSize"])
        else:
            parameters["plotFunction"](x,y,'k.',markersize = parameters["MarkerSize"])

        x= x.reshape((np.size(x),))
        y = y.reshape((np.size(y),))

        # Calculate correlation coefficient
        [c1,p1] = stats.pearsonr(x[nonNaN],y[nonNaN])
        pearsonCorr["rho"] = c1
        pearsonCorr["pvalue"] = p1
        titleString = ''
        # Calculate correlation coefficient for log10 data
        if parameters["includeLog10"]:
            [c0,p0] = stats.pearsonr(np.log10(x[non0 & nonNaN]),np.log10(y[non0 & nonNaN]))

            pearsonCorr["log10rho"] = c0
            pearsonCorr["log10pvalue"] = p0
            titleString = "$\\rho_{log10} = " + f"{c0:.2f} (p= {p0:.4g})$"
        if parameters["includeLin"]:
            titleString = titleString+"\t$\\rho = " + f"{c1:.2f} (p={p1:.4g})$"
        parameters['axesHandle'].set_title(titleString,fontsize=10)
    return pearsonCorr

def GenerateMoleculeStatsReport(words, **kwargs):
    # ------------------------------------------------------------------------
    # moleculeStats = GenerateMoleculeStatsReport(words, varargin)
    # This function generates a series of basic statistics for all words and
    # also cell by cell.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array with an element for each found word.
    #   See CreateWordsStructure for information on field names
    #--------------------------------------------------------------------------
    # Outputs
    # moleculeStats/A structure containing information from the report
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # October 1, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports['molStats']=True
    defaultReports['molHist']=True
    defaultReports['molDistStats']=True

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['reportsToGenerate'] = defaultReports
    parameters['printedUpdates'] = True
    parameters['saveAndClose'] = True
    parameters['useSubFolderForCellReport'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    parameters['brightHistBins'] = 100
    parameters['distHistBins'] = 100

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(words, list) or len(words) == 0:
        error('GenerateFPKMReport:invalidArguments', 'Invalid words data.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    words = pd.DataFrame(words)

    moleculeStats = {}
    # -------------------------------------------------------------------------
    # Generate basic statistics: Molecule Properties
    # -------------------------------------------------------------------------
    mListFields = ['a', 'bg', 'h']
    numHyb = words["numHyb"][0]
    sem = lambda x: np.std(x)/np.sqrt(len(x))
    for j in mListFields:
        data = words[j].values
        moleculeStats[j + 'Hist'] = []
        moleculeStats[j + 'Mean'] = []
        moleculeStats[j + 'STD'] = []
        moleculeStats[j + 'SEM'] = []
        moleculeStats[j + 'Median'] = []
        moleculeStats[j + 'IQR'] = []
        moleculeStats[j + 'N'] = []
        for i in range(numHyb):
            localData = np.array([data_i[i] for data_i in data])
            localData = localData[~np.isnan(localData)]
            [n, x] = np.histogram(localData, parameters["brightHistBins"])
            moleculeStats[j + 'Hist'].append([n,x])
            moleculeStats[j + 'Mean'].append(np.mean(localData))
            moleculeStats[j + 'STD'].append(np.std(localData))
            moleculeStats[j + 'SEM'].append(sem(localData))
            moleculeStats[j + 'Median'].append(np.median(localData))
            moleculeStats[j + 'IQR'].append(iqr(localData))
            moleculeStats[j + 'N'].append(len(localData))

    # -------------------------------------------------------------------------
    # Generate basic statistics: Molecule distances from word center
    # -------------------------------------------------------------------------
    xPos = np.hstack(words["xc"][words["numOnBits"]>1].values)
    xPos[xPos==0] = np.nan
    xDist = xPos.reshape((numHyb, np.sum(words["numOnBits"]>1)),order="F") - \
            np.tile(words["wordCentroidX"][words["numOnBits"]>1], (numHyb, 1))

    yPos = np.hstack(words["yc"][words["numOnBits"]>1].values)
    yPos[yPos==0] = np.nan
    yDist = yPos.reshape((numHyb, np.sum(words["numOnBits"]>1)),order="F") -\
            np.tile(words["wordCentroidY"][words["numOnBits"]>1], (numHyb,1))

    moleculeStats["xDist"] = xDist
    moleculeStats["yDist"] = yDist

    posNames = ['x', 'y']
    for j in posNames:
        moleculeStats[j + 'Mean'] = []
        moleculeStats[j + 'STD'] = []
        moleculeStats[j + 'MeanAbs']=[]
        moleculeStats[j + 'Hist'] = []
        for i in range(numHyb):
            data = moleculeStats[j+'Dist'][i]
            moleculeStats[j+'Mean'].append(np.mean(data[~np.isnan(data)]))
            moleculeStats[j+'STD'].append(np.std(data[~np.isnan(data)]))
            moleculeStats[j+'MeanAbs'].append(np.mean(np.abs(data[~np.isnan(data)])))
            [n, x] = np.histogram(data[~np.isnan(data)], parameters["distHistBins"])
            moleculeStats[j+'Hist'].append([n,x])

    # -------------------------------------------------------------------------
    # Plot Molecule Stats
    # -------------------------------------------------------------------------
    if "molStats" in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["molStats"]:
        fig_name_i = "molStats"
        fig_i = plt.figure(fig_name_i)

        for i in range(len(mListFields)):
            ax = fig_i.add_subplot(len(mListFields), 2, 2*i+1)
            ax.errorbar(x = list(range(1,numHyb+1)), y = np.array(moleculeStats[mListFields[i]+'Mean']),
                        yerr=np.array(moleculeStats[mListFields[i] + 'SEM']),fmt = ".b",lw=1,ms=2)
            ax.set_xlabel('Hybe Number')
            ax.set_xlim(0,numHyb+1)
            ax.set_ylabel(mListFields[i])

            ax = fig_i.add_subplot(len(mListFields), 2, 2*(i+1));
            ax.bar(list(range(1,numHyb+1)), moleculeStats[mListFields[i]+'N'])
            ax.set_xlabel('Hybe Number')
            ax.set_ylabel('Counts')
            ax.set_xlim(0,numHyb+1)
        plt.tight_layout()
        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"],fig_name_i + "." + parameters["figFormats"])
            fig_i.savefig(saveFile)
            print("Saved: ", saveFile)

        plt.close(fig_i)

    # -------------------------------------------------------------------------
    # Plot Molecule Stats: Position
    # -------------------------------------------------------------------------
    if "molDistStats" in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["molDistStats"]:
        fig_name_i = "molDistStats"
        fig_i = plt.figure(fig_name_i)

        for i in range(len(posNames)):
            ax = fig_i.add_subplot(len(posNames), 2, 2*i+1)
            ax.errorbar(list(range(1, numHyb + 1)), moleculeStats[posNames[i] + 'Mean'],
                        moleculeStats[posNames[i] + 'STD'], fmt = 'b.')
            ax.set_xlabel('Hybe Number')
            ax.set_xlim(0,numHyb+1)
            ax.set_ylabel(posNames[i])
            ax.set_title('Average/STD Distances')

            ax = fig_i.add_subplot(len(posNames), 2, 2 * (i + 1))
            localData = moleculeStats[posNames[i]+'Dist']
            localData = localData[~np.isnan(localData)]
            [n, x] = np.histogram(localData, parameters["distHistBins"])
            ax.stairs(n,x)
            ax.set_xlabel('Position')
            ax.set_ylabel('Counts')
            ax.set_title(", ".join(posNames))
        plt.tight_layout()
        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
            fig_i.savefig(saveFile)
            print("Saved: ", saveFile)

        plt.close(fig_i)


    # -------------------------------------------------------------------------
    # Plot Molecule Histograms
    # -------------------------------------------------------------------------
    if "molHist" in parameters["reportsToGenerate"] and parameters["reportsToGenerate"]["molHist"]:
        for i in mListFields:
            fig_name_i = "molHist_"+i
            fig_i = plt.figure(fig_name_i,figsize=(12,9))

            maxValueX = np.max([H_i[0][-1] for H_i in moleculeStats[i + 'Hist']])
            maxValueY = np.max([np.max(H_i[1]) for H_i in moleculeStats[i + 'Hist']])
            for j in range(numHyb):
                ax = fig_i.add_subplot(4, int(np.ceil(numHyb/4)), j+1)
                ax.ticklabel_format(axis='both', style='sci')
                ax.stairs(moleculeStats[i+'Hist'][j][0], moleculeStats[i+'Hist'][j][1])
                ax.set_xlabel(i)
                ax.set_xlim(0,maxValueX)
                ax.set_ylabel('Counts')
                ax.set_ylim(0,1.5*maxValueY)
                ax.set_title('Hyb '+str(j+1))
            plt.tight_layout()

            if parameters["saveAndClose"]:
                saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
                fig_i.savefig(saveFile)
                print("Saved: ", saveFile)

            plt.close(fig_i)

    # -------------------------------------------------------------------------
    # Plot Distances
    # -------------------------------------------------------------------------
    return [moleculeStats, parameters]

def GenerateBitFlipReport(words, **kwargs):
    # ------------------------------------------------------------------------
    # [figHandles, parameters] = GenerateBitFlipReport(words, varargin)
    # This function creates an error report for bit flip probabilities
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array with an element for each word. See
    #   CreateWordsStructure for information about fields.
    #--------------------------------------------------------------------------
    # Outputs
    # figHandles/Handles for the generated figures.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # October 20, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports['bitFlipProbabilitiesAverage'] = True #
    defaultReports['bitFlipProbabilitiesAllGenes'] = True #

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['reportsToGenerate'] = defaultReports
    parameters['printedUpdates'] = True
    parameters['saveAndClose'] = True
    parameters['useSubFolderForCellReport'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    parameters['probToUse'] = "exact"  ## {'exact', 'firstOrder'}
    parameters['errCorrFunc'] = []
    parameters['numHybs'] = 16

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(words, list) or len(words) == 0:
        error('GenerateFPKMReport:invalidArguments', 'Invalid words data.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    words = pd.DataFrame(words)

    bitFlipReport = {}
    # -------------------------------------------------------------------------
    # Check error check function
    # -------------------------------------------------------------------------
    if parameters["errCorrFunc"] == []:
        print('[Error]: GenerateBitFlipReport:invalidArguments - '
              'An error correction function must be provided to estimate bit flip probabilities')
        return

    # -------------------------------------------------------------------------
    # Normalize codeword type
    # -------------------------------------------------------------------------
    exactMap = parameters["exactMap"]
    geneNames = list(exactMap.values())
    codewords = list(exactMap.keys())

    if isinstance(codewords[0],str):
        codewords = [[x=='1' for x in codeword_i] for codeword_i in codewords]
    else:
        codewords = [de2bi(x,parameters["numHybs"]) for x in codewords]

    # -------------------------------------------------------------------------
    # Compute Word Histogram
    # -------------------------------------------------------------------------
    [n, x] = np.histogram(words["intCodeword"].values, np.arange(0,2**parameters["numHybs"]+1,1))

    # -------------------------------------------------------------------------
    # Generate bit flip probabilities
    # -------------------------------------------------------------------------
    firstOrderbitFlipProbabilities = np.empty((len(geneNames), parameters["numHybs"], 2))
    firstOrderbitFlipProbabilities[::] = np.nan

    exactBitFlipProbabilities = np.zeros((len(geneNames), parameters["numHybs"], 2))

    numCounts = np.zeros((len(codewords),))
    for i in range(len(codewords)):
        # Identify words
        correctWord = bi2de(codewords[i][::-1])
        surroundingWords = [bi2de(x[::-1]) for x in parameters["errCorrFunc"](codewords[i])]

        # Compute first order estimates of probability: Assumes that no more
        # than one error is possible
        totalCounts = np.sum(n[[correctWord] + surroundingWords])
        localFirstOrderProb = n[surroundingWords]/totalCounts
        numCounts[i] = totalCounts

        # Compute exact probability: assumes that any number of errors are
        # present but that no cross contamination between words occurs
        alpha = n[surroundingWords]/n[correctWord]
        localExactProb = alpha/(1+alpha)

        surroundingWords = np.array(surroundingWords)
        # Identify different transition types
        oneToZeroInds = np.argwhere(surroundingWords < correctWord).ravel()
        zeroToOneInds = np.argwhere(surroundingWords > correctWord).ravel()

        # Sort on order (left bits > right bits)
        oneToZeroSind = np.argsort(surroundingWords[oneToZeroInds])
        zeroToOneSind = np.argsort(surroundingWords[zeroToOneInds])

        # Record probabilities: first order
        firstOrderbitFlipProbabilities[i, oneToZeroInds[oneToZeroSind], 0] = \
            localFirstOrderProb[oneToZeroInds[oneToZeroSind]]
        firstOrderbitFlipProbabilities[i, zeroToOneInds[zeroToOneSind], 1] = \
            localFirstOrderProb[zeroToOneInds[zeroToOneSind]]

        # Record probabilities: exact
        exactBitFlipProbabilities[i, oneToZeroInds[oneToZeroSind], 0] = \
            localExactProb[oneToZeroInds[oneToZeroSind]]
        exactBitFlipProbabilities[i, zeroToOneInds[zeroToOneSind], 1] = \
            localExactProb[zeroToOneInds[zeroToOneSind]]

    # -------------------------------------------------------------------------
    # Archive results
    # -------------------------------------------------------------------------
    bitFlipReport["geneNames"] = geneNames
    bitFlipReport["counts"] = numCounts
    bitFlipReport["firstOrderProbabilities"] = firstOrderbitFlipProbabilities
    bitFlipReport["exactProbabilities"] = exactBitFlipProbabilities

    # -------------------------------------------------------------------------
    # Determine method for calculating probability
    # -------------------------------------------------------------------------
    if parameters["probToUse"] == 'exact':
        bitFlipReport["probabilities"] = bitFlipReport["exactProbabilities"]
    if parameters["probToUse"] == 'firstOrder':
        bitFlipReport["probabilities"] = bitFlipReport["firstOrderProbabilities"]

    # Average probabilities scaled by gene
    bitFlipReport["hybProb"] = np.squeeze(np.nanmean(bitFlipReport["probabilities"],0))
    bitFlipReport["hybProbErr"] = np.squeeze(np.nanstd(bitFlipReport["probabilities"],0))
    bitFlipReport["numCounts"] = numCounts

    # Compute weights
    weigthVec = numCounts/np.sum(numCounts)
    weights2d = np.tile(weigthVec.reshape(-1,1), (1,parameters["numHybs"]))
    weights = np.repeat(weights2d[:, :, np.newaxis], 2, axis=2)

    # Average probabilities scaled by counts per gene
    bitFlipReport["scaledHybProb"] = np.squeeze(np.nansum(bitFlipReport["probabilities"]*weights,0))
    bitFlipReport["scaledHybProbErr"] = np.squeeze(np.sqrt(weighted_nanvar(bitFlipReport["probabilities"],axis=0,weights=weights)))/\
                                        np.sqrt(len(weigthVec))

    # -------------------------------------------------------------------------
    # Display Probabilites
    # -------------------------------------------------------------------------
    figCount = 1
    if "bitFlipProbabilitiesAverage" in parameters["reportsToGenerate"] and \
            parameters["reportsToGenerate"]["bitFlipProbabilitiesAverage"]:
        fig_name_i = "bitFlipProbs"
        fig_i = plt.figure(fig_name_i)
        titles = ['1->0', '0->1']
        for i in [0,1]:
            ax= fig_i.add_subplot(1,2,i+1)
            ax.bar(list(range(1,parameters["numHybs"]+1)), np.squeeze(bitFlipReport["scaledHybProb"][:,i]))
            ax.errorbar(list(range(1,parameters["numHybs"]+1)),np.squeeze(bitFlipReport["scaledHybProb"][:,i]),
                        np.squeeze(bitFlipReport["scaledHybProbErr"][:,i]), fmt = 'k.')
            ax.set_xlabel('Hyb')
            ax.set_ylabel('Probability')
            ax.set_title(titles[i])
            ax.set_xlim(0,parameters["numHybs"] + 1)
        plt.tight_layout()

        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
            fig_i.savefig(saveFile)
            print("Saved: ", saveFile)

        plt.close(fig_i)

        figCount = figCount + 1

    # -------------------------------------------------------------------------
    # Display Probabilites for All Genes
    # -------------------------------------------------------------------------
    if "bitFlipProbabilitiesAllGenes" in parameters["reportsToGenerate"] and \
            parameters["reportsToGenerate"]["bitFlipProbabilitiesAllGenes"]:
        fig_name_i = "bitFlipProbsAllGenes"
        fig_i = plt.figure(fig_name_i,figsize=(12,12))
        titles = ['1->0', '0->1']
        for i in [0,1]:
            ax = fig_i.add_subplot(1, 2, i + 1)
            data = bitFlipReport["probabilities"][:,:,i]
            data = np.nan_to_num(data,nan=0.0)
            im = ax.imshow(data,aspect='auto',cmap=parula_map,vmin=0.0,vmax=1.0)
            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
            ax.set_xlabel('Hyb')
            ax.set_ylabel('Genes')
            ax.set_title(titles[i])
            inds =  ax.get_yticks()
            inds = inds[(inds>=0) & (inds < len(bitFlipReport["geneNames"]))].astype(int)
            ax.set_yticks(inds)
            ax.set_yticklabels([bitFlipReport["geneNames"][i] for i in inds])
            ax.set_xticks(list(range(data.shape[1])))
            ax.set_xticklabels([str(i) for i in range(1,data.shape[1]+1)])
        cax = fig_i.add_axes([ax.get_position().x1 + 0.03, ax.get_position().y0, 0.02, ax.get_position().height])
        plt.colorbar(im, cax=cax)

        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
            fig_i.savefig(saveFile)
            print("Saved: ", saveFile)

        plt.close(fig_i)
        figCount = figCount + 1

    return [bitFlipReport, parameters]


def GenerateHammingSphereReport(words, **kwargs):
    # ------------------------------------------------------------------------
    # [reportStruct, parameters] = GenerateHammingSphereReport(words, exactMap, varargin)
    # This function generates the counts in different hamming spheres for each
    # gene. The default is to calculate hamming sphere 0 and 1.
    # This analysis produces the confidence ratio.
    #--------------------------------------------------------------------------
    # Necessary Inputs
    # words/A structure array with an element for each word. See
    #   CreateWordsStructure for information about fields.
    #--------------------------------------------------------------------------
    # Outputs
    # figHandles/Handles for the generated figures.
    #--------------------------------------------------------------------------
    # Variable Inputs (Flag/ data type /(default)):
    #--------------------------------------------------------------------------
    # Jeffrey Moffitt
    # lmoffitt@mcb.harvard.edu
    # November 30, 2014
    #--------------------------------------------------------------------------
    # Copyright Presidents and Fellows of Harvard College, 2016.

    # -------------------------------------------------------------------------
    # Default Reports to Generate
    # -------------------------------------------------------------------------
    defaultReports = {}
    defaultReports["confidenceRatioReport"] = True

    # -------------------------------------------------------------------------
    # Default variables
    # -------------------------------------------------------------------------
    parameters = {}
    parameters['reportsToGenerate'] = defaultReports
    parameters['printedUpdates'] = True
    parameters['saveAndClose'] = True
    parameters['overwrite'] = True
    parameters['figFormats'] = 'png'

    parameters['subFolder'] = ""
    parameters['maxHammingSphere'] =1
    parameters['blankWordIdentifiers'] = ['blank', 'notarget']
    parameters['colorMap'] = "jet"
    parameters['numHistogramBins'] = 25

    # -------------------------------------------------------------------------
    # Parse necessary input
    # -------------------------------------------------------------------------
    if not isinstance(words, list) or len(words) == 0:
        error('GenerateFPKMReport:invalidArguments', 'Invalid words data.')

    # -------------------------------------------------------------------------
    # Parse variable input
    # -------------------------------------------------------------------------
    for k_i in kwargs:
        parameters[k_i] = kwargs[k_i]

    words = pd.DataFrame(words)

    reportStruct = {}
    # -------------------------------------------------------------------------
    # Define useful function for converting logical to string
    # -------------------------------------------------------------------------
    removeWs = lambda x: x.replace(" ","")

    # -------------------------------------------------------------------------
    # Extract properties of map
    # -------------------------------------------------------------------------
    exactMap = parameters["exactMap"]
    geneNames = list(exactMap.values())
    codeWords = list(exactMap.keys())

    intCodewords = [bi2de(x,direction="left") for x in codeWords]
    numHybs = len(codeWords[0])

    # -------------------------------------------------------------------------
    # Compute integer codeword histogram
    # -------------------------------------------------------------------------
    [wordCounts,x] = np.histogram(words["intCodeword"].values, np.arange(0,2**parameters["numHybs"]+1,1))

    # -------------------------------------------------------------------------
    # Compute hamming sphere counts
    # -------------------------------------------------------------------------
    reportStruct["geneNames"] = geneNames
    reportStruct["hammingSphereCounts"] = np.zeros((parameters["maxHammingSphere"] + 1,len(geneNames)))

    reportStruct["hammingSphereCounts"][0,:] = wordCounts[intCodewords]

    for i in range(parameters["maxHammingSphere"]):
        for j in range(len(geneNames)):
            surroundingWords = GenerateSurroundingCodewords(np.array(list((codeWords[j])))=='1', i+1)
            surroundingWordIntegers = [bi2de(x,"left") for x in surroundingWords]
            reportStruct["hammingSphereCounts"][i+1,j] = np.sum(wordCounts[surroundingWordIntegers])

    # -------------------------------------------------------------------------
    # Identify blank words
    # -------------------------------------------------------------------------
    # Identify blank word geneIDs
    isBlank = np.zeros((len(geneNames),))
    for i in parameters["blankWordIdentifiers"]:
        isBlank = np.logical_or([re.search(i, x)!=None for x in geneNames],isBlank)

    blankNames = np.array(geneNames)[isBlank]
    blankIDs = np.nonzero(isBlank)[0]
    nonBlankIDs = [i for i in range(len(geneNames)) if i not in blankIDs]

    reportStruct["blankNames"] = blankNames
    reportStruct["blankIDs"] = blankIDs
    reportStruct["nonBlankIDs"] = nonBlankIDs

    # -------------------------------------------------------------------------
    # Calculate histograms and CDF for 0/(1+0) ratio
    # -------------------------------------------------------------------------
    # This is the confidence ratio
    hammingSphereRatio = reportStruct["hammingSphereCounts"][0,:] / np.sum(reportStruct["hammingSphereCounts"][[0,1],:],0)

    [nNonBlank, x] = np.histogram(hammingSphereRatio[(~isBlank)&(~np.isnan(hammingSphereRatio))], bins=parameters["numHistogramBins"])
    [nBlank, x] = np.histogram(hammingSphereRatio[(isBlank)&(~np.isnan(hammingSphereRatio))], x)

    reportStruct["hammingSphere01Ratio"] = hammingSphereRatio
    reportStruct["nNonBlank"] = nNonBlank
    reportStruct["nBlank"] = nBlank
    reportStruct["histBins"] = x

    reportStruct["NonBlankCDF"] = np.cumsum(nNonBlank)/np.sum(nNonBlank)
    reportStruct["blankCDF"] = np.cumsum(nBlank)/np.sum(nBlank)

    sind = np.argsort(reportStruct["hammingSphere01Ratio"])[::-1]
    reportStruct["sortedInd"] = sind

    # -------------------------------------------------------------------------
    # Display Report
    # -------------------------------------------------------------------------
    if "confidenceRatioReport" in parameters["reportsToGenerate"] and \
            parameters["reportsToGenerate"]["confidenceRatioReport"]:
        fig_name_i = "confidenceRatioReport"
        fig_i = plt.figure(fig_name_i,figsize=(2550/300,1500/300),dpi=300)

        ax = fig_i.add_subplot(2,1, 1)
        bline = ax.plot(reportStruct["hammingSphere01Ratio"][sind], 'b',label = "RNA barcodes")
        revSind = np.zeros((len(geneNames, )))
        revSind[sind] = list(range(len(sind)))
        rx = ax.plot(revSind[blankIDs], reportStruct["hammingSphere01Ratio"][blankIDs], 'rx', label ='Blank barcodes' )
        ax.set_xlabel('Sorted Gene ID')
        ax.set_ylabel('Confidence Ratio')
        ax.set_xlim(0,len(geneNames)+1)
        # ax.set_ylim(0,1.2*np.nanmax(reportStruct["hammingSphere01Ratio"]))
        ax.legend(fontsize = 6)

        bar_center = [(reportStruct["histBins"][i] + reportStruct["histBins"][i + 1]) / 2 for i in
                      range(len(reportStruct["histBins"]) - 1)]
        ax=fig_i.add_subplot(2,2,3)
        nbh = ax.bar(bar_center,reportStruct["nNonBlank"],color = 'blue',alpha=0.5,width=1/len(bar_center),
                     edgecolor="white", linewidth=0.7, label = "RNA barcodes")
        bh = ax.bar(bar_center,reportStruct["nBlank"],color="red",alpha=0.5, width=1/len(bar_center),
                    edgecolor="white", linewidth=0.7, label = "Blank barcodes")
        ax.set_xlabel('Confidence Ratio')
        ax.set_ylabel('Counts')
        ax.set_xlim(reportStruct["histBins"][0] - np.mean(np.diff(reportStruct["histBins"])),
                reportStruct["histBins"][-1] + np.mean(np.diff(reportStruct["histBins"])))
        ax.set_ylim(0,1.2*np.max([np.max(reportStruct["nNonBlank"]),np.max(reportStruct["nBlank"])]))
        ax.legend(fontsize=6)

        ax = fig_i.add_subplot(2, 2, 4)
        nbh = ax.plot(bar_center, reportStruct["NonBlankCDF"], 'b',label = "RNA barcodes")
        bh = ax.plot(bar_center, reportStruct["blankCDF"], 'r',label = "Blank barcodes")
        ind = np.argwhere(reportStruct["blankCDF"] == 1).ravel()[0]
        ax.plot(reportStruct["histBins"][ind]*np.ones((2,)), [0,1], 'k--')

        ax.set_xlabel('Confidence Ratio')
        ax.set_ylabel('CDF')
        ax.legend(fontsize = 6)

        numAbove = np.sum(reportStruct["hammingSphere01Ratio"][reportStruct["nonBlankIDs"]] > np.nanmax(
            reportStruct["hammingSphere01Ratio"][reportStruct["blankIDs"]]))

        ax.set_title('Number Above: ' + str(numAbove))
        plt.tight_layout()

        if parameters["saveAndClose"]:
            saveFile = os.path.join(parameters["savePath"], fig_name_i + "." + parameters["figFormats"])
            fig_i.savefig(saveFile)
            print("Saved: ", saveFile)

        plt.close(fig_i)

    return [reportStruct, parameters]





