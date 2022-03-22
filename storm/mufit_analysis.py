

import storm.sa_library.parameters as params
import storm.sa_utilities.std_analysis as std_analysis

from storm import find_peaks
from utils.funcs import error


def mufit_analysis(*args):
    if(len(args)==2):
        parameters = params.Parameters(args[1])
        mlist_file = args[0][:-4] + "_mlist.bin"
    elif(len(args)==3):
        parameters = params.Parameters(args[2])
        mlist_file = args[1]
    else:
        print("usage: mufit_analysis(movie_file.tif, mlist.bin ,parameters.xml)")
        error("Error in running mufit_analsyis")

    finder = find_peaks.initFindAndFit(parameters)
    std_analysis.standardAnalysis(finder,args[0], mlist_file, parameters)
