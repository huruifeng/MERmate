# This is a main entrance for running functions.
##
#3 GitHub Token: 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, Expire: 01292022

from version import __version__

import os

def print_logo():
    ##
    # Use a breakpoint in the code line below to debug your script.
    print("MERmate version %s" % __version__)
    print("#######################################################")
    print("   _____ _____ _____ _____ _____ _____ __ __           ")
    print("  |  .  |   __| __  |   __|_  _ |   __|  |  |___ _ _   ")
    print("  | | | |   __|    -|   __| | | |__   |     | . | | |  ")
    print("  |_|_|_|_____|__|__|__|  |_____|_____|__|__|  _|_  |  ")
    print("                                            |_| |__|  ")
    print("#######################################################")


if __name__ == '__main__':
    print_logo()