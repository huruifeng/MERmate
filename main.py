# This is a main entrance for running functions.

from version import __version__

import os

def print_logo():
    ## http://www.patorjk.com/software/taag/#p=display&f=Crazy&t=MERmate

    ver_str =" MERmate version: "+ __version__ + " "
    print("#"* int(31-len(ver_str)/2) + ver_str + "#"*int(31-len(ver_str)/2))
    print("    __  __   _____   ____                        _            ")
    print("   |  \/  | | ____| |  _ \   _ __ ___     __ _  | |_    ___   ")
    print("   | |\/| | |  _|   | |_) | | '_ ` _ \   / _` | | __|  / _ \\ ")
    print("   | |  | | | |___  |  _ <  | | | | | | | (_| | | |_  |  __/  ")
    print("   |_|  |_| |_____| |_| \_\ |_| |_| |_|  \__,_|  \__|  \___|  ")
    print("01001101 01000101 01010010 01101101 01100001 01110100 01100101")
    print("##############################################################")


if __name__ == '__main__':
    print_logo()