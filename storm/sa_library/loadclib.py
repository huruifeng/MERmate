#!/usr/bin/python
#
# Handle loading the correct C library.
#
# Hazen 11/14
#

import os
import ctypes
import sys

import storm


def loadCLibrary(library):

    # This assumes that all the code that will call C modules is one level
    # below the root storm_analysis directory.
    #
    # c_lib_path is something like:
    #    /usr/lib/python3.5/site-packages/storm_analysis
    #
    c_lib_path = os.path.dirname(os.path.abspath(storm.__file__))

    # All the C libraries are in the c_libraries directory.
    c_lib_path = os.path.join(c_lib_path, "c_libraries")
    directory = c_lib_path
    if (sys.platform == "win32"):
        return ctypes.cdll.LoadLibrary(os.path.join(directory,library + ".dll"))
    else:
        return ctypes.cdll.LoadLibrary(os.path.join(directory,library + ".so"))

#
# The MIT License
#
# Copyright (c) 2014 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
