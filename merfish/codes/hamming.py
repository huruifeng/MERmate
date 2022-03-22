from utils.funcs import *
from utils.misc import *

import numpy as np

def hammgen(m,*args):
    #HAMMGEN Produce parity-check and generator matrices for Hamming code.
    #   H = HAMMGEN(M) produces the parity-check matrix H for a given integer
    #   M, M >= 2. Hamming code is a single-error-correction code. The code
    #   length is N=2^M-1. The message length is K = 2^M - M - 1. The
    #   parity-check matrix is an M-by-N matrix.
    #
    #   H = HAMMGEN(M, P) produces the parity-check matrix H using a primitive
    #   polynomial P in GF(2^M). P can be a binary row vector that gives the
    #   coefficients, in order of ascending powers, of the polynomial.
    #   Alternatively, it can be a character vector to specify the polynomial
    #   in textual representation. When P is not specified or empty, the
    #   default primitive polynomial in GF(2^M), returned by GFPRIMFD(M), is
    #   used.
    #
    #   [H, G] = HAMMGEN(...) produces the parity-check matrix H as well as the
    #   generator matrix G. The generator matrix is a K-by-N matrix.
    #
    #   [H, G, N, K] = HAMMGEN(...) produces the codeword length N and the
    #   message length K.
    #
    #   Example:
    #   #   The following 3 function calls return the same results because
    #   #   'X^3+X+1' is the default primitive polynomial in GF(2^3).
    #
    #   hammgen(3)
    #   hammgen(3, [1 1 0 1])
    #   hammgen(3, 'X^3+X+1')
    #
    #   See also ENCODE, DECODE, GEN2PAR, GFPRIMFD.

    #   Copyright 1996-2018 The MathWorks, Inc.

    if len(args)>=1:
        p=args[0]
    else:
        p = 2

    if p<2 or (not is_prime(p)):
        error('[Error]: hammgen:InvalidP')

    p = gfprimdf(m)

    n = int(2**m - 1)
    k = int(n - m)

    h = gftuple(np.array(range(n)), p, 2)

    return h.transpose()


## The following fouction gfprimdf was modified from MATLAB file gfprimdf.m
def gfprimdf(*args):
    #GFPRIMDF Provide default primitive polynomials for a Galois field.
    #   POL = GFPRIMDF(M) outputs the default primitive polynomial POL in
    #   GF(2^M).
    #
    #   POL = GFPRIMDF(M, P) outputs the default primitive polynomial POL
    #   in GF(P^M).
    #
    #   Note: This function performs computations in GF(P^M) where P is prime. To
    #   work in GF(2^M), you can also use the PRIMPOLY function.
    #
    #   The default primitive polynomials are monic polynomials.
    #
    #   See also GFPRIMCK, GFPRIMFD, GFTUPLE, GFMINPOL.

    #   Copyright 1996-2014 The MathWorks, Inc.

    if len(args) < 1:
        error('[Error]: gfprimdf:Invalid number of arguments')
    elif len(args)==1:
        m = args[0]
        p = 2
    else:
        m = args[0]
        p = args[1]

    # Error checking
    if m<1:
        error('[Error]: gfprimdf:InvalidM')

    if p<2 or (not is_prime(p)):
        error('[Error]: gfprimdf:InvalidP')

    # The polynomials that are stored in the database over GF(2).
    if ( (p == 2) and (m <= 26) ):
        if m==1:
            pol = [1,1]
        elif m==2:
            pol = [1,1,1]
        elif m==3:
            pol = [1,1,0,1]
        elif m==4:
            pol = [1,1,0,0,1]
        elif m==5:
            pol = [1,0,1,0,0,1]
        elif m==6:
            pol = [1,1,0,0,0,0,1]
        elif m==7:
            pol = [1,0,0,1,0,0,0,1]
        elif m==8:
            pol = [1,0,1,1,1,0,0,0,1]
        elif m==9:
            pol = [1,0,0,0,1,0,0,0,0,1]
        elif m==10:
            pol = [1,0,0,1,0,0,0,0,0,0,1]
        elif m==11:
            pol = [1,0,1,0,0,0,0,0,0,0,0,1]
        elif m==12:
            pol = [1,1,0,0,1,0,1,0,0,0,0,0,1]
        elif m==13:
            pol = [1,1,0,1,1,0,0,0,0,0,0,0,0,1]
        elif m==14:
            pol = [1,1,0,0,0,0,1,0,0,0,1,0,0,0,1]
        elif m==15:
            pol = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==16:
            pol = [1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1]
        elif m==17:
            pol = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==18:
            pol = [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]
        elif m==19:
            pol = [1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==20:
            pol = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==21:
            pol = [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==22:
            pol = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==23:
            pol = [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==24:
            pol = [1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==25:
            pol = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif m==26:
            pol = [1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

    # The polynomials that are stored in the database over GF(3).
    elif (p == 3) and (m <= 12):
        if m== 1:
            pol = [1,1]
        elif m== 2:
            pol = [2,1,1]
        elif m== 3:
            pol = [1,2,0,1]
        elif m== 4:
            pol = [2,1,0,0,1]
        elif m== 5:
            pol = [1,2,0,0,0,1]
        elif m== 6:
            pol = [2,1,0,0,0,0,1]
        elif m== 7:
            pol = [1,0,2,0,0,0,0,1]
        elif m== 8:
            pol = [2,0,0,1,0,0,0,0,1]
        elif m== 9:
            pol = [1,0,0,0,2,0,0,0,0,1]
        elif m== 10:
            pol = [2,1,0,1,0,0,0,0,0,0,1]
        elif m== 11:
            pol = [1,0,2,0,0,0,0,0,0,0,0,1]
        elif m== 12:
            pol = [2,1,0,0,0,1,0,0,0,0,0,0,1]

    # The polynomials that are stored in the database over GF(5).
    elif (p == 5) and (m <= 9):
        if m== 1:
            pol = [2,1]
        elif m== 2:
            pol = [2,1,1]
        elif m== 3:
            pol = [2,3,0,1]
        elif m== 4:
            pol = [2,2,1,0,1]
        elif m== 5:
            pol = [2,4,0,0,0,1]
        elif m== 6:
            pol = [2,1,0,0,0,0,1]
        elif m== 7:
            pol = [2,3,0,0,0,0,0,1]
        elif m== 8:
            pol = [3,2,1,0,0,0,0,0,1]
        elif m== 9:
            pol = [3,0,0,0,2,0,0,0,0,1]

    # The polynomials that are stored in the database over GF(7).
    elif (p == 7) and (m <= 7):
        if m== 1:
            pol = [2,1]
        elif m== 2:
            pol = [3,1,1]
        elif m== 3:
            pol = [2,3,0,1]
        elif m== 4:
            pol = [5, 3,1,0,1]
        elif m== 5:
            pol = [4,1,0,0,0,1]
        elif m== 6:
            pol = [5, 1,3,0,0,0,1]
        elif m== 7:
            pol = [2,6, 0,0,0,0,0,1]

    elif (p == 11) and (m <= 5):
        if m== 1:
            pol = [3,1]
        elif m== 2:
            pol = [7, 1,1]
        elif m== 3:
            pol = [4,1,0,1]
        elif m== 4:
            pol = [2,1,0,0,1]
        elif m== 5:
            pol = [9,0,2,0,0,1]

    elif (p == 13) and (m <= 5):
        if m== 1:
            pol = [2,1]
        elif m== 2:
            pol = [2,1,1]
        elif m== 3:
            pol = [6,1,0,1]
        elif m== 4:
            pol = [2,1,1,0,1]
        elif m== 5:
            pol = [2,4,0,0,0,1]

    elif (p == 17) and (m <= 5):
        if m== 1:
            pol = [3,1]
        elif m== 2:
            pol = [3,1,1]
        elif m== 3:
            pol = [3,1,0,1]
        elif m== 4:
            pol = [11,1,0,0,1]
        elif m== 5:
            pol = [3,1,0,0,0,1]

    else:
        # Call GFPRIMFD for polynomials that are not stored in the database over GF(P>2).
        error('[Warning]: gfprimdf:OutsideDatabase', m, p)
        # pol = gfprimfd(m,'min',p)

    return pol

    # -- end of gfprimdf--




def gftuple(a, *args): ## [poly_form, exp_form]
    #GFTUPLE Simplify or convert the format of elements of a Galois field.
    #   For all syntaxes, A is a matrix, each row of which represents an
    #   element of a Galois field.  If A is a column of integers, then MATLAB
    #   interprets each row as an exponential format of an element.  Negative
    #   integers and -Inf all represent the zero element of the field.  If A
    #   has more than one column MATLAB interprets each row as a polynomial
    #   format of an element. In that case, each entry of A must be an integer
    #   between 0 and P-1.
    #
    #   All formats are relative to a root of a primitive polynomial specified
    #   by the second input argument, described below.
    #
    #   TP = GFTUPLE(A, M) returns the simplest polynomial format of the
    #   elements that A represents, where the kth row of TP corresponds to the
    #   kth row of A.  Formats are relative to a root of the default primitive
    #   polynomial for GF(2^M).  M is a positive integer.
    #
    #   TP = GFTUPLE(A, PRIM_POLY) is the same as the syntax above, except that
    #   PRIM_POLY is a polynomial string or a row vector that lists the
    #   coefficients of a degree-M primitive polynomial for GF(2^M) in order of
    #   ascending exponents.
    #
    #   TP = GFTUPLE(A, M, P) is the same as TP = GFTUPLE(A, M) except that 2
    #   is replaced by a prime number P.
    #
    #   TP = GFTUPLE(A, PRIM_POLY, P) is the same as TP = GFTUPLE(A, PRIM_POLY)
    #   except that 2 is replaced by a prime number P.
    #
    #   TP = GFTUPLE(A, PRIM_POLY, P, PRIM_CK) is the same as the syntax above
    #   except that GFTUPLE checks whether PRIM_POLY represents a polynomial
    #   that is indeed primitive.  If not, then GFTUPLE generates an error and
    #   does not return TP.  The input argument PRIM_CK can be any number or
    #   string.
    #
    #   [TP, EXPFORM] = GFTUPLE(...) returns the additional matrix EXPFORM.
    #   The kth row of EXPFORM is the simplest exponential format of the
    #   element that the kth row of A represents.
    #
    #   Note: This function performs computations in GF(P^M) where P is prime.
    #   To perform equivalent computations in GF(2^M), you can also apply the
    #   .^ operator and the LOG function to Galois arrays.
    #
    #   In exponential format, the number k represents the element alpha^k,
    #   where alpha is a root of the chosen primitive polynomial.  In
    #   polynomial format, the row vector k represents a list of coefficients
    #   in order of ascending exponents. For E.g.: For GF(5), k = [4 3 0 2]
    #   represents 4 + 3x + 2x^3.
    #
    #   To generate a FIELD matrix over GF(P^M), as used by other GF functions
    #   such as GFADD, the following command may be used:
    #   FIELD = GFTUPLE([-1 : P^M-2]', M, P)
    #
    #   See also GFADD, GFMUL, GFCONV, GFDIV, GFDECONV, GFPRIMDF.
    
    #   Copyright 1996-2018 The MathWorks, Inc.
    

    # Error checking - P.
    if len(args) < 1:
        error('[Error]:gftuple:Invalid Args numbers.')
    elif len(args) <2:
        p = 2
    else:
        p = args[1]
        if p<2 or (not is_prime(p)):
            error('[Error]:gftuple:InvalidP')

    # [m_a, n_a] = a.shape
    if len(a.shape) == 1:
        n_a = 1
        m_a = len(a)
    else:
        [m_a, n_a] = a.shape

    if n_a == 1:
        if not np.isreal(a[0]):
            error('[Error]: gftuple:ElementsOfANotInt')
    else:
        int_flag = 0
        for a_i in a:
            if np.floor(a_i) != a_i:
                int_flag = 1
        if np.any(a<0) or int_flag:
            error('[Error]:gftuple:ElementsOfANotPositiveInt')

    # Error checking - PRIM_POLY.
    prim_poly = np.array(args[0])

    if len(prim_poly.shape) == 1:
        m_pp = 1
        n_pp = len(prim_poly)
    else:
        [m_pp, n_pp] = prim_poly.shape

    if m_pp > 1:
        error('[Error]: gftuple:InvalidSecondArg-prim_poly')
    else:
        if (n_pp == 1):
            if ( len(prim_poly)==0 or np.any(prim_poly < 1) ):
                error('[Error]:gftuple:InvalidM')
            else:
                m = prim_poly
                prim_poly = gfprimdf(m, p)
        else:
            if len(prim_poly) == 0:
                error('[Error]:gftuple:InvalidPrim_poly1')
            if (prim_poly[0]==0 or prim_poly[n_pp-1]!=1):
                error('[Error]:gftuple:Prim_polyNotMonic')
            if np.any( (prim_poly >= p) | (prim_poly < 0) ):
                if p == 2 :
                    error('[Error]:gftuple:InvalidPrim_polyForP2')
                else:
                    error('[Error]:gftuple:InvalidPrim_poly2')
            m = n_pp - 1
            if len(args) >=3:
                # if (gfprimck(prim_poly, p) ~= 1)
                #     error('[Error]:gftuple:NotAPrim_poly')
                error('[Error]:gftuple:NotAPrim_poly')
    
    q = p**m
    
    # The 'alpha^m' equivalence is determined from the primitive polynomial.
    alpha_m = (-1*prim_poly[0:m]) % p
    
    # First calculate the full 'field' matrix.  Each row of the 'field'
    # matrix is the polynomial representation of one element in GF(P^M).
    field = np.zeros((q,m))
    field[1:m+1,:] = np.eye(m)
    for k in range(m+1,q):
        fvec = np.asarray([0] + list(field[k-1,:]))
        if (fvec[m]>0):
            fvec[0:m] = (fvec[0:m]+fvec[m]*alpha_m) % p
        field[k,:] = fvec[0:m]
    
    # Calculate the simplest polynomial form of the input 'a'.
    poly_form = np.zeros((m_a, m))
    if n_a == 1:
        # Exponential input representation case.
        idx = np.where(a > (q-2))
        a[idx[0]] = a[idx] % (q-1)
    
        a[a<0] = -1
        poly_form = field[a+1 , : ]
    
    else:
        # Polynomial input representation case.
        # Cycle through each input row.
        for k in range(m_a):
            at1 = gftrunc(a[k])
            at = at1 + np.zeros((1, q-1-(len(at1) % (q-1) )))
            at = np.reshape(at,q-1,len(at)/(q-1))
            at = sum(at,1) # p
            poly_form[k,:] =  at*field[1:q,:] #  p

    # Calculate the simplest exponential form of the input 'a' if requested.
    # exp_form = np.zeros((m_a,1));
    # pvec = [p**i for i in range(m)]
    # for k in range(m_a):
    #     exp_form[k,:] = np.where(field*pvec == poly_form[k,:]*pvec ) - 2
    # exp_form[exp_form < 0] = -np.Inf

    return poly_form
    #--- end of gftuple --

def gftrunc(arr):
    arr_rev =arr[::-1]
    flag = 0
    new_arr = []
    for x in arr_rev:
        if x == 0 and flag==0:
            continue
        else:
            flag=1
            new_arr.append(x)

    return np.array(new_arr[::-1])


def gen2par(g):
    #GEN2PAR Convert between parity-check and generator matrices.
    #   H = GEN2PAR(G) computes parity-check matrix H from the standard form of a
    #   generator matrix G. The function can also used to convert a parity-check
    #   matrix to a generator matrix. The conversion is used in GF(2) only.
    #
    #   See also CYCLGEN, HAMMGEN.
    
    #   Copyright 1996-2016 The MathWorks, Inc.
    
    [n,m] = g.shape
    if n >= m:
        error('[Error]:gen2par:InvalidInput')
    
    I = np.eye(n)
    if np.array_equal(g[:, m-n:m], I):
        h = np.concatenate((np.eye(m-n),np.transpose(g[:,:m-n])),axis=1)
    elif np.array_equal(g[:, :n], I):
        h = np.concatenate((np.transpose(g[:,n:m]), np.eye(m-n)),axis=1)
    else:
        error('[Error]:gen2par:InvalidInput')

    return h
    
    # eof
    
