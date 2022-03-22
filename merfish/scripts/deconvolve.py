import cv2
import numpy as np
from numpy.fft import ifftn, fftn
from scipy import ndimage

from utils.funcs import error
from utils.misc import fgauss2D

"""
This module containts utility functions for performing deconvolution on
images.
"""


def calculate_projectors(windowSize: int, sigmaG: float) -> list:
    """Calculate forward and backward projectors as described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function

    Returns:
        A list containing the forward and backward projectors to use for
        Lucy-Richardson deconvolution.
    """
    pf = fgauss2D(size=(windowSize, windowSize), sigma=sigmaG)
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    #
    # These values are from Guo et al.
    alpha = 0.001
    beta = 0.001
    n = 8

    # This is the cut-off frequency.
    kc = 1.0/(0.5 * 2.355 * sigmaG)

    # FFT frequencies
    kv = np.fft.fftfreq(pfFFT.shape[0])

    kx = np.zeros((kv.size, kv.size))
    for i in range(kv.size):
        kx[i, :] = np.copy(kv)

    ky = np.transpose(kx)
    kk = np.sqrt(kx*kx + ky*ky)

    # Wiener filter
    bWiener = pfFFT/(np.abs(pfFFT) * np.abs(pfFFT) + alpha)

    # Buttersworth filter
    eps = np.sqrt(1.0/(beta*beta) - 1)

    kkSqr = kk*kk/(kc*kc)
    bBWorth = 1.0/np.sqrt(1.0 + eps * eps * np.power(kkSqr, n))

    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth

    # back projector.
    pb = np.real(np.fft.ifft2(pbFFT))

    return [pf, pb]


def deconvolve_lucyrichardson(image: np.ndarray,
                              windowSize: int,
                              sigmaG: float,
                              iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function.

    Ported from Matlab deconvlucy.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    eps = np.finfo(float).eps
    Y = np.copy(image)
    J1 = np.copy(image)
    J2 = np.copy(image)
    wI = np.copy(image)
    imR = np.copy(image)
    reblurred = np.copy(image)
    tmpMat1 = np.zeros(image.shape, dtype=float)
    tmpMat2 = np.zeros(image.shape, dtype=float)
    T1 = np.zeros(image.shape, dtype=float)
    T2 = np.zeros(image.shape, dtype=float)
    l = 0

    if windowSize % 2 != 1:
        gaussianFilter = fgauss2D(size=(windowSize, windowSize),sigma=sigmaG)

    for i in range(iterationCount):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1) / (np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        if windowSize % 2 == 1:
            cv2.GaussianBlur(Y, (windowSize, windowSize), sigmaG, reblurred,
                             borderType=cv2.BORDER_REPLICATE)
        else:
            reblurred = ndimage.convolve(Y, gaussianFilter, mode='constant')
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        if windowSize % 2 == 1:
            cv2.GaussianBlur(imR, (windowSize, windowSize), sigmaG, imR,
                             borderType=cv2.BORDER_REPLICATE)
        else:
            imR = ndimage.convolve(imR, gaussianFilter, mode='constant')
            imR[imR > 2 ** 16] = 0
        np.copyto(J2, J1)
        np.multiply(Y, imR, out=J1)
        np.copyto(T2, T1)
        np.subtract(J1, Y, out=T1)
    return J1


def deconvolve_lucyrichardson_x(image, psf, num_iter):
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function.

    Ported from Matlab deconvlucy.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    eps = np.finfo(float).eps
    Y = np.copy(image)
    J1 = np.copy(image)
    J2 = np.copy(image)
    wI = np.copy(image)
    T1 = np.zeros((Y.shape[0]*Y.shape[1],), dtype=float)
    T2 = np.zeros((Y.shape[0]*Y.shape[1],), dtype=float)
    l = 0

    for i in range(num_iter):
        if i > 1:
            l = (T1@T2.T)/(T2@T2.T + eps)

        Y = J1 + l * (J1-J2)
        Y[Y<0] = 0

        ## 1)
        reblurred = np.rint(ndimage.convolve(Y, psf, mode='constant'))
        # reblurred = np.rint(ndimage.correlate(Y, psf, mode='constant', origin=-1))
        reblurred[reblurred<=0] = eps

        ## 2)
        imR = wI / reblurred
        imR = np.rint(imR+eps)

        ## 3)
        imR = ndimage.convolve(imR, psf, mode='constant')
        imR[imR<0] = 0
        imR= np.rint(imR)

        ## 4)
        np.copyto(J2, J1)
        J1 = np.multiply(Y, imR)
        np.copyto(T2, T1)
        T1 = J1.flatten("F") -  Y.flatten("F")
        T1[T1<0] = 0
    return J1

def deconvolve_lucyrichardson_guo(image: np.ndarray,
                                  windowSize: int,
                                  sigmaG: float,
                                  iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    [pf, pb] = calculate_projectors(windowSize, sigmaG)

    eps = 1.0e-6
    i_max = 2**16-1

    ek = np.copy(image)
    np.clip(ek, eps, None, ek)

    for i in range(iterationCount):
        ekf = cv2.filter2D(ek, -1, pf,
                           borderType=cv2.BORDER_REPLICATE)
        np.clip(ekf, eps, i_max, ekf)

        ek = ek*cv2.filter2D(image/ekf, -1, pb,
                             borderType=cv2.BORDER_REPLICATE)
        np.clip(ek, eps, i_max, ek)

    return ek

###########################################################
##Created on Mon Jan 18 18:05:26 2016
## @author: olgag

# this module contains all functions used to apply deconvolution to image

def psf2otf(psf, otf_size):
    # calculate otf from psf with size >= psf size

    if psf.any():  # if any psf element is non-zero
        # pad PSF with zeros up to image size
        pad_size = ((0, otf_size[0] - psf.shape[0]), (0, otf_size[1] - psf.shape[1]))
        psf_padded = np.pad(psf, pad_size, 'constant')

        # circularly shift psf
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[0] / 2)), axis=0)
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[1] / 2)), axis=1)

        # calculate otf
        otf = fftn(psf_padded)
        # this condition depends on psf size
        num_small = np.log2(psf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(otf.imag)) / np.max(abs(otf)) <= num_small:
            otf = otf.real
    else:  # if all psf elements are zero
        otf = np.zeros(otf_size)
    return otf


def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size

    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(psf.imag)) / np.max(abs(psf)) <= num_small:
            psf = psf.real

            # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0] / 2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1] / 2)), axis=1)

        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf


def deconvlucy(image, psf, num_iter, weight=None, subsmpl=1):
    # apply Richardson-Lucy deconvolution to image

    # calculate otf from psf with the same size as image
    otf = psf2otf(psf, np.array(image.shape) * subsmpl)

    # create list to be used for iterations
    data = [image, 0, [0, 0]]

    # create indexes taking into account subsampling
    idx = [np.tile(np.arange(0, image.shape[0]), (subsmpl, 1)).flatten(),
           np.tile(np.arange(0, image.shape[1]), (subsmpl, 1)).flatten()]

    if weight is None:
        weight = np.ones(image.shape)  # can be input parameter
    # apply weight to image to exclude bad pixels or for flat-field correction
    image_wtd = np.maximum(weight * image, 0)
    data[0] = data[0].take(idx[0], axis=0).take(idx[1], axis=1)
    weight = weight.take(idx[0], axis=0).take(idx[1], axis=1)
    # normalizing constant
    norm_const = np.real(ifftn(otf.conj() * fftn(weight))) + np.sqrt(np.spacing(1))

    if subsmpl != 1:
        vec = np.zeros(len(image.shape) * 2, dtype=np.int)
        vec[np.arange(0, len(image.shape) * 2, 2)] = image.shape
        vec[vec == 0] = subsmpl

    # iterations
    alpha = 0  # acceleration parameter
    for k in range(num_iter):
        if k > 2:
            alpha = np.dot(data[2][0].T, data[2][1]) \
                    / (np.dot(data[2][1].T, data[2][1]) + np.spacing(1))
            alpha = max(min(alpha, 1), 0)  # 0<alpha<1

        # make the estimate for the next iteration and apply positivity constraint
        estimate = np.maximum(data[0] + alpha * (data[0] - data[1]), 0)

        # construct the expected image from the estimate
        reblurred = np.real(ifftn(otf * fftn(estimate)))

        # If subsmpl is not 1, bin reblurred back to original image size by
        # calculating mean
        if subsmpl != 1:
            reblurred = reblurred.reshape(vec)
            for i in np.arange(1, len(image.shape) * 2, 2)[::-1]:
                reblurred = reblurred.mean(axis=i)

        reblurred[reblurred == 0] = np.spacing(1)

        # calculate the ratio of the measured image to the expected image
        ratio = image_wtd / reblurred + np.spacing(1)
        ratio = ratio.take(idx[0], axis=0).take(idx[1], axis=1)

        # determine next estimate and apply positivity constraint
        data[1] = data[0]
        data[0] = np.maximum(estimate * np.real(ifftn(otf.conj() * fftn(ratio))) \
                             / norm_const, 0)
        data[2] = [np.array([np.ravel(data[0] - estimate, order='F')]) \
                       .T, data[2][0]]
    return data[0]


def deconvblind(image, psf, num_iter, weight=None, subsmpl=1):
    # apply blind deconvolution to image

    # create lists to be used for iterations
    data_img = [image, 0, [0, 0]]  # image data
    data_psf = [psf, 0, [0, 0]]  # psf data

    # create indexes taking into account subsampling
    idx = [np.tile(np.arange(0, image.shape[0]), (subsmpl, 1)).flatten(),
           np.tile(np.arange(0, image.shape[1]), (subsmpl, 1)).flatten()]
    idx1 = [np.tile(np.arange(0, psf.shape[0]), (subsmpl, 1)).flatten(),
            np.tile(np.arange(0, psf.shape[1]), (subsmpl, 1)).flatten()]

    if weight is None:
        weight = np.ones(image.shape)  # can be input parameter
    # apply weight to image to exclude bad pixels or for flat-field correction
    image_wtd = np.maximum(weight * image, 0)
    data_img[0] = data_img[0].take(idx[0], axis=0).take(idx[1], axis=1)
    data_psf[0] = data_psf[0].take(idx1[0], axis=0).take(idx1[1], axis=1)
    weight = weight.take(idx[0], axis=0).take(idx[1], axis=1)

    if subsmpl != 1:
        vec = np.zeros(len(image.shape) * 2, dtype=np.int)
        vec[np.arange(0, len(image.shape) * 2, 2)] = image.shape
        vec[vec == 0] = subsmpl

    # iterations
    alpha_img = 0  # acceleration parameter
    alpha_psf = 0  # acceleration parameter
    for k in range(num_iter):
        if k > 1:
            alpha_img = np.dot(data_img[2][0].T, data_img[2][1]) / (np.dot(data_img[2][1].T, data_img[2][1]) + np.spacing(1))
            alpha_img = max(min(alpha_img, 1), 0)  # 0<alpha<1
            alpha_psf = np.dot(data_psf[2][0].T, data_psf[2][1]) / (np.dot(data_psf[2][1].T, data_psf[2][1]) + np.spacing(1))
            alpha_psf = max(min(alpha_psf, 1), 0)  # 0<alpha<1

        # make the image and psf estimate for the next iteration and apply
        # positivity constraint
        estimate_img = np.maximum(data_img[0] + alpha_img * (data_img[0] - data_img[1]), 0)
        estimate_psf = np.maximum(data_psf[0] + alpha_psf * (data_psf[0] - data_psf[1]), 0)
        # normalize psf
        estimate_psf = estimate_psf / (np.sum(estimate_psf)+ (np.sum(estimate_psf) == 0) * np.spacing(1))

        # calculate otf from psf with the same size as image
        otf = psf2otf(estimate_psf, np.array(image.shape) * subsmpl)

        # construct the expected image from the estimate
        reblurred = np.real(ifftn(otf * fftn(estimate_img)))

        # If subsmpl is not 1, bin reblurred back to original image size by
        # calculating mean
        if subsmpl != 1:
            reblurred = reblurred.reshape(vec)
            for i in np.arange(1, len(image.shape) * 2, 2)[::-1]:
                reblurred = reblurred.mean(axis=i)

        reblurred[reblurred == 0] = np.spacing(1)

        # calculate the ratio of the measured image to the expected image
        ratio = image_wtd / reblurred + np.spacing(1)
        ratio = ratio.take(idx[0], axis=0).take(idx[1], axis=1)

        # determine next image estimate and apply positivity constraint
        data_img[1] = data_img[0]
        h1 = psf2otf(data_psf[0], np.array(image.shape) * subsmpl)
        # normalizing constant
        norm_const1 = np.real(ifftn(h1.conj() * fftn(weight))) \
                      + np.sqrt(np.spacing(1))
        data_img[0] = np.maximum(estimate_img * np.real(ifftn(h1.conj() \
                                                              * fftn(ratio))) / norm_const1, 0)
        data_img[2] = [np.array([np.ravel(data_img[0] - estimate_img,
                                          order='F')]).T, data_img[2][0]]

        # determine next psf estimate and apply positivity constraint
        data_psf[1] = data_psf[0]
        h2 = fftn(data_img[1])
        # normalizing constant
        norm_const2 = otf2psf(h2.conj() * fftn(weight), np.array(psf.shape) * subsmpl) + np.sqrt(np.spacing(1))
        data_psf[0] = np.maximum(estimate_psf * otf2psf(h2.conj() * fftn(ratio), np.array(psf.shape) * subsmpl) / norm_const2, 0)
        data_psf[0] = data_psf[0] / (np.sum(data_psf[0]) + (np.sum(data_psf[0]) == 0) * np.spacing(1))
        data_psf[2] = [np.array([np.ravel(data_psf[0] - estimate_psf,
                                          order='F')]).T, data_psf[2][0]]
    return data_img[0], data_psf[0]


def deconvlucy_x(image, psf, num_iter, dampar=0, weight=None, subsmpl=1):
    # apply Richardson-Lucy deconvolution to image

    ## ported from matlab

    eps = np.finfo(float).eps
    J0 = image.copy()
    J0.astype("double")
    PSF = psf
    NUMIT = num_iter
    SUBSMPL = subsmpl
    WEIGHT = weight
    DAMPAR = dampar
    READOUT = 0
    sizeI, sizePSF = np.array(image.shape), np.array(PSF.shape)

    numNSdim = np.nonzero(sizePSF!= 1)[0]
    if np.prod(sizePSF) <2:
        error("[Error]: deconvlucy:psfMustHaveAtLeast2Elements")
    elif np.all(PSF==0):
        error("[Error]: deconvlucy:psfMustNotBeZeroEverywhere")
    elif np.any(sizePSF[numNSdim]/SUBSMPL > sizeI[numNSdim]):
        error("[Error]: deconvlucy:psfMustBeSmallerThanImage")

    J = [J0/65535.0, J0/65535.0, 0, np.zeros((np.prod(sizeI) * SUBSMPL ** len(numNSdim), 2))]
    if isinstance(WEIGHT,(int,float)):
        WEIGHT = np.tile(WEIGHT,sizeI)
    else:
        WEIGHT = np.ones(sizeI)

    sizeOTF = sizeI
    sizeOTF[numNSdim] = SUBSMPL * sizeI[numNSdim]

    H = psf2otf(PSF, sizeOTF)

    idx =[[]]*len(sizeI)
    for k in numNSdim: # index replicates for non-singleton PSF sizes only
        idx[k] = np.reshape(np.tile(np.arange(0, sizeI[k]), (SUBSMPL,1)), (SUBSMPL * sizeI[k],))

    wI = np.maximum(WEIGHT*(READOUT + J[0]), 0)
    J[1] = J[1].take(idx[0], axis=0).take(idx[1], axis=1)
    WEIGHT = WEIGHT.take(idx[0], axis=0).take(idx[1], axis=1)
    # normalizing constant
    scale = np.real(ifftn(H.conj() * fftn(WEIGHT))) + np.sqrt(eps)

    DAMPAR22 = (DAMPAR**2) / 2

    if SUBSMPL != 1:
        vec = np.zeros((len(sizeI) * 2,), dtype='int')
        vec[np.arange(1, len(sizeI) * 2, 2)] = sizeI
        vec[2 * numNSdim] = -1
        vec[vec == 0] = []
        num = np.nonzero(vec == -1)[0][::-1]
        vec[num] = SUBSMPL
    else:
        vec = []
        num = []

    # 3. L_R Iterations
    #
    la = 2 * np.any(J[3].flatten()!=0)
    for k in range(la,NUMIT):
        # 3.a Make an image predictions for the next iteration
        if k > 1:
            la = np.dot(J[3][:,0], J[3][:,1]) / (np.dot(J[3][:,1], J[3][:,1]) + eps)
            la = max(min(la,1),0) # stability enforcement

        Y = np.maximum(J[1] + la*(J[1] - J[2]),0)# plus positivity constraint

        # 3.b  Make core for the LR estimation
        CC = corelucy(Y,H,DAMPAR22,wI,READOUT,SUBSMPL,idx,vec,num)

        # 3.c Determine next iteration image & apply positivity constraint
        J[2] = J[1].copy()
        J[1] = np.maximum(Y*np.real(ifftn(H.conj()*CC))/scale,0)
        J[3] = np.stack([J[1].flatten("F")-Y.flatten("F"),J[3][:,0]]).T

    return np.rint(J[1]*65535.0)

def corelucy(Y,H,DAMPAR22,wI,READOUT,SUBSMPL,idx,vec,num):

    eps = np.finfo(float).eps

    ReBlurred = np.real(ifftn(H * fftn(Y)))
    if SUBSMPL != 1:
        ReBlurred = np.reshape(ReBlurred, vec)

        for k in num: # new appeared singleton.
            vec[k] = []
            ReBlurred = np.reshape(np.mean(ReBlurred, k), vec)

    # 2. An Estimate for the next step
    ReBlurred = ReBlurred + READOUT
    ReBlurred[ReBlurred == 0] = eps
    AnEstim = wI/ReBlurred + eps

    # 3. Damping if needed
    if DAMPAR22 == 0: # No Damping
      ImRatio = AnEstim[idx[0]]
    else: # Damping of the image relative to DAMPAR22 = (N*sigma)^2
      gm = 10;
      g = (wI*np.log(AnEstim)+ ReBlurred - wI)/DAMPAR22
      g = min(g,1)
      G = (g**(gm-1))*(gm-(gm-1)*g)
      ImRatio = 1 + G[idx[0]]*(AnEstim(idx[0]) - 1)

    f = fftn(ImRatio)
    return f






















