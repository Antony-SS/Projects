#!/usr/bin/python3

import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    filter_pad_h = filter.shape[0] // 2
    filter_pad_w = filter.shape[1] // 2

    padded_filter = np.pad(filter, 
                          ((filter_pad_h, image.shape[0] - filter.shape[0] - filter_pad_h),
                           (filter_pad_w, image.shape[1] - filter.shape[1] - filter_pad_w)),
                          mode='constant', constant_values=0) # make filter the size of the image by 0 padding
    
    freq_domain_image = np.fft.fft2(a = image)
    freq_domain_filter = np.fft.fft2(a = padded_filter)

    image_freq = freq_domain_image
    filter_freq = freq_domain_filter

    conv_result_freq = filter_freq*image_freq

    conv_result = np.real(np.fft.ifft2(conv_result_freq))
    

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, conv_result_freq, conv_result 


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    

    filter_pad_h = filter.shape[0] // 2
    filter_pad_w = filter.shape[1] // 2


    # now just repeat steps
    padded_filter = np.pad(filter, 
                          ((filter_pad_h, image.shape[0] - filter.shape[0] - filter_pad_h),
                           (filter_pad_w, image.shape[1] - filter.shape[1] - filter_pad_w)),
                          mode='constant', constant_values=0) # make filter the size of the image by 0 padding
    

    freq_domain_image = np.fft.fft2(a = image)
    filter_freq = np.fft.fft2(a = padded_filter)
    

    image_freq = freq_domain_image
    inverse_filter_freq = 1 / filter_freq

    deconv_result_freq = inverse_filter_freq*image_freq

    deconv_result = np.real(np.fft.ifft2(deconv_result_freq))
    

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, deconv_result_freq, deconv_result





