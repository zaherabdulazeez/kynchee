# Author: Zaher Abdul Azeez 
# Tagline: you can say, I am mad enough

# Date: 2019-03-05
# Subject: modules methods for trimming screenshots to main images

"""
Trim a screenshot into the main image.
"""

import numpy as np 
from matplotlib.image import imread
from scipy.misc import imsave
from scipy.signal import argrelextrema
from skimage.color import rgb2gray

import logging 
logger = logging.getLogger(__name__)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=lininflections(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def trimmed_image_array(img_arr):
	"""
	From the img_arr(gray_scale), get the boundaries of the actual image

	Parameters:
	img_arr: gray_scale image numpy array
	"""
	gray_scale = rgb2gray(img_arr)

	gr0, gr1 = np.gradient(gray_scale)

	# gradient aggregate 1D arrays
	gr0s = np.abs(np.sum(gr0, axis=1))
	gr1s = np.abs(np.sum(gr1, axis=0))

	gr0s = smooth(gr0s)
	gr1s = smooth(gr1s)

	bounds_0 = get_image_boundaries(gr0s)
	bounds_1 = get_image_boundaries(gr1s)

	if not bounds_0 and not bounds_1:
		logger.info("image cannot be cropped as it has no blank spaces")
		return None

	if bounds_0:
		img_arr = img_arr[bounds_0[0]:bounds_0[1], :]
	if bounds_1:
		img_arr = img_arr[:, bounds_1[0]:bounds_1[1]]

	return img_arr

def get_image_boundaries(gradient, window_size = 15, min_image_size=100):
	"""
	from a gradient aggregate 1D array, get the indices where the image boundaries start

	a window of certain size is slid over the entire array from the beginning to the end. and inflection points are
	defined as those where the the dot product of the kernel and window switches from 0 to nonzero and vice versa

	Parameters:
	==========
	window_size: min size of the image
	min_image_size: min pixel size of the image. only sections of width above this will be considered
	"""
	if window_size > len(gradient):
		raise ValueError("window_size must be less than size of gradient array")

	conv = np.ones(window_size)
	inflections = []
	i = 0

	while True:
		kernel = gradient[i:i+window_size]
		dot = np.dot(kernel, conv)
		if dot == 0:
			inflections.append(i)
			non_zero = np.where(gradient[i:] != 0)[0]
			if len(non_zero) > 0:
				non_zero = non_zero[0]
				inflections.append(i+non_zero)
				i += non_zero
			else:
				inflections.append(len(gradient)-1)
				break
		else:
			i += 1

		if i >= len(gradient) - window_size:
			inflections.append(len(gradient)-1)
			break

	if not inflections:
		logger.info("no inflections found")
		return None
	if 0 in inflections:
		sections = [(inflections[i], inflections[i+1]) for i in range(len(inflections) - 1)]
		img_sections = [sections[k] for k in range(len(sections)) if k%2==1]
		img_sections = [sec for sec in img_sections if sec[1] - sec[0] >= min_image_size]
	else:
		inflections = [0] + inflections
		sections = [(inflections[i], inflections[i+1]) for i in range(len(inflections) - 1)]
		img_sections = [sections[k] for k in range(len(sections)) if k%2==0]
		img_sections = [sec for sec in img_sections if sec[1] - sec[0] >= min_image_size]

	if not img_sections:
		logger.info("no sufficiently long image sections found")
		return None

	img = max(img_sections, key=lambda x: x[1]-x[0])

	return img

def crop_image(source_image, target_image):
	"""
	crop source_image and write to target_image
	"""
	img_arr = imread(source_image)

	trimmed_image = trimmed_image_array(img_arr)

	if not np.any(trimmed_image):
		logger.info("not a screenshot; not saving the screenshot")
		return None

	imsave(target_image, trimmed_image)










