import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import skimage.draw
import copy

def show_im(img, title="Img"):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def show_key_pts(img, key_pts, title="Img"):
    for (row, col,) in key_pts:
        if isinstance(row, int):
            cv2.circle(img, (col, row), 2, (255,), 2) # switch (r, c) it for cv2
        else:
            img[skimage.draw.disk((row, col), radius=2)] = 255
    print(title)
    show_im(img, title)

def get_gaussian_coefficient(x, y, sigma):
    return (1 / (2*math.pi * sigma ** 2)) * math.e ** ( -(x**2 + y**2) / (2 * sigma ** 2))  # lambda (the 2 coefficient in the denominator of the exponent) significantly affects the blurring and brightness)

def display_DoG(blurred_images, diffs, title="DoG"):

    n_blur = len(blurred_images)
    n_diffs = n_blur - 1
    assert n_diffs == len(diffs)

    fig, axes = plt.subplots(nrows=2, ncols=n_blur, figsize=(4*n_blur, 4))
    fig.suptitle(title)
    for i in range(n_blur):
        axes[0, i].imshow(blurred_images[i], cmap='gray', vmin=-1 if np.amin(blurred_images[i]) < 0 else 0, vmax= 1 if np.amax(blurred_images[i]) <= 1 else 255)
        axes[0, i].set_axis_off()
    for i in range(n_diffs):
        diff = diffs[i]
        if np.amin(diffs[i]) < 0 and np.amax(diffs[i] > 1): # an integer image
            diff = diff.copy() # don't tamper with original DoG
            diff += abs(np.amin(diff)) # add shift so that pixel range is [0, 255]
            if np.amax(diff) > 255:
                diff = (diff.astype(np.float64) * 255 / np.amax(diff)).astype(np.uint8)
        if np.amax(diff) > 1:
            diff = diff.astype(np.uint8)
        axes[1, i].imshow(diff, cmap='gray', vmin= -1 if np.amin(diff[i]) < 0 else 0, vmax= 1 if np.amax(diff[i]) <= 1 else 255)
        axes[1, i].set_axis_off()
    plt.show()

def get_kernel(k_size, sigma):
    size = get_ksize_from_sigma(sigma)
    assert k_size == size, "The size inputted into get_kernel does not match up with the kernel size calculated from sigma"
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()
    
    # size = int(size)
    # assert size % 2 == 1
    # kernel = np.zeros((size, size))
    # min_idx = -int(size / 2) # min idx when center is (0, 0)  (e.g. -5 for size 11)
    # max_idx = int(size / 2)  # max idx when center is (0, 0)  (e.g. 5 for size 11)
    # for y in range(min_idx, max_idx  +1):
    #     for x in range(min_idx, max_idx  +1):
    #         kernel[y - min_idx, x - min_idx] = get_gaussian_coefficient(x, y, sigma)
    # return kernel

def get_sigma(i, root=2):  # (i is zero indexed)
    k = 1
    return k * 2**((i+1)/root)

def get_ksize_from_sigma(sigma):
    # https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/smooth.dispatch.cpp#L289
    return int(6 * sigma + 1) | 1

def reduce_img_size(img:np.ndarray, select_every=2):
    return img[1::select_every, 1::select_every]

def get_integer_array_from_floats(kernel):
    scaling_factor = 1 / np.amin(kernel)
    new_kernel = (scaling_factor * kernel)
    if np.any(new_kernel > 2 ** 32 -1): 
        print(f"Scaled kernel will have the following values, will overflow... {new_kernel[new_kernel > 2**32 -1]}")
        raise NotImplementedError
    else:
        new_kernel = new_kernel.astype(np.uint32)
    return new_kernel

def convolve(im, kern, row, col, pad_size, to_int=False):
    # row, col = center indices of image where convolving window is centered
    assert (len(kern) % 2 == 1) and (len(kern) // 2 == pad_size)
    mul = np.multiply(im[row-pad_size : row+pad_size+1, col-pad_size : col+pad_size+1], kern, dtype=np.uint64 if to_int else np.float64)
    return np.sum(mul)

def manual_convolve(image, kernel, integer_convolution=False):
    '''
    if [integer_convolution], image is a uint8 image and kernel is integers
    '''
    if integer_convolution:
        assert image.dtype == np.uint8 and (kernel.dtype == np.uint8 or kernel.dtype == np.uint16 or kernel.dtype == np.uint32), "Image and kernel must be uint arrays"
        
    # Pad zeros to border of image
    pad_size = kernel.shape[0] // 2
    padded_image = np.zeros((image.shape[0] + 2 * pad_size, image.shape[1] + 2 * pad_size), dtype=image.dtype)
    padded_image[pad_size: padded_image.shape[0] - pad_size, pad_size: padded_image.shape[1] - pad_size] = image

    new_image = np.zeros(image.shape, dtype=np.uint64 if integer_convolution else np.float64)
    for y in range(new_image.shape[0]):
        for x in range(new_image.shape[1]):
            output_pix = convolve(padded_image, kernel, y+pad_size, x+pad_size, pad_size, to_int=integer_convolution)
            if output_pix > 2 ** 64 - 1:
                print(f"Overflowing! [{y}][{x}]: {output_pix}")
                raise OverflowError
            new_image[y, x] = output_pix
    
    return new_image

def cart_to_polar_grad(d_row, d_col): 
  if d_col == 0:
      return 0, 0
  m = np.sqrt(d_row**2 + d_col**2) 
  theta = (np.arctan(d_row / d_col) + np.pi) * 180/np.pi 
  return m, theta 

def quantize_orientation(theta, num_bins): 
  bin_width = 360//num_bins 
  return int(theta//bin_width)

def clamp(x, l, h):
    if x < l:
        return l
    elif x > h:
        return h
    return x


    



