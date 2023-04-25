from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import numpy as np
import os
import sys

from gaussian_filter import gaussian_filter
from gaussian_pyramid import generate_gaussian_pyramid
from DoG_pyramid import generate_DoG_pyramid
from keypoints import get_keypoints
from orientation import assign_orientation
from descriptors import get_local_descriptors

from _SIFT import DoG_fixed_kernel_size
from _SIFT_utils import display_DoG, reduce_img_size, get_sigma, get_ksize_from_sigma 

'''
2/24 notes: the algorithm works when the image is read in [0,1] range (float). No outlier points in the sky
The algorithm doesn't yet work when the image is read in in [0, 255] range (int). Runtime errors during execution, fix them, and check if it works. 
Might SIFT only work with floating values? Probably not, but this implementation was geared towards it.
Idea; do all convolutoin in integers, and then when in software convert all ints to floats and run the DoGs through this python code translated in c.

Tested running convolution with image between [0,1] -> works.
Tried running their convolution using an uint8 image and then converting the DoGs to [-1,1] -> does not work

TODO: It seems the floating point convolution is what is making or breaking the SIFT accuracy. Investigate how floating convolution is so different from integer convolution
and then converting back to floats. Is the lack of precision from integer convolution really so important here? Is there another issue after I convert to float that's causing the problem?
'''

class SIFT(object):
    def __init__(self, im, s=2, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussian_filter(s0))
        # self.im = (rgb2gray(im) * 255).astype(np.uint8)
        # self.im = (convolve(rgb2gray(im), gaussian_filter(s0)) * 255).astype(np.uint8)
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr, ims = generate_gaussian_pyramid(self.im, self.num_octave, self.s, self.sigma)  # np.uint8 if done with uint8 image
        
        # for i, g_octave in enumerate(gaussian_pyr):
        #     for j, g_blur in enumerate(g_octave):
        #         gaussian_pyr[i][j] = g_blur.astype(np.float64) / np.amax(g_blur)
        #         assert np.amax(gaussian_pyr[i][j]) <= 1
        #         assert np.amin(gaussian_pyr[i][j]) >= 0
        
        DoG_pyr = generate_DoG_pyramid(gaussian_pyr) # range is [-1, 1], in theory
        # for i, DoG in enumerate(DoG_pyr):
        #     DoG_pyr[i] = DoG.astype(np.float64) / max(np.amax(DoG), abs(np.amin(DoG)))
        #     assert np.amax(DoG_pyr[i]) <= 1
        #     assert np.amin(DoG_pyr[i]) >= -1

        # num_blurs = 5
        # sigmas = [get_sigma(i, root=2.5) for i in range(num_blurs)]
        # kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 

        # blurred_images, diffs = DoG_fixed_kernel_size(self.im, num_blurs, kernel_sizes, sigmas)
        # img_2 = reduce_img_size(self.im, select_every=2)
        # blurred_images_2, diffs_2 = DoG_fixed_kernel_size(img_2, num_blurs, kernel_sizes, sigmas)
        # img_4 = reduce_img_size(img_2, select_every=2)
        # blurred_images_4, diffs_4 = DoG_fixed_kernel_size(img_4, num_blurs, kernel_sizes, sigmas)
        # img_8 = reduce_img_size(img_4, select_every=2)
        # blurred_images_8, diffs_8 = DoG_fixed_kernel_size(img_8, num_blurs, kernel_sizes, sigmas)

        # # gaussian_pyr = [np.zeros((int(self.s+3), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # # DoG_pyr = [np.zeros((int(self.s+2), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # # for i, (blur_ims, diffs) in enumerate(zip([blurred_images, blurred_images_2, blurred_images_4, blurred_images_8], [diffs, diffs_2, diffs_4, diffs_8])):
        # #     gaussian_pyr[i] = np.array(blur_ims).reshape(len(blur_ims), *blur_ims[0].shape)
        # #     DoG_pyr[i] = np.array(diffs).transpose(1,2,0)

        # ims = [self.im, img_2, img_4, img_8]
        # for i in range(len(gaussian_pyr)):
        #     gaussian_octave = gaussian_pyr[i]
        #     DoG_pyr[i] = (DoG_pyr[i].astype(np.float64) * 255.0/np.amax(DoG_pyr[i])).astype(np.uint8)
        #     DoG_octave = DoG_pyr[i] # uint16, [0,45 around]
        #     im = ims[i]
        #     display_DoG(gaussian_octave, [DoG_octave[:,:,i].reshape(1, *DoG_octave.shape[:2]).squeeze() for i in range(DoG_octave.shape[2])], f"DoG skimage {im.shape}", zero_to_one=True if np.amax(im) <= 1 else False)

        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
