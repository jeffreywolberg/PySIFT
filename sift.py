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
from _SIFT_utils import display_DoG, reduce_img_size, get_sigma, get_ksize_from_sigma, manual_convolve, get_kernel

'''
4/24 notes: the algorithm works when the image is read in [0,1] range (float). No outlier points in the sky
The algorithm doesn't yet work when the image is read in in [0, 255] range (int). Runtime errors during execution, fix them, and check if it works. 
Might SIFT only work with floating values? Probably not, but this implementation was geared towards it.
Idea; do all convolutoin in integers, and then when in software convert all ints to floats and run the DoGs through this python code translated in c.

Tested running convolution with image between [0,1] -> works.
Tried running their convolution using an uint8 image and then converting the DoGs to [-1,1] -> does not work

TODO: It seems the floating point convolution is what is making or breaking the SIFT accuracy. Investigate how floating convolution is so different from integer convolution
and then converting back to floats. Is the lack of precision from integer convolution really so important here? Is there another issue after I convert to float that's causing the problem?

4/25 Notes: Works when I do my cv2_conv in float64 and keep the original bounds that it gives me. Proved that it works by convolving with different kern sizes with different sigmas,
as opposed to their approach of using same kernel but convolving on most recently convolved image. 
Need to confirm if i can do manual convolution in floats and have it work, then need to confirm 
if I can do manual convolution in ints, then scale them down to floats, and see if it works. Confirmed that there was no issue with my kernel computation.
Confirmed that there is no issue with my function manual_convolve when doing ops with float64 images and kernels. Need to confirm that it translates properly to [0,255] images and int kernels

4/27 Notes: Tried passing in int kernel and img into manual convolve, only use one kernel size and use the prev blur as the input into the new blur. Then scale output to be [0,255] in 
order to pass it into manual_convolve again (it is much larger when it gets outputted originally because of kernel scaling to reach ints only). Then, before doing DoG scale to [0,1] range such that
DoG is [-1, 1] range. If we were to ever implement one kernel size and reuse previous blur's output for new blur's input, we would need to scale values such that the input into next blur is [0,255]. 
I haven't actually tried this out yet to see if it's necessary, but without scaling, blurs will get very huge, certainly bigger than 32 bit int but maybe even 64.
We can scale in hardware by knowing the max value that the img can take on (1/np.amin(kernel) * 255), but it's more work. The diff kernel sizes may be a better approach.

4/27 Notes: Passed in an int kernel and img into manual_convolve, scaled it down to [0,1] when doing DoG. I used multiple kernels for the diff blur imgs on same octave. This worked.
Had to scale it down by (255 * 1 / np.amin(k)) DoG range [-0.08, 0.07], I remember it being [-0.15, .15] need to confirm
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
        ###### WORKS, THEIR IMPLEMENTATION IN FLOAT64 START ######
        gaussian_pyr, ims = generate_gaussian_pyramid(self.im, self.num_octave, self.s, self.sigma)  # np.uint8 if done with uint8 image
        # DoG_pyr = generate_DoG_pyramid(gaussian_pyr) # range is [-1, 1], in theory
        ###### WORKS, THEIR IMPLEMENTATION IN FLOAT64 END ######

        num_blurs = 5
        k = 2**(1/self.s)
        kernel = gaussian_filter(k * self.sigma)

        ###### DOESN'T WORK, YOU DON'T WANT TO TOUCH THE RANGE/SCALING OF G_BLUR or DOG START ######
        # for i, g_octave in enumerate(gaussian_pyr):
        #     for j, g_blur in enumerate(g_octave):
        #         gaussian_pyr[i][j] = g_blur.astype(np.float64) / np.amax(g_blur)
        #         assert np.amax(gaussian_pyr[i][j]) <= 1
        #         assert np.amin(gaussian_pyr[i][j]) >= 0
        
        # for i, DoG in enumerate(DoG_pyr):
        #     DoG_pyr[i] = DoG.astype(np.float64) / max(np.amax(DoG), abs(np.amin(DoG)))
        #     assert np.amax(DoG_pyr[i]) <= 1
        #     assert np.amin(DoG_pyr[i]) >= -1
        ###### DOESN'T WORK, YOU DON'T WANT TO TOUCH THE RANGE/SCALING OF G_BLUR or DOG END ######
        
        ###### WORKS, PASSING IN INT IMAGE AND INT KERNEL TO MANUAL CONVOLVE, THAN SCALING TO FLOATS FOR DOG START ##########
        # scaling_factor = 1/np.amin(kernel)
        # kernel = (scaling_factor * kernel).astype(np.uint16)
        # im_uint8 = (self.im * 255).astype(np.uint8)

        # img_2 = reduce_img_size(im_uint8, select_every=2)
        # img_4 = reduce_img_size(img_2, select_every=2)
        # img_8 = reduce_img_size(img_4, select_every=2)

        # # gaussian_pyr dims: [num_octaves, num_blurs, num_rows, num_cols]
        # gaussian_pyr = [np.zeros((int(self.s+3), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # # DoG_pyr expected dims: [num_octaves, num_rows, num_cols, num_blurs-1]. In initialization below I don't give it proper shape, I use .transpose later
        # DoG_pyr = [np.zeros((int(self.s+2), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale)), dtype=np.float64) for scale in [1,2,4,8]]
        # for i, img in enumerate([im_uint8, img_2, img_4, img_8]):
        #     for j in range(num_blurs):
        #         gaussian_pyr[i][j, :, :] = manual_convolve((255 * gaussian_pyr[i][j-1, :, :] / np.amax(gaussian_pyr[i][j-1, :, :])).astype(np.uint8) if j > 0 else img, kernel, integer_convolution=True)
        #         gaussian_pyr[i][j, :, :] /= np.amax(gaussian_pyr[i][j, :, :]) # bring to [0,1]
        #         if j > 0:
        #             DoG_pyr[i][j-1, :, :] = gaussian_pyr[i][j, :, :] - gaussian_pyr[i][j-1, :, :]
        #             x = 0
        #     DoG_pyr[i] = DoG_pyr[i].transpose(1,2,0)  
        ###### WORKS, PASSING IN INT IMAGE AND INT KERNEL TO MANUAL CONVOLVE, THAN SCALING TO FLOATS FOR DOG END ########## 

        ###### WORKS, 4/27 int kernel, int img, convolve with different kernel sizes and sigmas, scale to float before DoG START ######
        kernel = None 
        im_uint8 = (self.im * 255).astype(np.uint8)
        
        img_2 = reduce_img_size(im_uint8, select_every=2)
        img_4 = reduce_img_size(img_2, select_every=2)
        img_8 = reduce_img_size(img_4, select_every=2)

        sigmas = [get_sigma(i, root=2.5) for i in range(num_blurs)]
        kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 
        kernels = [get_kernel(size, sig) for size, sig in zip(kernel_sizes, sigmas)]
        scaling_factors = [1 / np.amin(k) for k in kernels]
        kernels = [(k * scale).astype(np.uint32) for k, scale in zip(kernels, scaling_factors)]
        k_over_16_bit = [np.any(k > 2**16-1) for k in kernels] ## kernel is over 16 bit, nee 32 bit representation with sigma root = 2.5
        print(k_over_16_bit)

        # gaussian_pyr dims: [num_octaves, num_blurs, num_rows, num_cols]
        gaussian_pyr = [np.zeros((int(self.s+3), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # DoG_pyr expected dims: [num_octaves, num_rows, num_cols, num_blurs-1]. In initialization below I don't give it proper shape, I use .transpose later
        DoG_pyr = [np.zeros((int(self.s+2), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale)), dtype=np.float64) for scale in [1,2,4,8]]
        for i, img in enumerate([im_uint8, img_2, img_4, img_8]):
            for j, (kernel, scl_factor) in enumerate(zip(kernels, scaling_factors)):
                gaussian_pyr[i][j, :, :] = manual_convolve(img, kernel, integer_convolution=True) # [0, SOME_LARGE_INT_VALUE]
                gaussian_pyr[i][j, :, :] = gaussian_pyr[i][j, :, :] / scl_factor / 255 # [0, 1]
                if j > 0:
                    DoG_pyr[i][j-1, :, :] = gaussian_pyr[i][j, :, :] - gaussian_pyr[i][j-1, :, :]
                    x = 0
            DoG_pyr[i] = DoG_pyr[i].transpose(1,2,0)   
        ###### WORKS, 4/27 int kernel, int img, convolve with different kernel sizes and sigmas, scale to float before DoG END ######
        
        # ###### WORKS, convolve with different kernel sizes and sigmas START ######
        # img_2 = reduce_img_size(self.im, select_every=2)
        # img_4 = reduce_img_size(img_2, select_every=2)
        # img_8 = reduce_img_size(img_4, select_every=2)

        # sigmas = [get_sigma(i, root=2.5) for i in range(num_blurs)]
        # kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 
        # blurred_images, diffs = DoG_fixed_kernel_size(self.im, num_blurs, kernel_sizes, sigmas)
        # blurred_images_2, diffs_2 = DoG_fixed_kernel_size(img_2, num_blurs, kernel_sizes, sigmas)
        # blurred_images_4, diffs_4 = DoG_fixed_kernel_size(img_4, num_blurs, kernel_sizes, sigmas)
        # blurred_images_8, diffs_8 = DoG_fixed_kernel_size(img_8, num_blurs, kernel_sizes, sigmas)

        # gaussian_pyr = [np.zeros((int(self.s+3), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # DoG_pyr = [np.zeros((int(self.s+2), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # for i, (blur_ims, diffs) in enumerate(zip([blurred_images, blurred_images_2, blurred_images_4, blurred_images_8], [diffs, diffs_2, diffs_4, diffs_8])):
        #     gaussian_pyr[i] = np.array(blur_ims).reshape(len(blur_ims), *blur_ims[0].shape)
        #     DoG_pyr[i] = np.array(diffs).transpose(1,2,0)
        # ###### WORKS, convolve with different kernel sizes and sigmas END ######

        
        ###### WORKS, confirmed that manual_conolve gets desired result with float img and float kernel START ######
        # img_2 = reduce_img_size(self.im, select_every=2)
        # img_4 = reduce_img_size(img_2, select_every=2)
        # img_8 = reduce_img_size(img_4, select_every=2)

        # # gaussian_pyr dims: [num_octaves, num_blurs, num_rows, num_cols]
        # gaussian_pyr = [np.zeros((int(self.s+3), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale))) for scale in [1,2,4,8]]
        # # DoG_pyr expected dims: [num_octaves, num_rows, num_cols, num_blurs-1]. In initialization below I don't give it proper shape, I use .transpose later
        # DoG_pyr = [np.zeros((int(self.s+2), int(self.im.shape[0] / scale), int(self.im.shape[1] // scale)), dtype=np.float64) for scale in [1,2,4,8]]
        # for i, img in enumerate([self.im, img_2, img_4, img_8]):
        #     for j in range(num_blurs):
        #         gaussian_pyr[i][j, :, :] = manual_convolve(gaussian_pyr[i][j-1, :, :] if j > 0 else img, kernel, integer_convolution=False)
        #         if j > 0:
        #             DoG_pyr[i][j-1, :, :] = gaussian_pyr[i][j, :, :] - gaussian_pyr[i][j-1, :, :]
        #     DoG_pyr[i] = DoG_pyr[i].transpose(1,2,0)   
        ###### WORKS, confirmed that manual_conolve gets desired result with float img and float kernel END ######
        
        x = 0
        for i in range(len(gaussian_pyr)):
            gaussian_octave = gaussian_pyr[i]
            ###### DOESN'T WORK, DON'T TOUCH SCALING/RANGE START ######
            # DoG_pyr[i] = (DoG_pyr[i].astype(np.float64) * 255.0/np.amax(DoG_pyr[i])).astype(np.uint8)
            # DoG_octave = (DoG_pyr[i].astype(np.float64) * 255.0/np.amax(DoG_pyr[i])).astype(np.uint8) # uint16, [0,45 around]
            ###### DOESN'T WORK, DON'T TOUCH SCALING/RANGE END ######
            DoG_octave = DoG_pyr[i]
            im = ims[i]
            # display_DoG(gaussian_octave, [DoG_octave[:,:,i].reshape(1, *DoG_octave.shape[:2]).squeeze() for i in range(DoG_octave.shape[2])], f"DoG skimage {im.shape}")

        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
