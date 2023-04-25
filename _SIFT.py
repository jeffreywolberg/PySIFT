import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
from _SIFT_utils import *
from _keypoints import get_keypoints, localize_keypoints, assign_orientation
from _descriptors import get_local_descriptors
# from PySIFT_repo.

# SIFT orig paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    
# Kernel sizes are hardcoded, seems to be equal to dynamic kernel sizes. Sigma does the heavy lifting it seems
def DoG_fixed_kernel_size(image, num_blurs=5, kernel_sizes=[], sigmas=[]):
    assert num_blurs > 3 and len(kernel_sizes) == num_blurs == len(sigmas)
    out_images = []
    kernel_arrays = []
    
    for i in range(num_blurs):
        sigma = sigmas[i]
        ksize = kernel_sizes[i]
        kernel_arrays.append(get_kernel(ksize, sigma))
        img = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        print(f"Hardcoded kernel size: {ksize}, Sigma: {sigma}")
        out_images.append(img)
    
    diffs = []
    for i in range(len(out_images)-1):
        diff = out_images[i+1].astype(np.int16) - out_images[i].astype(np.int16) # DoG is done here
        # diff = diff.astype(np.float64) * 255 / np.amax(diff)
        diff = diff + abs(np.amin(diff))
        assert np.amax(diff) <= 255
        diffs.append(diff)

    return out_images, diffs

def run_SIFT_cv2(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
    cv2.imshow('Image with Keypoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_SIFT_manual(img):
    num_blurs = 5
    sigmas = [get_sigma(i, root=2.5) for i in range(num_blurs)]
    kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 
    print(f"Sigmas: {sigmas}")
    print(f"Kernel sizes: {kernel_sizes}")

    blurred_images, diffs = DoG_fixed_kernel_size(img, num_blurs, kernel_sizes, sigmas)
    img_2 = reduce_img_size(img, select_every=2)
    blurred_images_2, diffs_2 = DoG_fixed_kernel_size(img_2, num_blurs, kernel_sizes, sigmas)
    img_4 = reduce_img_size(img_2, select_every=2)
    blurred_images_4, diffs_4 = DoG_fixed_kernel_size(img_4, num_blurs, kernel_sizes, sigmas)
    img_8 = reduce_img_size(img_4, select_every=2)
    blurred_images_8, diffs_8 = DoG_fixed_kernel_size(img_8, num_blurs, kernel_sizes, sigmas)

    # for i, d in enumerate(diffs):
        # print(0, len(np.argwhere(d == 0)))
        # print(1, len(np.argwhere(d == 1)))
        # print(255, len(np.argwhere(d == 255)))
        # print("None of the above", len(np.argwhere((d != 0) & (d != 1) & (d != 255))))
    display_DoG(blurred_images, diffs, f"DoG Cv2 1/1 {img.shape}  hardcoded ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    display_DoG(blurred_images_2, diffs_2, f"DoG 1/2 {img_2.shape} Cv2 hardcoded ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    display_DoG(blurred_images_4, diffs_4, f"DoG 1/4 {img_4.shape} Cv2 hardcoded ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    display_DoG(blurred_images_8, diffs_8, f"DoG 1/8 {img_8.shape} Cv2 hardcoded ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    
    key_pts = get_keypoints(diffs, sigmas)
    key_pts_2 = get_keypoints(diffs_2, sigmas)
    key_pts_4 = get_keypoints(diffs_4, sigmas)
    key_pts_8 = get_keypoints(diffs_8, sigmas)

    key_pts_l = localize_keypoints(diffs, key_pts,)
    key_pts_2_l = localize_keypoints(diffs, key_pts_2)
    key_pts_4_l = localize_keypoints(diffs, key_pts_4)
    key_pts_8_l = localize_keypoints(diffs, key_pts_8)

    # show_key_pts(img.copy(),   [(kp[0], kp[1]) for kp in key_pts_l], title=f"Keypoints 1/1 {img.shape}, orig_count={len(key_pts)}, new_count={len(key_pts_l)}")
    # show_key_pts(img_2.copy(), [(kp[0], kp[1]) for kp in key_pts_2_l], title=f"Keypoints 1/2, {img_2.shape}, orig_count={len(key_pts_2)}, new_count={len(key_pts_2_l)}")
    # show_key_pts(img_4.copy(), [(kp[0], kp[1]) for kp in key_pts_4_l], title=f"Keypoints 1/4, {img_4.shape}, orig_count={len(key_pts_4)}, new_count={len(key_pts_4_l)}")
    # show_key_pts(img_8.copy(), [(kp[0], kp[1]) for kp in key_pts_8_l], title=f"Keypoints 1/8, {img_8.shape}, orig_count={len(key_pts_8)}, new_count={len(key_pts_8_l)}")

    key_pts_oriented = assign_orientation(blurred_images, key_pts_l, sigmas, kernel_sizes)
    key_pts_2_oriented = assign_orientation(blurred_images_2, key_pts_2_l, sigmas, kernel_sizes)
    key_pts_4_oriented = assign_orientation(blurred_images_4, key_pts_4_l, sigmas, kernel_sizes)
    key_pts_8_oriented = assign_orientation(blurred_images_8, key_pts_8_l, sigmas, kernel_sizes)

    show_key_pts(img.copy(), [(kp[0], kp[1]) for kp in key_pts_oriented])
    show_key_pts(img.copy(), [(kp[0], kp[1]) for kp in key_pts_2_oriented])
    show_key_pts(img.copy(), [(kp[0], kp[1]) for kp in key_pts_4_oriented])
    show_key_pts(img.copy(), [(kp[0], kp[1]) for kp in key_pts_8_oriented])


    descriptors = get_local_descriptors(key_pts_oriented, blurred_images, sigmas)
    descriptors_2 = get_local_descriptors(key_pts_2_oriented, blurred_images, sigmas)
    descriptors_4 = get_local_descriptors(key_pts_4_oriented, blurred_images, sigmas)
    descriptors_8 = get_local_descriptors(key_pts_8_oriented, blurred_images, sigmas)


if __name__ == "__main__":
    file = "input_images/statue_of_liberty2.jpg"
    # file = 'input_images/house.jpg'
    # file = 'input_images/barn.jpg'
    # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(file)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    print(img.shape)

    # run_SIFT_cv2(img)

    run_SIFT_manual(img)
    quit()



    num_blurs = 5
    sigmas = [get_sigma(i, root=2.75) for i in range(num_blurs)]
    kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 
    print(f"Sigmas: {sigmas}")
    print(f"Kernel sizes: {kernel_sizes}")

    blurred_images_cv2, diffs_cv2 = DoG_fixed_kernel_size(img, num_blurs, kernel_sizes, sigmas)
    display_DoG(blurred_images_cv2, diffs_cv2, f"DoG Cv2 hardcoded ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    plt.show()
    
    kernels = [get_kernel(k_size, sig) for k_size, sig in zip(kernel_sizes, sigmas)]
    blurred_images_conv_float = []
    blurred_images_conv_int = []  
    for i, k in enumerate(kernels):
        k_int = get_integer_array_from_floats(k)

        print(f"Size: {k.shape[0]}, Min: {np.amin(k)}, Max: {np.amax(k)}, 1/Min: {1 / np.amin(k)}, Sigma: {sigmas[i]}")
        print(k)
        print("\n")
        print(k_int)
        print("\n")

        blurred_im_float = manual_convolve(img, k)
        blurred_im_int = manual_convolve(img, k_int, integer_convolution=True) # img is already uint8
        blurred_im_int = (blurred_im_int / int(1/np.amin(k))).astype(np.uint8)

        blurred_images_conv_float.append(blurred_im_float.astype(np.uint8)) # convert to uint8 for visualization later
        blurred_images_conv_int.append(blurred_im_int)
    
    diffs_conv_float = []
    for i in range(len(blurred_images_conv_float)-1):
        diff = blurred_images_conv_float[i+1] - blurred_images_conv_float[i] # DoG is done here
        diffs_conv_float.append(diff)

    diffs_conv_int = []
    for i in range(len(blurred_images_conv_int)-1):
        diff = blurred_images_conv_int[i+1] - blurred_images_conv_int[i] # DoG is done here
        diffs_conv_int.append(diff)

    display_DoG(blurred_images_conv_float, diffs_conv_float, f"DoG Manual Float Blurs, ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    plt.show()

    display_DoG(blurred_images_conv_int, diffs_conv_int, f"DoG Manual Int Blurs, ksizes: {kernel_sizes}, sigmas: {[round(sig, 2) for sig in sigmas]}")
    plt.show()


