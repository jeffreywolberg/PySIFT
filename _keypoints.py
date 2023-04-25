import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math
from _SIFT_utils import clamp, get_kernel, cart_to_polar_grad, quantize_orientation

def neighborhood(img, row, col):
    '''
    [[0, 1, 2],
     [3, 4, 5],  
     [6, 7, 8]]    ---->    [0,1,2,3,4,5,6,7,8]
    '''
    assert row >= 1 and col >= 1 and row < img.shape[0]-1 and col < img.shape[1]-1
    return list(img[row-1:row+2, col-1:col+2].flatten())

def get_keypoints(diffs, sigmas):
        n = len(diffs)
        key_pts = []
        for i in range(n-2):
            key_pts += _get_keypoints(diffs[i], diffs[i+1], diffs[i+2], sigmas[i+1])  # list addition (e.g. [0] + [1,2] = [0,1,2])
        return list(set(key_pts))

def _get_keypoints(img_a, img_b, img_c, sigma):
    keypoints = []
    for row in range(2, img_b.shape[0]-2):
        for col in range(2, img_b.shape[1]-2):
            pix = img_b[row, col]
            img_a_neigbs = neighborhood(img_a, row, col)
            img_b_neighbs = neighborhood(img_b, row, col)
            img_b_neighbs.pop(4) # remove current pix
            img_c_neighbs = neighborhood(img_c, row, col)
            all_neighbs = img_a_neigbs + img_b_neighbs + img_c_neighbs
            if pix > max(all_neighbs) or pix < min(all_neighbs):        # should it be >= or >?
                keypoints.append((row, col, sigma))
    return keypoints

def get_pD_pRow(DoG, row, col):  # partial D wrt partial r
    return (DoG[row+1, col] - DoG[row-1, col]) / 2

def get_pD_pCol(DoG, row, col):  # partial D wrt partial c
     return (DoG[row, col+1] - DoG[row, col-1]) / 2

def get_pD_pSigma(DoG_a, DoG_b, row, col): # partial D wrt partial s
    '''
    DoG_a: G_blur(sigma^4) - G_blur(sigma^3)
    DoG_b: G_blur(sigma^2) - G_blur(sigma)
    '''
    return (DoG_a[row, col] - DoG_b[row, col]) / 2

def get_DoG_jacobian(diffs, row, col):
     jacobian = np.zeros((3,))  # [partial_D/partial_row, partial_D/partial_col, partial_D/partial_sigma]
     second_DoG = diffs[1] # G_blur(sigma^3) - G_blur(sigma^2)
     jacobian[0] = get_pD_pRow(second_DoG, row, col)  
     jacobian[1] = get_pD_pCol(second_DoG, row, col)
     jacobian[2] = get_pD_pSigma(diffs[2], diffs[0], row, col)
     return jacobian

def get_DoG_hessian(diffs, row, col):
     DoG = diffs[1] # G_blur(sigma^3) - G_blur(sigma^2)
     d_rr = (DoG[row+1, col] - DoG[row, col]) - (DoG[row, col] - DoG[row-1, col])
     d_rc = ((DoG[row+1, col+1] - DoG[row-1, col+1]) - (DoG[row+1, col-1] - DoG[row-1, col-1])) / 4
     d_rs = ((diffs[2][row+1, col] - diffs[2][row-1, col]) - (diffs[0][row+1, col] - diffs[0][row-1, col])) / 4
     d_cc = (DoG[row, col+1] - DoG[row, col]) - (DoG[row, col] - DoG[row, col-1])
     d_cs = (diffs[2][row, col+1] - diffs[2][row, col-1]) - (diffs[0][row, col+1] - diffs[0][row, col-1]) / 4
     d_ss = (diffs[2][row, col] - diffs[1][row, col]) - (diffs[1][row, col] - diffs[0][row, col])
     hessian = np.array([ [d_rr, d_rc, d_rs], [d_rc, d_cc, d_cs], [d_rs, d_cs, d_ss]])
     return hessian

def get_offset(diffs, row, col):
    DoG_jacobian = get_DoG_jacobian(diffs, row, col)
    DoG_hessian = get_DoG_hessian(diffs, row, col)
    return -LA.inv(DoG_hessian) @ DoG_jacobian

def get_new_row_col_from_offset(offset, row, col):
    move = np.round(offset[:2], 0) # [-.4, -1.1] -> [0, -1] 
    new_r = row + move[0]
    new_c = col + move[1]
    return int(new_r), int(new_c)

def localize_keypoints(diffs, keypoints):
     '''
     diffs: list of DoGs (should be 4 per octave)
     keypoints: list of (row, col) kps
     sigma: the sigma corresponding to diffs[1] (should be sigmas[1])
     This link explains the math: https://dsp.stackexchange.com/questions/10403/sift-taylor-expansion

     '''
     localized_kps = []
     diffs = [d.astype(np.int16) for d in diffs]
     second_DoG_float = diffs[1].astype(np.float64) / 255  # [0, 1] range
     for (row, col, sigma) in keypoints:
          try:
            h = get_offset(diffs, row, col) # given row, col, a guess at the detla btwn where the actual kp lies 
          except LA.LinAlgError:
               continue
          if np.any(np.abs(h[:3]) >= 1):  # if a new guess of the kp is very diff from original, don't include kp
                # print(f"Discarding kp {row, col}")
                continue
          extrema_val = second_DoG_float[row, col] + .5 * get_DoG_jacobian(diffs, row, col).T @ h
          if abs(extrema_val) > 1.0:  # TODO: how to finetune this threshold? Not clear
               continue
        #   print(f"Keeping kp {row, col}, h is {h}, extrema val: {extrema_val}")
        #   localized_kps.append( (row, col) )
        #   print(sigma, h[2])
          localized_kps.append( (row + h[0], col + h[1], sigma + h[2]))
    
     return localized_kps

def get_mag_and_ori(im, row, col):
    h, w = im.shape
    row, col = clamp(int(row), 0, h-1), clamp(int(col), 0, w-1)
    d_row = im[min(h-1, row+1), col].astype(np.int16) - im[max(0, row-1), col]
    d_col = im[row, min(w-1, col+1)].astype(np.int16) - im[row, max(0, col-1)]
    mag, ang = cart_to_polar_grad(d_row, d_col)  # cvt to degrees
    return mag, ang

def fit_parabola(hist, bin_no, bin_width): 
    centerval = bin_no*bin_width + bin_width/2. 
    rightval = (bin_no+1)*bin_width + bin_width/2. 
    if bin_no == 0: leftval = -bin_width/2. 
    else: leftval = (bin_no-1)*bin_width + bin_width/2. 
    A = np.array([ 
      [centerval**2, centerval, 1], 
      [rightval**2, rightval, 1], 
      [leftval**2, leftval, 1]]) 
    b = np.array([ 
      hist[bin_no], 
      hist[(bin_no+1)%len(hist)], 
      hist[(bin_no-1)%len(hist)]]) 
    x = LA.lstsq(A, b, rcond=None)[0] 
    if x[0] == 0: x[0] = 1e-6 
    return -x[1]/(2*x[0])

def assign_orientation(blurred_ims, kps, sigmas, kernel_sizes):
     assert len(blurred_ims) == len(sigmas) == len(kernel_sizes)

     new_kps = []
     num_bins = 36
     bin_width = 360//num_bins
     sigmas = np.array(sigmas)
     hist = np.zeros(num_bins)
     for i, (row, col, kp_sigma) in enumerate(kps):
        assert kp_sigma > 0
        assert kp_sigma < np.ceil(sigmas[-1])  # Not a hard rule, just want to make sure sigma is not too out of hand

        # Extract im which is closest to the sigma of the kp
        sigma_idx = np.argmin(np.abs(sigmas - kp_sigma))
        im = blurred_ims[sigma_idx]

        k_size = kernel_sizes[sigma_idx]
        k_sigma = sigmas[sigma_idx]
        kernel = get_kernel(k_size, k_sigma)

        for offset_r in range(-k_size//2, k_size//2 +1):
            for offset_c in range(-k_size//2, k_size//2 +1):
                m, theta = get_mag_and_ori(im, row + offset_r, col + offset_c)
                assert not np.isnan(m)
                assert not np.isnan(theta)
                weight = kernel[offset_r + k_size//2, offset_c + k_size//2] * m
                bin_no = quantize_orientation(theta, num_bins)
                hist[bin_no] += weight
        
        max_bin = np.argmax(hist) 
        new_kps.append([row, col, kp_sigma, fit_parabola(hist, max_bin, bin_width)]) 
        max_val = np.max(hist) 
        for bin_no, val in enumerate(hist): 
          if bin_no == max_bin: continue 
          if .8 * max_val <= val: 
            new_kps.append([row, col, kp_sigma, fit_parabola(hist, bin_no, bin_width)])
     
     return new_kps





    






