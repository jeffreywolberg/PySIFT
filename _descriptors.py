import numpy as np
from _SIFT_utils import cart_to_polar_grad, quantize_orientation
import numpy.linalg as LA


def get_patch_grads(p):
    r1 = np.zeros_like(p)
    r1[-1] = p[-1]
    r1[:-1] = p[1:]
    r2 = np.zeros_like(p)
    r2[0] = p[0]
    r2[1:] = p[:-1]
    d_row = r1-r2
    r1[:, -1] = p[:, -1]
    r1[:, :-1] = p[:, 1:]
    r2[:, 0] = p[:, 0]
    r2[:, 1:] = p[:, :-1]
    d_col = r1-r2
    return d_row, d_col


def get_local_descriptors(kps, blurred_images, sigmas, w_size=16, num_subregion=4, num_bin=8):
    descs = []
    bin_width = 360//num_bin
    for kp in kps:
        row, col, s = int(kp[0]), int(kp[1]), int(kp[2])
        sigma_id_col = np.argmin(np.abs(sigmas - s))
        im = blurred_images[sigma_id_col]
        h, w = im.shape
        t, l = max(0, row-w_size//2), max(0, col-w_size//2)
        b, r = min(h, row+w_size//2+1), min(w_size, col+w_size//2+1)
        patch = im[t:b, l:r].astype(np.int16)
        d_row, d_col = get_patch_grads(patch)
        if d_col.shape[0] < w_size+1:
            if t == 0:
                kernel = kernel[kernel.shape[0]-d_col.shape[0]:]
            else:
                kernel = kernel[:d_col.shape[0]]
        if d_col.shape[1] < w_size+1:
            if l == 0:
                kernel = kernel[kernel.shape[1]-d_col.shape[1]:]
            else:
                kernel = kernel[:d_col.shape[1]]
        if d_row.shape[0] < w_size+1:
            if t == 0:
                kernel = kernel[kernel.shape[0]-d_row.shape[0]:]
            else:
                kernel = kernel[:d_row.shape[0]]
        if d_row.shape[1] < w_size+1:
            if l == 0:
                kernel = kernel[kernel.shape[1]-d_row.shape[1]:]
            else:
                kernel = kernel[:d_row.shape[1]]
        m, theta = cart_to_polar_grad(d_row, d_col)
        d_row, d_col = d_row*kernel, d_col*kernel,
        subregion_w = w_size//num_subregion
        featvec = np.zeros(num_bin * num_subregion**2, dtype=np.float32)
        for i in range(0, subregion_w):
            for j in range(0, subregion_w):
                t, l = i*subregion_w, j*subregion_w
                b, r = min(
                    h, (i+1)*subregion_w), min(w, (j+1)*subregion_w)
                hist = get_histogram_for_subregion(m[t:b, l:r].flatten(), theta[t:b, l:r].flatten(), num_bin, kp[3], bin_width, subregion_w)
                featvec[i*subregion_w*num_bin + j*num_bin:i *
                        subregion_w*num_bin + (j+1)*num_bin] = hist.flatten()
        featvec /= max(1e-6, LA.norm(featvec))
        featvec[featvec > 0.2] = 0.2
        featvec /= max(1e-6, LA.norm(featvec))
        descs.append(featvec)
    return np.array(descs)


def get_histogram_for_subregion(m, theta, num_bin, reference_angle, bin_width, subregion_w):
    hist = np.zeros(num_bin, dtype=np.float32)
    c = subregion_w/2 - .5
    for i, (mag, angle) in enumerate(zip(m, theta)):
        angle = (angle-reference_angle) % 360
        bin_no = quantize_orientation(angle, num_bin)
        vote = mag

        hist_interp_weight = 1 - \
            abs(angle - (bin_no*bin_width + bin_width/2))/(bin_width/2)
        vote *= max(hist_interp_weight, 1e-6)
        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)
        vote *= x_interp_weight * y_interp_weight
        hist[bin_no] += vote
    hist /= max(1e-6, LA.norm(hist))
    hist[hist > 0.2] = 0.2
    hist /= max(1e-6, LA.norm(hist))
    return hist
