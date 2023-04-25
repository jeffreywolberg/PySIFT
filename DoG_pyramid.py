import numpy as np

def generate_DoG_octave(gaussian_octave):
    octave = []

    for i in range(1, len(gaussian_octave)):
        if np.amax(gaussian_octave[i]) <= 1:
            octave.append(gaussian_octave[i] - gaussian_octave[i-1])
        else: # integer representation
            octave.append(gaussian_octave[i].astype(np.int16) - gaussian_octave[i-1].astype(np.int16))


    return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)

def generate_DoG_pyramid(gaussian_pyramid):
    pyr = []

    for gaussian_octave in gaussian_pyramid:
        pyr.append(generate_DoG_octave(gaussian_octave))

    return pyr
    