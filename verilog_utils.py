import numpy as np
from _SIFT_utils import get_sigma, get_ksize_from_sigma, get_kernel

def write_verilog_output(kernel_size, sigma):
    bit_number = 24
    print(f"Kernel size: {kernel_size}, sigma: {sigma}")
    kernel = get_kernel(kernel_size, sigma)
    assert len(kernel) == kernel_size
    scaling_factor = 1 / np.amin(kernel)
    kernel = (kernel * scaling_factor).astype(np.uint32) 
    file = f"verilog_output_ksize_{kernel_size}_sigma_{round(sigma, 4)}"
    print(f"Writing to file {file}")
    text = ""
    for i in range(kernel_size**2):
        kernel_val = kernel[i//kernel_size, i % kernel_size]
        text += f"\tassign kernel[{i}] = {bit_number}'d{kernel_val};\n"
    with open(file, 'w') as f:
        f.write(text)
    print(f"Wrote text to file {file}")

if __name__ == "__main__":
    num_blurs = 5
    sigmas = [get_sigma(i, root=2.5) for i in range(num_blurs)]
    kernel_sizes = [get_ksize_from_sigma(sig) for sig in sigmas] 
    write_verilog_output(kernel_size=kernel_sizes[0], sigma=sigmas[0])
