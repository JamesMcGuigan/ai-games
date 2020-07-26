
# Source: https://stackoverflow.com/questions/12681945/reversing-bits-of-python-integer

import numpy as np

# Create masks and reverse bitstring lookup tables for 15 bit segments = 32kb memory
# Configuration options: 42 = 14 x 3 | 60 = 15 x 4 | 63 = 9 x 7 | more loops = slower
from numba import int64
from numba import int8
from numba import njit



blocksize   = 14                            # BUGFIX: OverflowError: Python int too large to convert to C long
max_int     = 42  # blocksize * (63 // blocksize) # sys.maxsize == 2**63-1 | python requires sign bit which is unusable
max_bit     = (1 << blocksize) - 1          # == 0xFFFF
inset_start = max_int - blocksize
masks_bits  = np.array([ max_bit << offset for offset in range(0, max_int, blocksize) ], dtype=np.int64)
reverse_bit = np.array([ max_bit - n       for n      in range(max_bit)               ], dtype=np.int16)

@njit(int64(int64, int8))
def reverse_bits( bitstring: int, bitsize: int ) -> int:
    # BUG: reverse_bit[bits] out of range
    try:
        if bitstring == 0: return (1 << bitsize) - 1               # short-circuit simplest edgecase
        output = 0                                                 # blank slate
        for n, offset in enumerate(range(0, bitsize, blocksize)):  # look over 60 bit in 4 x 15 bit loops
            mask   = masks_bits[n]                                 # create bit mask of bits we wish to extract
            bits   = (bitstring & mask) >> offset                  # extract values as short int | ignoring sign bit
            stib   = reverse_bit[bits]                             # perform reverse via lookup then cast back to int64
            inset  = inset_start - offset                          # calculate start position of replacement
            output = output | (stib << inset)                      # replace lookup on other end of bitstring
        # output = output & (sys.maxsize >> blocksize << blocksize)  # mask out any excess bits | BUG: numba casting error
        return output
    except:
        return bitstring

#
# reverse_64bit_mask_front = np.array([ 1 << n for n in range(0, 63)    ], dtype=np.int64)
# reverse_64bit_mask_back  = np.array([ 1 << n for n in range(62,-1,-1) ], dtype=np.int64)
#
# # @njit
# def reverse_64bit_naive( bitstring: int ) -> int:
#
#     output = 0
#     for n in range(63):
#         bit =
#         output
#
#         mask   = 1 << n
#         output = output | (bitstring & mask)
#     return output
#
#
# # @njit
# def reverse_32bit( bitstring: int ) -> int:
#     bitstring = ((bitstring & 0x55555555) <<  1) | ((bitstring & 0xAAAAAAAA) >>  1)
#     bitstring = ((bitstring & 0x33333333) <<  2) | ((bitstring & 0xCCCCCCCC) >>  2)
#     bitstring = ((bitstring & 0x0F0F0F0F) <<  4) | ((bitstring & 0xF0F0F0F0) >>  4)
#     bitstring = ((bitstring & 0x00FF00FF) <<  8) | ((bitstring & 0xFF00FF00) >>  8)
#     bitstring = ((bitstring & 0x0000FFFF) << 16) | ((bitstring & 0xFFFF0000) >> 16)
#     return bitstring

