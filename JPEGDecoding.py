from Preprocess import preprocess
from Preprocess import postprocess
from DCT import dct
from DCT import inv_dct
from Quantize import quantize
from Quantize import inv_quantize
from Zig import zigscan, inv_zigscan
from RunLength import runlength, inv_runlength
from Huffman import huffman, inv_huffman

import os
import sys
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt

encode_path = './encode'
decode_path = './decode'
files = os.listdir(encode_path)
count = 1
for file in files:
    print('\nNow Operating img ', count, ' : ', file)
    filename = os.path.join(encode_path, file)

    with open((filename), 'rb') as f:
        jpeg = pickle.load(f)
    Y_H, Cr1_H, Cb1_H = jpeg['Y'], jpeg['Cr'], jpeg['Cb']

    Y_Q, Cr1_Q, Cb1_Q = inv_huffman(Y_H, 'Y'), inv_huffman(Cr1_H, 'C'), inv_huffman(Cb1_H, 'C')
    Y_DCT, Cr1_DCT, Cb1_DCT = inv_quantize(Y_Q, 'Y'), inv_quantize(Cr1_Q, 'C'), inv_quantize(Cb1_Q, 'C')
    Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
    img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))

    cv2.imwrite(os.path.join(decode_path, os.path.splitext(file)[0] + '.bmp'), img=img)
    
    count += 1

print('\n===== JPEG decoding complete =====')