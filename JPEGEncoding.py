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

img_path = './pics'
encode_path = './encode'
files = os.listdir(img_path)
count = 1
for file in files:
    print('\nNow Operating img ', count, ' : ', file)
    filename = os.path.join(img_path, file)
    img = cv2.imread(filename)
    Y, Cr1, Cb1 = preprocess(filename)
    Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
    Y_Q, Cr1_Q, Cb1_Q = quantize(Y_DCT, 'Y'), quantize(Cr1_DCT, 'C'), quantize(Cb1_DCT, 'C')
    Y_H, Cr1_H, Cb1_H = huffman(Y_Q, 'Y'), huffman(Cr1_Q, 'C'), huffman(Cb1_Q, 'C')
    jpeg = {'Y':Y_H, 'Cr':Cr1_H, 'Cb':Cb1_H}

    with open(os.path.join(encode_path, os.path.splitext(file)[0] + '.txt'), 'wb') as f:
        pickle.dump(jpeg, f)

    count += 1

print('\n===== JPEG encoding complete =====')