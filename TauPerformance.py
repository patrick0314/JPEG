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
import scipy.io
import cv2
from matplotlib import pyplot as plt

img_path = './pics'
files = os.listdir(img_path)
count = 1
for file in files:
    print('\nNow Operating img ', count, ' : ', file)
    filename = os.path.join(img_path, file)
    img = cv2.imread(filename)

    taus = list(np.arange(0.5, 2.01, 0.05))
    bpps = []
    psnrs = []
    for tau in taus:
        Y, Cr1, Cb1 = preprocess(filename)
        Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
        Y_Q, Cr1_Q, Cb1_Q = quantize(Y_DCT, 'Y', tau), quantize(Cr1_DCT, 'C', tau), quantize(Cb1_DCT, 'C', tau)
        Y_H, Cr1_H, Cb1_H = huffman(Y_Q, 'Y'), huffman(Cr1_Q, 'C'), huffman(Cb1_Q, 'C')
        Y_Q, Cr1_Q, Cb1_Q = inv_huffman(Y_H, 'Y'), inv_huffman(Cr1_H, 'C'), inv_huffman(Cb1_H, 'C')
        Y_DCT, Cr1_DCT, Cb1_DCT = inv_quantize(Y_Q, 'Y', tau), inv_quantize(Cr1_Q, 'C', tau), inv_quantize(Cb1_Q, 'C', tau)
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        img1 = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))

        total_number_of_bits = sys.getsizeof(Y_H) + sys.getsizeof(Cr1_H) + sys.getsizeof(Cb1_H)
        #print('bpp = ', total_number_of_bits, '/', img.shape[:2], ' = ', total_number_of_bits/(img.shape[0]*img.shape[1]))
        ss = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    ss += (int(img[i, j, k]) - int(img1[i, j, k]))**2
        tmp = 255**2 / ((1/(3*img.shape[0]*img.shape[1])) * ss)
        PSNR = 10 * np.log10(tmp)
        #print('PSNR = ', PSNR)

        bpps.append(total_number_of_bits/(img.shape[0]*img.shape[1]))
        psnrs.append(PSNR)

    figure = plt.figure(figsize=(15, 5))
    figure.add_subplot(1, 3, 1)
    plt.plot(taus, bpps, color='blue')
    plt.title('tau vs bpp')
    plt.axvline(x=1, color='black')
    figure.add_subplot(1, 3, 2)
    plt.plot(taus, psnrs, color='blue')
    plt.title('tau vs PSNR')
    plt.axvline(x=1, color='black')
    figure.add_subplot(1, 3, 3)
    plt.plot(bpps, psnrs, color='blue')
    plt.title('bpp vs PSNR')
    plt.show()

    count += 1
    break

print('\n===== mission complete =====')