from Preprocess import preprocess
from Preprocess import postprocess
from DCT import dct
from DCT import inv_dct
from Quantize import quantize
from Quantize import inv_quantize
from Zig import zigscan, inv_zigscan
from RunLength import runlength, inv_runlength

import os
import sys
import numpy as np
import scipy.io
import cv2
from matplotlib import pyplot as plt

def huffman(F, channal):
    # Get DC Terms and AC Terms
    table = np.loadtxt('./ref/zig_table.txt').astype(np.uint8)
    for i in range(F.shape[0]//8):
        for j in range(F.shape[1]//8):
            # DC terms
            if i == 0 and j == 0:
                d = F[i*8, j*8]
                DC = d
            elif j == 0:
                last = (F.shape[1]//8 - 1) * 8
                d = F[i*8, j*8] - F[(i-1)*8, last]
                DC = np.append(DC, d)
            else:
                d = F[i*8, j*8] - F[i*8, (j-1)*8]
                DC = np.append(DC, d)
            # AC terms
            if i == 0 and j == 0:
                tmp = F[i*8:i*8+8, j*8:j*8+8]
                ac = zigscan(table, tmp).astype(np.uint8)
                AC = ac.reshape((1, 63))
            else:
                tmp = F[i*8:i*8+8, j*8:j*8+8]
                ac = zigscan(table, tmp).astype(np.uint8)
                ac = ac.reshape((1, 63))
                AC = np.append(AC, ac, axis=0)

    # Huffman coding
    dc_coding = runlength(DC.astype(np.int8), channal, 'DC')
    dc_codes = ''
    for code in dc_coding:
        dc_codes += code
    ac_coding = runlength(AC.astype(np.int8), channal, 'AC')
    ac_codes = ''
    for codes in ac_coding:
        for code in codes:
            ac_codes += code
    return dc_codes + '.' + ac_codes + '.' + str(F.shape[0]) + '.' + str(F.shape[1])

def inv_huffman(coding, channal):
    coding = coding.split('.')
    dc_coding = coding[0]
    ac_coding = coding[1]
    DC = np.array(inv_runlength(dc_coding, channal, 'DC'))
    AC = np.array(inv_runlength(ac_coding, channal, 'AC'))

    F = np.zeros((int(coding[2]), int(coding[3])))
    table = np.loadtxt('./ref/zig_table.txt').astype(np.uint8)
    prev = 0
    for i in range(F.shape[0]//8):
        for j in range(F.shape[1]//8):
            # AC term
            tmp = inv_zigscan(table, AC[i*F.shape[0]//8+j])
            F[i*8:i*8+8, j*8:j*8+8] = tmp
            # DC term
            F[i*8, j*8] = prev + DC[i*F.shape[0]//8+j]
            prev = F[i*8, j*8]
    return F

if __name__ == '__main__':
    img_path = './pics'
    files = os.listdir(img_path)
    count = 1
    for file in files:
        print('Now Operating img ', count, ' : ', file)
        filename = os.path.join(img_path, file)
        Y, Cr1, Cb1 = preprocess(filename)
        Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
        Y_Q, Cr1_Q, Cb1_Q = quantize(Y_DCT, 'Y'), quantize(Cr1_DCT, 'C'), quantize(Cb1_DCT, 'C')
        Y_H, Cr1_H, Cb1_H = huffman(Y_Q, 'Y'), huffman(Cr1_Q, 'C'), huffman(Cb1_Q, 'C')
        Y_Q, Cr1_Q, Cb1_Q = inv_huffman(Y_H, 'Y'), inv_huffman(Cr1_H, 'C'), inv_huffman(Cb1_H, 'C')
        Y_DCT, Cr1_DCT, Cb1_DCT = inv_quantize(Y_Q, 'Y'), inv_quantize(Cr1_Q, 'C'), inv_quantize(Cb1_Q, 'C')
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))

        count += 1