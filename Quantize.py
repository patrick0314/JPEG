from Preprocess import preprocess
from Preprocess import postprocess
from DCT import dct
from DCT import inv_dct

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def quantize(F, channal, tau):
    if channal == 'Y':
        Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], \
                    [12, 12, 14, 19, 26, 58, 60, 55], \
                    [14, 13, 16, 24, 40, 57, 69, 56], \
                    [14, 17, 22, 29, 51, 87, 80, 62], \
                    [18, 22, 37, 56, 68, 109, 103, 77], \
                    [24, 35, 55, 64, 81, 104, 113, 92], \
                    [49, 64, 78, 87, 103, 121, 120, 101], \
                    [72, 92, 95, 98, 112, 100, 103, 99]])
    elif channal == 'C':
        Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99], \
                    [18, 21, 26, 66, 99, 99, 99, 99], \
                    [24, 26, 56, 99, 99, 99, 99, 99], \
                    [47, 66, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99]])
    Q *= tau
    for i in range(F.shape[0]//8):
        for j in range(F.shape[1]//8):
            F[i*8:i*8+8, j*8:j*8+8] = F[i*8:i*8+8, j*8:j*8+8] / Q

    F = np.round(F)
    return F

def inv_quantize(F, channal, tau):
    if channal == 'Y':
        Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], \
                    [12, 12, 14, 19, 26, 58, 60, 55], \
                    [14, 13, 16, 24, 40, 57, 69, 56], \
                    [14, 17, 22, 29, 51, 87, 80, 62], \
                    [18, 22, 37, 56, 68, 109, 103, 77], \
                    [24, 35, 55, 64, 81, 104, 113, 92], \
                    [49, 64, 78, 87, 103, 121, 120, 101], \
                    [72, 92, 95, 98, 112, 100, 103, 99]])
    elif channal == 'C':
        Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99], \
                    [18, 21, 26, 66, 99, 99, 99, 99], \
                    [24, 26, 56, 99, 99, 99, 99, 99], \
                    [47, 66, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99], \
                    [99, 99, 99, 99, 99, 99, 99, 99]])
    
    Q *= tau
    F = F.astype(np.float16)
    for i in range(F.shape[0]//8):
        for j in range(F.shape[1]//8):
            F[i*8:i*8+8, j*8:j*8+8] = F[i*8:i*8+8, j*8:j*8+8] * Q

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
        Y_Q, Cr1_Q, Cb1_Q = quantize(Y_DCT, 'Y', 1), quantize(Cr1_DCT, 'C', 1), quantize(Cb1_DCT, 'C', 1)
        Y_DCT, Cr1_DCT, Cb1_DCT = inv_quantize(Y_Q, 'Y', 1), inv_quantize(Cr1_Q, 'C', 1), inv_quantize(Cb1_Q, 'C', 1)
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))

        count += 1

        '''
        figure = plt.figure(figsize=(15, 25))
        Y, Cr1, Cb1 = preprocess(filename)
        figure.add_subplot(3, 5, 1)
        plt.title('Y')
        plt.imshow(Y.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 6)
        plt.title('Cr')
        plt.imshow(Cr1.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 11)
        plt.title('Cb')
        plt.imshow(Cb1.astype(np.uint8), interpolation='nearest')
        Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
        figure.add_subplot(3, 5, 2)
        plt.title('Y-DCT')
        plt.imshow(Y_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 7)
        plt.title('Cr-DCT')
        plt.imshow(Cr1_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 12)
        plt.title('Cb-DCT')
        plt.imshow(Cb1_DCT.astype(np.uint8), interpolation='nearest')
        Y_Q, Cr1_Q, Cb1_Q = quantize(Y_DCT, 'Y'), quantize(Cr1_DCT, 'C'), quantize(Cb1_DCT, 'C')
        figure.add_subplot(3, 5, 3)
        plt.title('Y-Q')
        plt.imshow(Y_Q.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 8)
        plt.title('Cr-Q')
        plt.imshow(Cr1_Q.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 13)
        plt.title('Cb-Q')
        plt.imshow(Cb1_Q.astype(np.uint8), interpolation='nearest')
        Y_DCT, Cr1_DCT, Cb1_DCT = inv_quantize(Y_Q, 'Y'), inv_quantize(Cr1_Q, 'C'), inv_quantize(Cb1_Q, 'C')
        figure.add_subplot(3, 5, 4)
        plt.title('Y-Q-Recovered')
        plt.imshow(Y_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 9)
        plt.title('Cr-Q-Recovered')
        plt.imshow(Cr1_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 14)
        plt.title('Cb-Q-Recovered')
        plt.imshow(Cb1_DCT.astype(np.uint8), interpolation='nearest')
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        figure.add_subplot(3, 5, 5)
        plt.title('Y-DCT-Recovered')
        plt.imshow(Y.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 10)
        plt.title('Cr-DCT-Recovered')
        plt.imshow(Cr1.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 5, 15)
        plt.title('Cb-Recovered')
        plt.imshow(Cb1.astype(np.uint8), interpolation='nearest')
        plt.savefig('Results/Quantize.jpg')
        plt.close()
        img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))
        '''