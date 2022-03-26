from Preprocess import preprocess
from Preprocess import postprocess

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def dct(f):
    # Extend Original Matrix
    V, U = f.shape[0], f.shape[1]
    while V % 8 != 0 or U % 8 != 0:
        if V % 8 != 0:
            f = np.append(f, f[-1, :].reshape(1, f.shape[1]), axis=0)
            V += 1
        if U % 8 != 0:
            f = np.append(f, f[:, -1].reshape(f.shape[0], 1), axis=1)
            U += 1

    # 8x8 DCT-CMatrix
    c = [1/np.sqrt(2), 1, 1, 1, 1, 1, 1, 1]
    c1, c2 = np.zeros((8, 8), dtype=np.float16), np.zeros((8, 8), dtype=np.float16)
    for u in range(8):
        for i in range(8):
            c1[u, i] = np.sqrt(2/8) * np.cos((2*i+1)*u*np.pi / (2*8)) * c[u]
    for j in range(8):
        for v in range(8):
            c2[j, v] = np.sqrt(2/8) * np.cos((2*j+1)*v*np.pi / (2*8)) * c[v]

    # 8x8 DCT-transform
    f = f.astype(np.float16)
    F = np.zeros(f.shape, dtype=np.float16)
    for i in range(f.shape[0]//8):
        for j in range(f.shape[1]//8):
            tmp = f[i*8:i*8+8, j*8:j*8+8]
            F[i*8:i*8+8, j*8:j*8+8] = np.dot(np.dot(c1, tmp), c2)

    return F

def inv_dct(F):
    # 8x8 DCT-CMatrix
    c = [1/np.sqrt(2), 1, 1, 1, 1, 1, 1, 1]
    c1, c2 = np.zeros((8, 8), dtype=np.float16), np.zeros((8, 8), dtype=np.float16)
    for u in range(8):
        for i in range(8):
            c1[u, i] = np.sqrt(2/8) * np.cos((2*i+1)*u*np.pi / (2*8)) * c[u]
    for j in range(8):
        for v in range(8):
            c2[j, v] = np.sqrt(2/8) * np.cos((2*j+1)*v*np.pi / (2*8)) * c[v]

    # 8x8 DCT-transform
    F = F.astype(np.float16)
    f = np.zeros(F.shape, dtype=np.float16)
    for i in range(F.shape[0]//8):
        for j in range(F.shape[1]//8):
            tmp = F[i*8:i*8+8, j*8:j*8+8]
            f[i*8:i*8+8, j*8:j*8+8] = np.dot(np.dot(c2, tmp), c1)
    
    return f

if __name__ == '__main__':
    img_path = './pics'
    files = os.listdir(img_path)
    count = 1
    for file in files:
        print('Now Operating img ', count, ' : ', file)
        filename = os.path.join(img_path, file)

        Y, Cr1, Cb1 = preprocess(filename)
        Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))

        count += 1
        
        '''
        figure = plt.figure(figsize=(15, 15))
        Y, Cr1, Cb1 = preprocess(filename)
        figure.add_subplot(3, 3, 1)
        plt.title('Y')
        plt.imshow(Y.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 4)
        plt.title('Cr')
        plt.imshow(Cr1.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 7)
        plt.title('Cb')
        plt.imshow(Cb1.astype(np.uint8), interpolation='nearest')
        Y_DCT, Cr1_DCT, Cb1_DCT = dct(Y), dct(Cr1), dct(Cb1)
        figure.add_subplot(3, 3, 2)
        plt.title('Y-DCT')
        plt.imshow(Y_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 5)
        plt.title('Cr-DCT')
        plt.imshow(Cr1_DCT.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 8)
        plt.title('Cb-DCT')
        plt.imshow(Cb1_DCT.astype(np.uint8), interpolation='nearest')
        Y, Cr1, Cb1 = inv_dct(Y_DCT), inv_dct(Cr1_DCT), inv_dct(Cb1_DCT)
        figure.add_subplot(3, 3, 3)
        plt.title('Y-DCT-Recovered')
        plt.imshow(Y.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 6)
        plt.title('Cr-DCT-Recovered')
        plt.imshow(Cr1.astype(np.uint8), interpolation='nearest')
        figure.add_subplot(3, 3, 9)
        plt.title('Cb-Recovered')
        plt.imshow(Cb1.astype(np.uint8), interpolation='nearest')
        plt.savefig('Results/DCT_image.jpg')
        plt.close()
        img = postprocess(Y.astype(np.uint8), Cr1.astype(np.uint8), Cb1.astype(np.uint8))
        '''