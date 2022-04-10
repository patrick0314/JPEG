import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess(img_path):
    # Convert from BGR to YCbCr
    img = cv2.imread(img_path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    '''
    # Manually Convert from BGR to YCbCr
    B, G, R = cv2.split(img)
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.169*R + -0.331*G + 0.5*B + 127.5
    Cr = 0.5*R + -0.419*G + -0.081*B + 127.5
    img1 = cv2.merge((Y, Cr, Cb)).astype(np.uint8)
    '''
    
    # Down-Sampling to 4:2:0
    Y, Cr, Cb = cv2.split(img1)
    Cr2 = cv2.resize(Cr, (Cr.shape[1]//2, Cr.shape[0]//2))
    Cb2 = cv2.resize(Cb, (Cb.shape[1]//2, Cb.shape[0]//2))
    '''
    # Manually Down-Sampling to 4:2:0
    Y, Cr, Cb = cv2.split(img1)
    Cr1 = np.zeros((Cr.shape[0], Cr.shape[1]//2), dtype=np.uint8)
    Cb1 = np.zeros((Cb.shape[0], Cb.shape[1]//2), dtype=np.uint8)
    for i in range(Cr.shape[0]):
        for j in range(Cr.shape[1]//2):
            Cr1[i, j] = (int(Cr[i, 2*j]) + int(Cr[i, 2*j+1])) / 2
            Cb1[i, j] = (int(Cb[i, 2*j]) + int(Cb[i, 2*j+1])) / 2
    Cr2 = np.zeros((Cr1.shape[0]//2, Cr1.shape[1]), dtype=np.uint8)
    Cb2 = np.zeros((Cb1.shape[0]//2, Cb1.shape[1]), dtype=np.uint8)
    for j in range(Cr1.shape[1]):
        for i in range(Cr1.shape[0]//2):
            Cr2[i, j] = (int(Cr1[2*i, j]) + int(Cr1[2*i+1, j])) / 2
            Cb2[i, j] = (int(Cb1[2*i, j]) + int(Cb1[2*i+1, j])) / 2
    '''

    #print('Original Size of Image :', sys.getsizeof(img))
    #print('After Down-Sampling    :', sys.getsizeof(Y)+sys.getsizeof(Cr1)+sys.getsizeof(Cb1))
    return Y, Cr2, Cb2

def postprocess(Y, Cr1, Cb1):
    # Up-Sampling from 4:2:0
    Cr = cv2.resize(Cr1, (Cr1.shape[1]*2, Cr1.shape[0]*2))
    Cb = cv2.resize(Cb1, (Cb1.shape[1]*2, Cb1.shape[0]*2))
    img1 = cv2.merge((Y, Cr, Cb))
    '''
    # Manually Up-Sampling from 4:2:0
    Cr = np.zeros(Y.shape, dtype=np.uint8)
    Cb = np.zeros(Y.shape, dtype=np.uint8)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Cr[i, j] = Cr1[i//2, j//2]
            Cb[i, j] = Cb1[i//2, j//2]
    '''
    
    img1 = cv2.merge((Y, Cr, Cb)).astype(np.uint8)

    # Convert from YCbCr to BGR
    img = cv2.cvtColor(img1, cv2.COLOR_YCR_CB2BGR)

    # Output Recovered Image
    #print('After Up-Sampling      :', sys.getsizeof(img))
    #print('Successfully Recover')
    return img

if __name__ == '__main__':
    img_path = './pics'
    files = os.listdir(img_path)
    count = 1
    for file in files:
        print('Now Operating img ', count, ' : ', file)
        filename = os.path.join(img_path, file)

        Y, Cr1, Cb1 = preprocess(filename)
        print(Y.shape)
        print(Cr1.shape)
        print(Cr1)
        print(Cb1.shape)
        img = postprocess(Y, Cr1, Cb1)

        cv2.imshow('test', img)
        cv2.waitKey(0)

        count += 1