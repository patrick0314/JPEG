from turtle import shape
import numpy as np
import scipy.io
import pickle

'''
JPEGtable :
    dict_keys(['__header__', '__version__', '__globals__', 'C', 'C1', 'Co1', 'Co2', 'L1', 'L2', 'Lc',
            'Ly', 'Q', 'Q1', 'TAC', 'TAY', 'TDC', 'TDY', 'b1', 'b2', 'b3', 'c1', 'ma1', 'ma11',
            'ma2', 'ma21', 'mb1', 'mb11', 'mb2', 'mb21', 'mp1', 'pw2', 'v1', 'v2', 'v3', 'v4',
            'zig', 'ztb', 'zv'])
    where :
        'C'       : DCT transform matrix
        'C1'      : transpose matrix of 'C'
        'Co1'     : RGB to YCbCr
        'Q'&'Q1'  : matrix for quantizing relatively correponding to Y and CbCr
        'TAC'     : table for chrominance AC coefficients
        'TAY'     : table for luminance AC coefficients
        'TDC'     : table for chrominance DC coefficients
        'TDY'     : table for luminance DC coefficients
        'pw2'     : binary values ex: 1, 2, 4, ...
        'zig'     : table for lookup, index by vertical
        'ztb'     : 'zig' table added first element
'''

def table_generation():
    dctable = [[0]]
    i = 0
    while i < 11:
        tmp1 = [-pow(2, i+1)+1+j for j in range(pow(2, i))]
        tmp2 = [pow(2, i)+j for j in range(pow(2, i))]
        tmp = tmp1 + tmp2
        dctable.append(tmp)
        i += 1

    with open('./ref/dctable.txt', 'wb') as f:
        pickle.dump(dctable, f)
    with open('./ref/dctable.txt', 'rb') as f:
        dctable = pickle.load(f)
    #print(dctable)

    actable = []
    i = 0
    while i < 10:
        tmp1 = [-pow(2, i+1)+1+j for j in range(pow(2, i))]
        tmp2 = [pow(2, i)+j for j in range(pow(2, i))]
        tmp = tmp1 + tmp2
        actable.append(tmp)
        i += 1

    with open('./ref/actable.txt', 'wb') as f:
        pickle.dump(actable, f)
    with open('./ref/actable.txt', 'rb') as f:
        actable = pickle.load(f)
    #print(actable)

def get_bin(x, n=0):
    return format(x, 'b').zfill(n)

def runlength(terms, channal, mode):
    coding = []
    jpegtable = scipy.io.loadmat('./ref/JPEGtable.mat')
    if channal == 'Y':
        if mode == 'DC':
            with open('./ref/dctable.txt', 'rb') as f:
                dctable = pickle.load(f)
            TDY = jpegtable['TDY'] # table for luminance DC coefficients
            # Get the idx of the class and the order in the class
            for d in terms:
                for i in range(len(dctable)):
                    if abs(d) > dctable[i][-1]:
                        continue
                    else:
                        j = dctable[i].index(d)
                        break
                # encoding
                if i == 0:
                    coding.append(TDY[i, 0][0])
                else:
                    coding.append(TDY[i, 0][0] + get_bin(j, i))
            return coding
        elif mode == 'AC':
            with open('./ref/actable.txt', 'rb') as f:
                actable = pickle.load(f)
            TAY = jpegtable['TAY'] # table for luminance AC coefficients
            # Get the class of F and the L
            for term in terms:
                tmp = []
                L = 0
                for t in term: # 63
                    if t == 0:
                        L += 1
                    else:
                        for F in range(len(actable)):
                            if abs(t) > actable[F][-1]:
                                continue
                            else:
                                j = actable[F].index(t)
                                break
                        # encoding
                        while L > 15:
                            tmp.append(TAY[151, 0][0])
                            L -= 16
                        F += 1
                        if L < 15:
                            tmp.append(TAY[L*10+F, 0][0] + get_bin(j, F))
                        else:
                            tmp.append(TAY[151+F, 0][0] + get_bin(j, F))
                        L = 0
                if L != 0:
                    tmp.append(TAY[0, 0][0])
                coding.append(tmp)
            return coding
    elif channal == 'C':
        if mode == 'DC':
            with open('./ref/dctable.txt', 'rb') as f:
                dctable = pickle.load(f)
            TDC = jpegtable['TDC'] # table for chrominance DC coefficients
            # Get the idx of the class and the order in the class
            for d in terms:
                for i in range(len(dctable)):
                    if abs(d) > dctable[i][-1]:
                        continue
                    else:
                        j = dctable[i].index(d)
                        break
                # encoding
                if i == 0:
                    coding.append(TDC[i, 0][0])
                else:
                    coding.append(TDC[i, 0][0] + get_bin(j, i))
            return coding
        elif mode == 'AC':
            with open('./ref/actable.txt', 'rb') as f:
                actable = pickle.load(f)
            TAC = jpegtable['TAC'] # table for chrominance AC coefficients
            # Get the class of F and the L
            for term in terms:
                tmp = []
                L = 0
                for t in term: # 63
                    if t == 0:
                        L += 1
                    else:
                        for F in range(len(actable)):
                            if abs(t) > actable[F][-1]:
                                continue
                            else:
                                j = actable[F].index(t)
                                break
                        # encoding
                        while L > 15:
                            tmp.append(TAC[151, 0][0])
                            L -= 16
                        F += 1
                        if L < 15:
                            tmp.append(TAC[L*10+F, 0][0] + get_bin(j, F))
                        else:
                            tmp.append(TAC[151+F, 0][0] + get_bin(j, F))
                        L = 0
                if L != 0:
                    tmp.append(TAC[0, 0][0])
                coding.append(tmp)
            return coding

def inv_runlength(coding, channal, mode):
    jpegtable = scipy.io.loadmat('./ref/JPEGtable.mat')
    if channal == 'Y':
        if mode == 'DC':
            with open('./ref/dctable.txt', 'rb') as f:
                dctable = pickle.load(f)
            terms = []
            c = [2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
            for code in coding:
                i = c.index(len(code))
                idx = code[-i:]
                d = dctable[i][int(idx, 2)]
                terms.append(d)
            return terms
        elif mode == 'AC':
            with open('./ref/actable.txt', 'rb') as f:
                actable = pickle.load(f)
            TAY = jpegtable['TAY'] # table for luminance AC coefficients
            terms = []
            for code in coding:
                tmp = []
                for c in code:
                    # Find Code Word and Additional Code
                    i = 0
                    while i < 162:
                        l = len(TAY[i, 0][0])
                        if TAY[i, 0][0] == c[:l]:
                            break
                        i += 1
                    # Find R and S
                    R = i // 10
                    S = i % 10
                    if R == 16:
                        R -= 1
                        S += 9
                    elif R == 15 and S != 0:
                        S -= 1
                    elif R == 0 and S == 0:
                        pass
                    elif S == 0:
                        R -= 1
                        S = 10
                    # Inverse Huffman
                    if R == 15 and S == 0:
                        for _ in range(R+1): tmp.append(0)
                    elif R == 0 and S == 0:
                        while len(tmp) < 63:
                            tmp.append(0)
                    else:
                        for _ in range(R): tmp.append(0)
                        tmp.append(actable[S-1][int(c[l:], 2)])
                terms.append(tmp)
            return terms
    elif channal == 'C':
        if mode == 'DC':
            with open('./ref/dctable.txt', 'rb') as f:
                dctable = pickle.load(f)
            terms = []
            c = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
            for code in coding:
                i = c.index(len(code))
                idx = code[-i:]
                d = dctable[i][int(idx, 2)]
                terms.append(d)
            return terms
        elif mode == 'AC':
            with open('./ref/actable.txt', 'rb') as f:
                actable = pickle.load(f)
            TAC = jpegtable['TAC'] # table for luminance AC coefficients
            terms = []
            for code in coding:
                tmp = []
                for c in code:
                    # Find Code Word and Additional Code
                    i = 0
                    while i < 162:
                        l = len(TAC[i, 0][0])
                        if TAC[i, 0][0] == c[:l]:
                            break
                        i += 1
                    # Find R and S
                    R = i // 10
                    S = i % 10
                    if R == 16:
                        R -= 1
                        S += 9
                    elif R == 15 and S != 0:
                        S -= 1
                    elif R == 0 and S == 0:
                        pass
                    elif S == 0:
                        R -= 1
                        S = 10
                    # Inverse Huffman
                    if R == 15 and S == 0:
                        for _ in range(R+1): tmp.append(0)
                    elif R == 0 and S == 0:
                        while len(tmp) < 63:
                            tmp.append(0)
                    else:
                        for _ in range(R): tmp.append(0)
                        tmp.append(actable[S-1][int(c[l:], 2)])
                terms.append(tmp)
            return terms

if __name__ == '__main__':
    table_generation()
    #runlength()