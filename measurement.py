import matplotlib.image as mpimg
import numpy as np


def matchAB(fileA, fileB):
    predicted = mpimg.imread(fileA)
    target = mpimg.imread(fileB)
    [m,n]=target.shape
    e_2=0
    st_2=0
    for i in range(m):
        for j in range(n):
            e_square=np.square(predicted[i][j]-target[i][j])
            e_2=e_2+e_square
            t_sq=np.square(target[i][j])
            st_2=st_2+t_sq
    SDR=10*np.log10(st_2/e_2)

    return SDR

