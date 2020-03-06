import matplotlib.image as mpimg
import numpy as np


def matchAB(fileA, fileB):
    predicted = mpimg.imread(fileA)
    target = mpimg.imread(fileB)
    [m,n]=target.shape
    print("same shape:", target.shape==predicted.shape)
    e_2=0
    el_2=0
    er_2=0
    eu_2=0
    eb_2=0
    elu_2=0
    elb_2=0
    eru_2=0
    erb_2=0
    st_2=0

    for i in range(m):
        for j in range(n):
            if i+1==m:
                i_next=0
            else: i_next=i+1
            if j+1==n:
                j_next=0
            else: j_next=j+1
            e_square=np.square(predicted[i][j]-target[i][j])
            el_2_square=np.square(predicted[i][j-1]-target[i][j])
            er_2_square=np.square(predicted[i][j_next]-target[i][j])
            eu_2_square=np.square(predicted[i-1][j]-target[i][j])
            eb_2_square=np.square(predicted[i_next][j]-target[i][j])
            elu_2_square=np.square(predicted[i-1][j-1]-target[i][j])
            elb_2_square=np.square(predicted[i-1][j_next]-target[i][j])
            eru_2_square=np.square(predicted[i_next][j-1]-target[i][j])
            erb_2_square=np.square(predicted[i_next][j_next]-target[i][j])
            e_2=e_2+e_square
            el_2 =el_2+el_2_square
            er_2 = er_2+er_2_square
            eu_2 = eu_2+eu_2_square
            eb_2 = eb_2+eb_2_square
            elu_2 = elu_2+elu_2_square
            elb_2 = elb_2+elb_2_square
            eru_2 = eru_2+eru_2_square
            erb_2 = erb_2+erb_2_square
            t_sq=np.square(target[i][j])
            st_2=st_2+t_sq
    sdr=[]
    SDR=10*np.log10(st_2/e_2)
    SDRl=10*np.log10(st_2/el_2)
    SDRr=10*np.log10(st_2/er_2)
    SDRu=10*np.log10(st_2/eu_2)
    SDRb=10*np.log10(st_2/eb_2)
    SDRlu=10*np.log10(st_2/elu_2)
    SDRlb=10*np.log10(st_2/elb_2)
    SDRru=10*np.log10(st_2/eru_2)
    SDRrb=10*np.log10(st_2/erb_2)
    sdr.append(SDR)
    sdr.append(SDRl)
    sdr.append(SDRr)
    sdr.append(SDRu)
    sdr.append(SDRb)
    sdr.append(SDRlu)
    sdr.append(SDRlb)
    sdr.append(SDRru)
    sdr.append(SDRrb)
    SDR=np.max(sdr)


    return SDR

