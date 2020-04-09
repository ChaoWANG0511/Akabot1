import numpy as np
import wave
import struct
import librosa
import matplotlib.pyplot as plt
import scipy
import soundfile as sf

filePath='C:/Users/jy\Desktop/test/source/vocals.wav'
#filePath="C:/Users\jy\Desktop/to train_new\Sources\Dev/051 - AM Contra - Heart Peripheral/vocals.wav"

'''
def loadAudioFile(filePath,):
    wf = wave.open(filePath, 'rb')

    nframes = wf.getnframes()
    str_data = wf.readframes(nframes)
    sampleRate=wf.getframerate()
    samplewidth=wf.getsampwidth()
    nc=wf.getnchannels()
    wf.close()

    print(nframes)


    # 将波形数据转换成数组
    #wave_data = np.frombuffer(str_data, dtype='<i2').reshape(-1, nc)
    wave_data = np.fromstring(str_data)
    #print(wave_data)

    #print(np.min(wave_data),np.max(wave_data))

    wave_data=np.nan_to_num(wave_data)

    #print(np.min(wave_data),np.max(wave_data))
    #print(wave_data)


    return wave_data, sampleRate,samplewidth,nc

'''


#dda,sssr=scipy.io.wavfile.read(filePath)

#da,sr,sw,nc=loadAudioFile(filePath)

data,ssr=librosa.load(filePath,sr=None)

sig,srr=sf.read(filePath)


print("soundfile",sig,sig.shape,srr)
print("librosa:",data,data.shape,ssr)
#print ("scipy:", sssr,sssr.shape)
#print("wave:",da,da.shape,sr)

def saveAudioFile(audioFile, filePath, sampleRate,samplewidth,nc):

    wf_mono = wave.open(filePath, 'wb')
    wf_mono.setnchannels(nc)
    wf_mono.setframerate(sampleRate)
    wf_mono.setsampwidth(samplewidth)
    wf_mono.writeframes(audioFile)
    wf_mono.close()

filePath='C:/Users/jy\Desktop/test/source/vocals_test2.wav'
sf.write(filePath,sig,srr)

wf = wave.open(filePath, 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()


filePath='C:/Users/jy\Desktop/test/source/vocals_test3.wav'
sf.write(filePath,data,srr)
wf = wave.open(filePath, 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()

filePath='C:/Users/jy\Desktop/test/source/vocals_test4.wav'
data=np.vstack((data,data)).T
#print("librosa:",data,data.shape,ssr)

sf.write(filePath,data,srr)
wf = wave.open(filePath, 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()