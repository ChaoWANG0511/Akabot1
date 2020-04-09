import numpy as np
import wave
import librosa
import soundfile as sf
import museval

filePath='C:/Users/jy\Desktop/test/source/vocals.wav'

data,ssr=librosa.load(filePath,sr=None)
sig,srr=sf.read(filePath)


print("soundfile",sig,sig.shape,srr)
print("librosa:",data,data.shape,ssr)

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

sf.write(filePath,data,srr)
wf = wave.open(filePath, 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()

