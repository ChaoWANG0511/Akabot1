import museval
import librosa
import soundfile as sf
import wave


path_ref='C:/Users/jy/Desktop/test/source'
path_est='C:/Users/jy/Desktop/test/estimate'

wf = wave.open(path_ref+'/vocals.wav', 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()

w = wave.open(path_est+'/vocals.wav', 'rb')
nframes = w.getparams()
print(nframes)
w.close()

da,sr=sf.read(path_ref+'/vocals.wav')
sf.write(path_ref+"/vocals.wav",da,sr)

a=museval.eval_dir(path_ref,path_est)
print(a)