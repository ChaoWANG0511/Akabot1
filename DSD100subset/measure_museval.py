import museval
import librosa
import soundfile as sf
import wave


path_ref="C:/Users/jy/Desktop/test/source"
path_est="C:/Users/jy/Desktop/test/estimate"

wf = wave.open(path_ref, 'rb')
nframes = wf.getparams()
print(nframes)
wf.close()

da,sr=sf.read(path_ref)
sf.write(path_ref,da,sr)

a=museval.eval_dir(path_ref,path_est)
print(a)