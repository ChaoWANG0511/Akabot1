import os
import os.path

def traversalDir_FirstDir(path):
    # path 输入是 Dev/Test
    # 定义一个字典，用来存储结果————歌名：路径
    dict = {}
    # 获取该目录下的所有文件夹目录, 每个歌的文件夹

    files = os.listdir(path)
    for file in files:
        # 得到该文件下所有目录的路径
        m = os.path.join(path, file)
        h = os.path.split(m)
        dict[h[1]] = []
        song_wav = os.listdir(m)
        m = m + '/'
        for track in song_wav:
            value = os.path.join(m, track)
            dict[h[1]].append(value)
    return dict
mix_path=traversalDir_FirstDir('C:/Users/jy/Downloads/DSD100subset/Mixtures/Dev')
print("mix_path:\n",mix_path)
sou_path = traversalDir_FirstDir('C:/Users/jy/Downloads/DSD100subset/Sources/Dev')
print("sou_path\n",sou_path)
all_path = mix_path.copy()
for key in all_path.keys():
    all_path[key].extend(sou_path[key])
#     如果extend的是字符串，则字符串会被拆分成字符数组，如果extend的是字典，则字典的key会被加入到List中

print("all_path\n",all_path)