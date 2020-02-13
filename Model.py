from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model

def creat_model():
    mashup = Input(shape=(None, None, 1), name='input')  # shape不含batch size, None意味着可以随便取
    convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)  # 64个filter, 每个都是3*3, 输入padding加0至输出大小等于输入
    conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(
        convA)  # 默认True, 但这层不用bias vector
    conv = BatchNormalization()(conv)

    convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convB)
    conv = BatchNormalization()(conv)  # 不改变尺寸

    conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(128, 3, activation='relu', padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)  # 行重复2次，列重复2次。默认channels_last (default)：(batch, height, width, channels)   为什么大小？

    conv = Concatenate()([conv, convB])  # 默认沿axis=-1
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, convA])
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)

    conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(1, 3, activation='relu', padding='same')(conv)  # 输出只1channel
    acapella = conv

    m = Model(inputs=mashup, outputs=acapella)
    return m