import tensorflow as tf
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def res_block(x,units,kernel_initializer,name):
    x_c = x
    x = tf.keras.layers.Conv2D(units,(1,1),(1,1),
                      kernel_initializer=kernel_initializer,
                      padding='same',name=f'{name}_Conv00')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_Batch00')(x)
    x = tf.keras.layers.Activation('relu',name=f'{name}_Act00')(x)
    x = tf.keras.layers.Conv2D(units,(3,3),(1,1),
                      kernel_initializer=kernel_initializer,
                      padding='same',
                      name=f'{name}_Conv01')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_Batch01')(x)
    x_c = tf.keras.layers.Conv2D(units,(1,1),(1,1),
                        kernel_initializer=kernel_initializer,
                        padding='same',
                        name=f'{name}_Conv02')(x_c)
    x_c = tf.keras.layers.BatchNormalization(name=f'{name}_Batch02')(x_c)
    x = tf.keras.layers.Add(name=f'{name}_Add00')([x,x_c])
    x = tf.keras.layers.Activation('relu',name=f'{name}_Act01')(x)
    return x
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def upsample(filters,size,strides=2,padding="same",batchnorm=False,dropout=0):

    layer = tf.keras.Sequential()
    layer.add(
        res_block(u8,filters,kernel_initializer=kernel_initializer(32),name='Res14'))
    layer.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides,padding,use_bias = False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    if dropout != 0:
        layer.add(tf.keras.layers.Dropout(dropout))

    layer.add(tf.keras.layers.ReLU())

    return layer
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_encoder(input_shape=[None,None,3], trainable = True, name="encoder"): 
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.vgg16.VGG16(input_tensor=Input, include_top=False)
    layer_names = [
    'block1_conv2',   # 64x64
    'block2_conv2',   # 32x32
    'block3_conv3',   # 16x16
    'block4_conv3',  # 8x8
    'block5_conv3',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder  = tf.keras.Model(inputs=Input, outputs=layers,name=name)
    encoder.trainable = trainable

    return encoder
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_decoder(skips,dropout=0):
    up_stack = [
        upsample(512, 3,dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3,dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3,dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3,dropout=dropout),   # 32x32 -> 64x64
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
    return x
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def res_unet_vgg16(input_shape=(128,128,3), out_channels=1, out_ActFunction='sigmoid', trainable = False, name="unetVgg16"):
    input = tf.keras.layers.Input(shape=input_shape)

    skips = get_encoder(input_shape=list(x.shape[1:]), trainable = trainable)(input)

    x = get_decoder(skips, dropout=0)

    last = tf.keras.layers.Conv2DTranspose(
        out_channels, kernel_size=(1,1), strides=2,
        padding='same',activation=out_ActFunction)  #64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=input, outputs=x,name=name)
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = res_unet_vgg16()
    model.summary()