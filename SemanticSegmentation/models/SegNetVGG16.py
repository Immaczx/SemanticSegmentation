import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Input, Activation, BatchNormalization, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, Reshape


def get_model(output_channels=3, size=224, name="SegNet"):
    input_shape = (size,size,3)
    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Encoding layer

    inp = Input(input_shape)
    x = encoder.get_layer(name='block1_conv1')(inp)
    x = BatchNormalization(name='bn1')(x)
    x = encoder.get_layer(name='block1_conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D()(x)
    
    x = encoder.get_layer(name='block2_conv1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = encoder.get_layer(name='block2_conv2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPooling2D()(x)

    x = encoder.get_layer(name='block3_conv1')(x)
    x = BatchNormalization(name='bn5')(x)
    x = encoder.get_layer(name='block3_conv2')(x)
    x = BatchNormalization(name='bn6')(x)
    x = encoder.get_layer(name='block3_conv3')(x)
    x = BatchNormalization(name='bn7')(x)
    x = MaxPooling2D()(x)

    x = encoder.get_layer(name='block4_conv1')(x)
    x = BatchNormalization(name='bn8')(x)
    x = encoder.get_layer(name='block4_conv2')(x)
    x = BatchNormalization(name='bn9')(x)
    x = encoder.get_layer(name='block4_conv3')(x)
    x = BatchNormalization(name='bn10')(x)
    x = MaxPooling2D()(x)
    
    x = encoder.get_layer(name='block5_conv1')(x)
    x = BatchNormalization(name='bn11')(x)
    x = encoder.get_layer(name='block5_conv2')(x)
    x = BatchNormalization(name='bn12')(x)
    x = encoder.get_layer(name='block5_conv3')(x)
    x = BatchNormalization(name='bn13')(x)
    x = MaxPooling2D()(x)
    
    # Decoding Layer 
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
    x = BatchNormalization(name='bn17')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
    x = BatchNormalization(name='bn18')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
    x = BatchNormalization(name='bn19')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
    x = BatchNormalization(name='bn20')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
    x = BatchNormalization(name='bn21')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
    x = BatchNormalization(name='bn22')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
    x = BatchNormalization(name='bn23')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
    x = BatchNormalization(name='bn24')(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
    x = BatchNormalization(name='bn25')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
    x = BatchNormalization(name='bn26')(x)
    x = Conv2D(output_channels, 1, activation = tf.keras.activations.softmax, name='OutputLayer')(x)
    pred = Reshape((input_shape[0],input_shape[1],output_channels))(x)
    
    return Model(inputs=inp, outputs=pred, name=name)

if __name__ == '__main__':
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    image_size=224
    classes = 3
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = get_model(output_channels=classes, size=image_size)
    model.summary()
    tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=False,show_layer_names=False)