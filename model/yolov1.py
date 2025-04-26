import tensorflow as tf

class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, strides=1, padding="same",
                 activation='lrelu', alpha=0.1, batch_norm=False):
        super().__init__()
        self.batch_norm_enabled = batch_norm

        self.conv = tf.keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not batch_norm  # Don't use bias if using batch norm
        )

        if self.batch_norm_enabled:
            self.batch_norm = tf.keras.layers.BatchNormalization()

        if activation == 'lrelu':
            self.Activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        elif activation == 'relu':
            self.Activation = tf.keras.layers.ReLU()
        elif activation is None:
            self.Activation = tf.keras.layers.Activation('linear')  # Identity
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def call(self, x, training=False):
        x = self.conv(x)
        if self.batch_norm_enabled:
            x = self.batch_norm(x, training=training)
        x = self.Activation(x)
        return x



def YoloV1_(input_size = 448):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = inputs
    x = Conv2d(64, 7, strides=2,batch_norm=True)(x)
    x = Conv2d(64,7,batch_norm=True)(x)
    x = tf.keras.layers.MaxPool2D(strides=(2,2),padding="same")(x)
    x = Conv2d(192,3,batch_norm=True)(x)
    x = tf.keras.layers.MaxPool2D(strides=(2,2),padding="same")(x)
    x = Conv2d(128,1,batch_norm=True)(x)
    x = Conv2d(256,3,batch_norm=True)(x)
    x = Conv2d(256,1,batch_norm=True)(x)
    x = Conv2d(512,3,batch_norm=True)(x)
    x = tf.keras.layers.MaxPool2D(strides=(2,2),padding="same")(x)

                        # ---*
    x = Conv2d(256,1,batch_norm=True)(x)#    | 1/4
    x = Conv2d(512,3,batch_norm=True)(x)#    | conv sub blocks
                        # ---*

                        # ---*
    x = Conv2d(256,1,batch_norm=True)(x)#    | 2/4
    x = Conv2d(512,3,batch_norm=True)(x)#    | conv sub blocks
                        # ---*

                        # ---*
    x = Conv2d(256,1,batch_norm=True)(x,)#    | 3/4
    x = Conv2d(512,3,batch_norm=True)(x)#    | conv sub blocks
                        # ---*

                        # ---*
    x = Conv2d(256,1,batch_norm=True)(x)#    | 4/4
    x = Conv2d(512,3,batch_norm=True)(x)#    | conv sub blocks
                        # ---*

    x = Conv2d(512,1,batch_norm=True)(x)
    x = Conv2d(1024,3,batch_norm=True)(x)
    x = tf.keras.layers.MaxPool2D(strides=(2,2),padding="same")(x)

    x = Conv2d(512,1,batch_norm=True)(x)
    x = Conv2d(1024,3,batch_norm=True)(x)
    x = Conv2d(512,1,batch_norm=True)(x)
    x = Conv2d(1024,3,batch_norm=True)(x)

    x = Conv2d(1024,3,batch_norm=True)(x)
    x = Conv2d(1024,3,strides=2,batch_norm=True)(x)
    x = Conv2d(1024,3,batch_norm=True)(x)
    x = Conv2d(1024,3,batch_norm=True)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = tf.keras.layers.Dense(7 * 7 * 30, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    output = tf.keras.layers.Reshape((7, 7, 30))(x) # back into the tensor 
    return tf.keras.Model(inputs=inputs, outputs=output)





    





    










# class YoloV1(tf.keras.Model):
#     def __init__(self, input_size = 416): # __init__ is a special python method for class constructor
#         super().__init__() # since using python3 we dont need to specify self or any of that 
#         self.input = tf.keras.layers.InputLayer(input_shape=(input_size,input_size,3)) # square image size of rgb, thus 3 channels
#         self.Conv2d = Conv2d()
#         self.MaxPool2x2 = tf.keras.layers.MaxPool2D(strides=(2,2),padding="same")
#         self.LRelu_Activation = tf.keras.layers.LeakyReLU(alpha=.1)


    