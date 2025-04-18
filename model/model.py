import tensorflow as tf


def random_model(num_classes):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32,32,3),name="Input"))

    model.add(tf.keras.layers.Conv2D(32,(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 32x32 -> 16x16 output size

    model.add(tf.keras.layers.Conv2D(64,(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3)) 

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 16x16 -> 8x8 output size


    model.add(tf.keras.layers.Conv2D(128,(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    # model.add(tf.keras.layers.Conv2D(128,(1,1), padding="same"))
    # model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 8x8 -> 4x4 output size

    model.add(tf.keras.layers.Conv2D(256,(2,2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    
    # model.add(tf.keras.layers.Conv2D(256,(1,1), padding="same"))
    # model.add(tf.keras.layers.BatchNormalization()) # axis = -1 as data is in channels last format 
    # model.add(tf.keras.layers.LeakyReLU(alpha=0.3))                        

    # Classification 
    model.add(tf.keras.layers.Flatten()) # 4096 from 4x4*256
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))  # regularization
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))  # Or 'sigmoid' for binary

    # Compile model using the Adam flavor of gradient decent, since the data is labeled as being 1 of 10 possibles images,
    # then we can justify using sparse_categorical_crossentropy. If the data has more than one output per image, then we would use
    # categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # if using label integers
        metrics=['accuracy'] # good as long as dataset is not imbalanced
    )

    return model






def random_model_skip(num_classes):


    x_input = x = tf.keras.Input(shape=(32,32,3),name="Input")

    x = tf.keras.layers.Conv2D(32,(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = skip1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 32x32 -> 16x16 output size

    # just added, have no clue lol
    x = tf.keras.layers.Conv2D(64,(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x) 

    # just added, have no clue lol
    skip1 = tf.keras.layers.Conv2D(64,(3,3), padding="same")(skip1)
    skip1 = tf.keras.layers.BatchNormalization()(skip1) # axis = -1 as data is in channels last format 
    skip1 = tf.keras.layers.LeakyReLU(alpha=0.3)(skip1)

    x = skip2 = tf.keras.layers.Add()([x,skip1])
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 16x16 -> 8x8 output size


    x = tf.keras.layers.Conv2D(128,(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    skip2 = tf.keras.layers.Conv2D(128,(1,1), padding="same")(skip2)
    skip2 = tf.keras.layers.MaxPooling2D((2,2))(skip2)
    skip2 = tf.keras.layers.LeakyReLU(alpha=0.3)(skip2)

    x = skip3 = tf.keras.layers.Add()([x,skip2])

    x = tf.keras.layers.Conv2D(256,(2,2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 8x8 -> 4x4output size





    # # # just added, have no clue lol
    # # x = tf.keras.layers.Conv2D(64,(4,4), padding="same")(x)
    # # x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    # # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x) 

    # x = tf.keras.layers.Conv2D(64,(3,3), padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    # skip1 = tf.keras.layers.Conv2D(64,(1,1), padding="same")(skip1) #ensures dim match 
    # skip1 = tf.keras.layers.BatchNormalization()(skip1) # axis = -1 as data is in channels last format 
    # skip1 = tf.keras.layers.LeakyReLU(alpha=0.3)(skip1) # 16x16x64

    # x = skip2 =  tf.keras.layers.Add()([x,skip1])
    # x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 16x16 -> 8x8 output size

    # x = tf.keras.layers.Conv2D(128,(3,3), padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # # x = tf.keras.layers.Conv2D(128,(1,1), padding="same"))
    # # x = tf.keras.layers.BatchNormalization() # axis = -1 as data is in channels last format 
    # # x = tf.keras.layers.LeakyReLU(alpha=0.3))
    # x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 8x8 -> 4x4 output size

    # x = tf.keras.layers.Conv2D(256,(2,2), padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    # skip2 = tf.keras.layers.Conv2D(256,(1,1), padding="same")(skip2)
    # skip2 = tf.keras.layers.BatchNormalization()(skip2) # axis = -1 as data is in channels last format 
    # skip2 = tf.keras.layers.LeakyReLU(alpha=0.3)(skip2)
    # skip2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(skip2) # 16x16 -> 8x8 output size
    # skip2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(skip2) # 8x8 -> 4x4 output size
    
    # skip1 =  tf.keras.layers.MaxPool2D(pool_size=(2,2))(skip1)
    # skip1 =  tf.keras.layers.MaxPool2D(pool_size=(2,2))(skip1) # 8x8 -> 4x4 output size
    # skip1 = tf.keras.layers.Conv2D(256,(1,1), padding="same")(skip1)



    # x = skip2 = tf.keras.layers.Add()([x,skip2,skip1])


    
    # x = tf.keras.layers.Conv2D(256,(1,1), padding="same"))
    # x = tf.keras.layers.BatchNormalization() # axis = -1 as data is in channels last format 
    # x = tf.keras.layers.LeakyReLU(alpha=0.3))                        

    # Classification 
    x = tf.keras.layers.Flatten()(x) # 4096 from 4x4*256
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # regularization
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)  # Or 'sigmoid' for binary

    # Compile model using the Adam flavor of gradient decent, since the data is labeled as being 1 of 10 possibles images,
    # then we can justify using sparse_categorical_crossentropy. If the data has more than one output per image, then we would use
    # categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model = tf.keras.Model(inputs=x_input, outputs = x)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # if using label integers
        metrics=['accuracy'] # good as long as dataset is not imbalanced
    )

    return model




def random_model_skip_2(num_classes):


    x_input = x = x_2 = x_3 = tf.keras.Input(shape=(32,32,3),name="Input")

    x = tf.keras.layers.Conv2D(32,(3,3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 32x32 -> 16x16 output size

    x = tf.keras.layers.Conv2D(64,(3,3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x) # axis = -1 as data is in channels last format 
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x) # 16x16 -> 8x8 output size

    x = tf.keras.layers.Dropout(.25)(x)



    x_2 = tf.keras.layers.Conv2D(32,(4,4), padding="same", use_bias=False)(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2) # ax_2is = -1 as data is in channels last format 
    x_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(x_2)
    x_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_2) # 32x32 -> 16x16 output size
    x_2 = tf.keras.layers.Conv2D(64,(3,3), padding="same", use_bias=False)(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2) # ax_2is = -1 as data is in channels last format 
    x_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(x_2)
    x_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_2) # 16x16 -> 8x8 output size
    x_2 = tf.keras.layers.Dropout(.25)(x_2)

    x_x_2 = tf.keras.layers.Add()([x,x_2])

    x_x_2 = tf.keras.layers.Conv2D(128,(3,3), padding="same", use_bias=False)(x_x_2)
    x_x_2 = tf.keras.layers.BatchNormalization()(x_x_2) # ax_x_2is = -1 as data is in channels last format 
    x_x_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(x_x_2)
    x_x_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_x_2) # 32x32 -> 16x16 output size



    x_3 = tf.keras.layers.Conv2D(32,(2,2), padding="same", use_bias=False)(x_3)
    x_3 = tf.keras.layers.BatchNormalization()(x_3) # ax_3is = -1 as data is in channels last format 
    x_3 = tf.keras.layers.LeakyReLU(alpha=0.3)(x_3)
    x_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_3) # 32x32 -> 16x16 output size
    x_3 = tf.keras.layers.Conv2D(64,(3,3), padding="same", use_bias=False)(x_3)
    x_3 = tf.keras.layers.BatchNormalization()(x_3) # ax_3is = -1 as data is in channels last format 
    x_3 = tf.keras.layers.LeakyReLU(alpha=0.3)(x_3)
    x_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_3)
    x_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x_3)
    x_3 = tf.keras.layers.Conv2D(128,(1,1), padding="same", use_bias=False)(x_3)


    x  = tf.keras.layers.Add()([x_3,x_x_2])


             
    # Classification 
    x = tf.keras.layers.Flatten()(x) # 4096 from 4x4*256
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # regularization
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)  # Or 'sigmoid' for binary

    # Compile model using the Adam flavor of gradient decent, since the data is labeled as being 1 of 10 possibles images,
    # then we can justify using sparse_categorical_crossentropy. If the data has more than one output per image, then we would use
    # categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model = tf.keras.Model(inputs=x_input, outputs = x)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # if using label integers
        metrics=['accuracy'] # good as long as dataset is not imbalanced
    )

    return model






def create_best_model(num_classes):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)  # 32x32 -> 16x16
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)  # 16x16 -> 8x8
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


import tensorflow as tf

def create_best_model_skips(num_classes):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x_1 = x  # store before downsampling
    x = tf.keras.layers.MaxPooling2D()(x)  # 32x32 -> 16x16
    x = tf.keras.layers.Dropout(0.25)(x)

    # Project x_1 to match shape of x
    x_1 = tf.keras.layers.Conv2D(32, (1,1), strides=2, padding='same', use_bias=False)(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)

    x = tf.keras.layers.Add()([x, x_1])

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x_2 = x  # store before downsampling
    x = tf.keras.layers.MaxPooling2D()(x)  # 16x16 -> 8x8
    x = tf.keras.layers.Dropout(0.3)(x)

    # Project x_2 to match shape of x
    x_2 = tf.keras.layers.Conv2D(64, (1,1), strides=2, padding='same', use_bias=False)(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)

    x = tf.keras.layers.Add()([x, x_2])

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
