import numpy as np
# specific stuff
from keras.layers import Input, Dense, Activation, \
        Conv1D, ZeroPadding1D, Cropping1D, MaxPooling1D, UpSampling1D, \
        Flatten, Reshape, Permute, Lambda, Layer, Add, Concatenate, \
        Dropout, BatchNormalization
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.utils import plot_model
from utils_model import VaeSamplingLayer, VAELossLayer


## 1D INCEPTION STACK (https://arxiv.org/pdf/1409.4842.pdf) 
def inception_stack_1d(x, filters, name):
    tower_1 = Conv1D(filters=filters, 
                     kernel_size=1, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t1_conv1')(x)
    tower_1 = Conv1D(filters=filters, 
                     kernel_size=3, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t1_conv3')(tower_1)
    
    tower_2 = Conv1D(filters=filters, 
                     kernel_size=1, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t2_conv1')(x)
    tower_2 = Conv1D(filters=filters, 
                     kernel_size=5, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t2_conv5')(tower_2)
    
    tower_3 = MaxPooling1D(pool_size=3, 
                           strides=1, 
                           padding='same',
                           name=f'{name}_t3_mpool')(x)
    tower_3 = Conv1D(filters=filters, 
                     kernel_size=1, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t3_conv1')(tower_3)
    
    tower_4 = Conv1D(filters=filters, 
                     kernel_size=1, 
                     padding='same', 
                     activation='relu',
                     name=f'{name}_t4_conv1')(x)
    
    x = Add(name=f'{name}_add')([tower_1, tower_2, tower_3, tower_4])
    #return Concatenate(axis=-1, name=f'{name}_concat')([tower_1, tower_2, tower_3, tower_4])
    return x


## 1D RESIDUAL STACK (https://arxiv.org/pdf/1512.03385.pdf)
def resnet_stack_1d(x, filters, kernel_size, name):    
    shortcut = Conv1D(filters=filters*4, 
               kernel_size=1, 
               strides=1,
               use_bias=False,
               padding='same',
               name=f'{name}_skip')(x)
    
    x = Conv1D(filters=filters, 
               kernel_size=1, 
               strides=1,
               use_bias=False,
               padding='same',
               name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_act1')(x)
    x = Conv1D(filters=filters, 
               kernel_size=kernel_size, 
               strides=1,
               use_bias=False,
               padding='same',
               name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Activation('relu', name=f'{name}_act2')(x)
    x = Conv1D(filters=filters*4, 
               kernel_size=1, 
               strides=1,
               use_bias=False,
               padding='same',
               name=f'{name}_conv3')(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu', name=f'{name}_act')(x)
    return x


## VARIATIONAL AUTOENCODER + ELEVATION,AZIMUTH
def vae_stack(input_shape, position_shape, encoder, decoder):
    # inputs
    x_true = Input(shape=input_shape, name='vae_input_hrtf')
    y_true = Input(shape=position_shape, name='vae_input_position')
    # autoencoder layers
    z_mean, z_log_var = encoder(x_true)
    z = VaeSamplingLayer(name='vae_sampling')([z_mean, z_log_var])
    z_position = Concatenate(axis=-1, name='vae_concat_pos')([z, y_true])
    x_pred = decoder(z_position)
    # loss layer
    vae_loss = VAELossLayer(name='vae_loss')([x_true, x_pred, z_mean, z_log_var])
    # model
    model_vae = Model(inputs=[x_true, y_true], outputs=[vae_loss], name='vae')
    model_vae.compile(optimizer='adam')
    # add metrics
    model_vae.metrics_tensors.append(K.mean(K.square(x_true - x_pred)))
    model_vae.metrics_names.append("loss_mse")
    model_vae.metrics_tensors.append(K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)))
    model_vae.metrics_names.append("loss_kl")
    return model_vae


## VARIATIONAL AUTOENCODER WITH DENSE LAYERS
def create_autoencoder_dense(n_input, filters_layers, n_latent, gen_plots=False):
    n_positions = 2
    input_shape = (n_input, )
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i, filters in enumerate(filters_layers):
        x = Dense(units=filters, name=f'encoder_{i}_dense')(x)
        x = BatchNormalization(name=f'encoder_{i}_bn')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = latent_inputs
    for i, filters in enumerate(filters_layers[::-1]):
        x = Dense(units=filters, name=f'decoder_{i}_dense')(x)
        x = BatchNormalization(name=f'decoder_{i}_bn')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
    outputs = Dense(units=input_shape[0], name=f'decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = vae_stack(input_shape, position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 

    
## VARIATIONAL AUTOENCODER WITH RESIDUAL (SKIP) 1D CONVOLUTIONAL LAYERS
def create_autoencoder_resnet(n_input, filters_layers, n_latent, kernel_size=3, gen_plots=False):
    n_positions = 2
    input_shape = (n_input, )
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Reshape((*input_shape, 1), name='encoder_reshape')(inputs)
    for i, filters in enumerate(filters_layers):
        x = resnet_stack_1d(x, filters, kernel_size, f'encoder_{i}')
        x = MaxPooling1D(pool_size=2, name=f'encoder_{i}_mp')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        x = UpSampling1D(size=2, name=f'decoder_{i}_up')(x)
        x = resnet_stack_1d(x, filters, 3, f'decoder_{i}')
    outputs = Conv1D(filters=1, 
                     kernel_size=kernel_size,
                     strides=1,
                     padding='same',
                     name='decoder_output')(x)
    outputs = Flatten()(outputs)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = vae_stack(input_shape, position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 

    
## VARIATIONAL AUTOENCODER WITH INCEPTION 1D CONVOLUTIONAL LAYERS
def create_autoencoder_inception(n_input, filters_layers, n_latent, gen_plots=False):
    n_positions = 2
    input_shape = (n_input, )
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Reshape((*input_shape, 1), name='encoder_reshape')(inputs)
    for i, filters in enumerate(filters_layers):
        x = inception_stack_1d(x, filters, f'encoder_{i}')
        x = MaxPooling1D(pool_size=2, name=f'encoder_{i}_mp')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        x = UpSampling1D(size=2, name=f'encoder_{i}_up')(x)
        x = inception_stack_1d(x, filters, f'decoder_{i}')
    outputs = Conv1D(filters=1, 
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     name='decoder_output')(x)
    outputs = Flatten()(outputs)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = vae_stack(input_shape, position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 


## DEEP NEURAL NETWORK WITH DENSE LAYERS
def create_dnn_dense(n_input, filters_layers, n_outputs, gen_plots=False):
    input_shape = (n_input, )
    # build model
    inputs = Input(shape=input_shape, name='dnn_input')
    x = inputs
    for i, filters in enumerate(filters_layers):
        x = Dense(units=filters, name=f'dnn_{i}_dense')(x)
        x = Dropout(rate=0.3, name=f'dnn_{i}_do')(x)
        #x = BatchNormalization(name=f'dnn_{i}_bn')(x)
        x = Activation('relu', name=f'dnn_{i}_act')(x)
    # generate latent vector Q(z|X)
    outputs = Dense(n_outputs, name='dnn_output')(x)
    # instantiate encoder model
    dnn = Model(inputs=inputs, outputs=outputs, name='dnn')
    #dnn.summary()
    
    def mean_squared_error_weighted(n_outputs):
        def lossfunc(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
        return lossfunc
    
    dnn.compile(optimizer='adam', loss=mean_squared_error_weighted(n_outputs))
    if gen_plots:
        plot_model(dnn, to_file='plots/dnn.png', show_shapes=True)
    return dnn


## DEEP NEURAL NETWORK WITH DENSE LAYERS (CONCATENATE DIRECTIONS TO EACH LAYER)
def create_dnn_concat(n_features, n_coords, filters_layers, n_outputs, gen_plots=False):
    # build model
    input_features = Input(shape=(n_features, ), name='dnn_input_features')
    input_coords = Input(shape=(n_coords, ), name='dnn_input_coords')
    x = input_features
    for i, filters in enumerate(filters_layers):
        x = Concatenate(axis=-1, name=f'dnn_{i}_concat')([x, input_coords])
        x = Dense(units=filters, name=f'dnn_{i}_dense')(x)
        x = Dropout(rate=0.3, name=f'dnn_{i}_do')(x)
        #x = BatchNormalization(name=f'dnn_{i}_bn')(x)
        x = Activation('relu', name=f'dnn_{i}_act')(x)
    # generate latent vector Q(z|X)
    outputs = Dense(n_outputs, name='dnn_output')(x)
    # instantiate encoder model
    dnn = Model(inputs=[input_features, input_coords], outputs=outputs, name='dnn')
    #dnn.summary()
    
    def mean_squared_error_weighted(n_outputs):
        def lossfunc(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
        return lossfunc
    
    dnn.compile(optimizer='adam', loss=mean_squared_error_weighted(n_outputs))
    if gen_plots:
        plot_model(dnn, to_file='plots/dnn.png', show_shapes=True)
    return dnn



## CONDITIONAL VARIATIONAL AUTOENCODER (ELEVATION,AZIMUTH)
def cvae_stack(input_shape, position_shape, encoder, decoder):
    # inputs
    x_true = Input(shape=input_shape, name='vae_input_hrtf')
    y_true = Input(shape=position_shape, name='vae_input_position')
    x_encoder = Concatenate(axis=-1, name='vae_input_encoder')([x_true, y_true])
    # autoencoder layers
    z_mean, z_log_var = encoder(x_encoder)
    z = VaeSamplingLayer(name='vae_sampling')([z_mean, z_log_var])
    z_position = Concatenate(axis=-1, name='vae_concat_pos')([z, y_true])
    x_pred = decoder(z_position)
    # loss layer
    vae_loss = VAELossLayer(name='vae_loss')([x_true, x_pred, z_mean, z_log_var])
    # model
    model_vae = Model(inputs=[x_true, y_true], outputs=[vae_loss], name='vae')
    model_vae.compile(optimizer='adam')
    # add metrics
    model_vae.metrics_tensors.append(K.mean(K.square(x_true - x_pred)))
    model_vae.metrics_names.append("loss_mse")
    model_vae.metrics_tensors.append(K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)))
    model_vae.metrics_names.append("loss_kl")
    return model_vae


## CONDITIONAL VARIATIONAL AUTOENCODER WITH DENSE LAYERS
def create_cvae_dense(n_input, filters_layers, n_latent, gen_plots=False):
    n_positions = 2
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=(n_input+n_positions, ), name='encoder_input')
    x = inputs
    for i, filters in enumerate(filters_layers):
        x = Dense(units=filters, name=f'encoder_{i}_dense')(x)
        x = BatchNormalization(name=f'encoder_{i}_bn')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = latent_inputs
    for i, filters in enumerate(filters_layers[::-1]):
        x = Dense(units=filters, name=f'decoder_{i}_dense')(x)
        x = BatchNormalization(name=f'decoder_{i}_bn')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
    outputs = Dense(units=n_input, name=f'decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = cvae_stack((n_input, ), position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 


## CONDITIONAL VARIATIONAL AUTOENCODER WITH RESIDUAL (SKIP) 1D CONVOLUTIONAL LAYERS
def create_cvae_resnet(n_input, filters_layers, n_latent, kernel_size=3, gen_plots=False):
    n_positions = 2
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=(n_input+n_positions, ), name='encoder_input')
    x = Reshape((n_input+n_positions, 1), name='encoder_reshape')(inputs)
    for i, filters in enumerate(filters_layers):
        x = resnet_stack_1d(x, filters, kernel_size, f'encoder_{i}')
        x = MaxPooling1D(pool_size=2, name=f'encoder_{i}_mp')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        x = UpSampling1D(size=2, name=f'decoder_{i}_up')(x)
        x = resnet_stack_1d(x, filters, 3, f'decoder_{i}')
    outputs = Conv1D(filters=1, 
                     kernel_size=kernel_size,
                     strides=1,
                     padding='same',
                     name='decoder_output')(x)
    outputs = Flatten()(outputs)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = cvae_stack((n_input, ), position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 

    
## CONDITIONAL VARIATIONAL AUTOENCODER WITH INCEPTION 1D CONVOLUTIONAL LAYERS
def create_cvae_inception(n_input, filters_layers, n_latent, gen_plots=False):
    n_positions = 2
    position_shape = (n_positions,)
    latent_shape = (n_latent+n_positions,)
    
    ## ENCODER
    # build encoder model
    inputs = Input(shape=(n_input+n_positions, ), name='encoder_input')
    x = Reshape((*input_shape, 1), name='encoder_reshape')(inputs)
    for i, filters in enumerate(filters_layers):
        x = inception_stack_1d(x, filters, f'encoder_{i}')
        x = MaxPooling1D(pool_size=2, name=f'encoder_{i}_mp')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    # generate latent vector Q(z|X)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=latent_shape, name='decoder_input')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        x = UpSampling1D(size=2, name=f'encoder_{i}_up')(x)
        x = inception_stack_1d(x, filters, f'decoder_{i}')
    outputs = Conv1D(filters=1, 
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     name='decoder_output')(x)
    outputs = Flatten()(outputs)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    model_vae = cvae_stack(input_shape, position_shape, encoder, decoder)
    model_vae.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
        plot_model(model_vae, to_file='plots/vae_model.png', show_shapes=True)
    return encoder, decoder, model_vae 

