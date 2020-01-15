import numpy as np
# specific stuff
from keras.layers import Input, Dense, Activation, \
        Conv1D, ZeroPadding1D, Cropping1D, MaxPooling1D, UpSampling1D, \
        Conv2D, Conv2DTranspose, \
        Conv3D, Conv3DTranspose, ZeroPadding3D, Cropping3D, SpatialDropout3D, \
        MaxPool2D, UpSampling2D, MaxPooling3D, UpSampling3D, \
        Flatten, Reshape, Permute, Lambda, Layer, Add, Concatenate, \
        Dropout, BatchNormalization
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.utils import plot_model


# sampling from normal dist. + reparametrization trick
class VaeSamplingLayer(Layer):
    __name__ = 'vae_sampling_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VaeSamplingLayer, self).__init__(**kwargs)

    def _sample_normal(self, z_mean, z_log_var):
        # batch_size = K.shape(z_mean)[0]
        # z_dims = K.shape(z_mean)[1]
        eps = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_mean, z_log_var = inputs
        return self._sample_normal(z_mean, z_log_var)
    

# loss function dummy layer
class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_mean, z_log_var):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        loss = self.lossfun(x_true, x_pred, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x_pred
    
    
def generate_vae(model_enc, model_dec, input_shape):
    # autoencoder connection
    x_true = Input(shape=input_shape, name='vae_input')
    z_mean, z_log_var = model_enc(x_true)
    z = VaeSamplingLayer(name='vae_sampling')([z_mean, z_log_var])
    x_pred = model_dec(z)
    # loss function
    loss_tensors = [x_true, x_pred, z_mean, z_log_var]
    vae_loss = VAELossLayer(name='vae_loss')(loss_tensors)
    # complete model
    model_vae = Model(inputs=[x_true], outputs=[vae_loss], name='vae')
    model_vae.summary()
    model_vae.compile(optimizer='adam')
    # add metrics
    model_vae.metrics_tensors.append(K.mean(K.square(x_true - x_pred)))
    model_vae.metrics_names.append("mse_loss")
    model_vae.metrics_tensors.append(K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)))
    model_vae.metrics_names.append("kl_loss")
    return model_vae

        
    
# construct a VAE model
def create_model(input_shape, n_layers, n_filters, kernel_size, n_inter, n_latent, 
                 use_batchnorm, use_maxpool, gen_plots):
    filters = n_filters
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(n_layers):
        #print(f'adding encoder stack {i}..')
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1 if use_maxpool else 2,
                   padding='same', 
                   name=f'encoder_{i}_conv')(x)
        if i < n_layers-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'encoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
        if use_maxpool:
            x = MaxPool2D(name=f'encoder_{i}_mpool')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(n_inter, activation='relu')(x)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
    
    ## DECODER
    # build decoder model
    decoder_input_shape = n_latent
    latent_inputs = Input(shape=(decoder_input_shape,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for i in range(n_layers):
        #print(f'adding decoder stack {i}..')
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=1 if use_maxpool else 2,
                            padding='same',
                            name=f'decoder_{i}_dconv')(x)
        if i < n_layers-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'decoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
        if use_maxpool:
            x = UpSampling2D(name=f'decoder_{i}_upsamp')(x)
        filters //= 2
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='linear',
                              padding='same',
                              name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    if gen_plots:
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
    
    ## VAE
    vae = generate_vae(encoder, decoder, input_shape)
    if gen_plots:
        plot_model(vae, to_file='plots/vae.png', show_shapes=True)
    return encoder, decoder, vae


# construct a fully-convolutional VAE model
def create_model_fconv(input_shape, filters_layers, kernel_size, n_latent, 
                 use_batchnorm, use_maxpool, gen_plots):
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i, filters in enumerate(filters_layers):
        #print(f'adding encoder stack {i}..')
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1 if use_maxpool else 2,
                   padding='same', 
                   name=f'encoder_{i}_conv')(x)
        if i < len(filters_layers)-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'encoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
        if use_maxpool:
            x = MaxPool2D(name=f'encoder_{i}_mpool')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
    
    ## DECODER
    # build decoder model
    decoder_input_shape = n_latent
    latent_inputs = Input(shape=(decoder_input_shape,), name='decoder_input')
    #print(latent_inputs)
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        #print(f'adding decoder stack {i}..')
        x = UpSampling2D(name=f'decoder_{i}_upsamp')(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   name=f'decoder_{i}_conv')(x)
        if i < len(filters_layers)-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'decoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
            
        filters //= 2
    outputs = Conv2D(filters=1,
                     kernel_size=3,
                     activation='linear',
                     padding='same',
                     name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    if gen_plots:
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
    
    ## VAE
    vae = generate_vae(encoder, decoder, input_shape)
    if gen_plots:
        plot_model(vae, to_file='plots/vae.png', show_shapes=True)
    return encoder, decoder, vae


# construct a fully-convolutional VAE model
def create_model_fconv_inception(input_shape, filters_layers, kernel_size, n_latent, 
                 use_batchnorm, use_maxpool, gen_plots):
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i, filters in enumerate(filters_layers):
        #print(f'adding encoder stack {i}..')
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1 if use_maxpool else 2,
                   padding='same', 
                   name=f'encoder_{i}_conv')(x)
        x = Conv2D(filters=filters,
                   kernel_size=1,
                   strides=1,
                   padding='same', 
                   name=f'encoder_{i}_1x1')(x)
        if i < len(filters_layers)-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'encoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
        if use_maxpool:
            x = MaxPool2D(name=f'encoder_{i}_mpool')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    if gen_plots:
        plot_model(encoder, to_file='plots/vae_encoder.png', show_shapes=True)
    
    ## DECODER
    # build decoder model
    decoder_input_shape = n_latent
    latent_inputs = Input(shape=(decoder_input_shape,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        #print(f'adding decoder stack {i}..')
        x = UpSampling2D(name=f'decoder_{i}_upsamp')(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   name=f'decoder_{i}_conv')(x)
        x = Conv2D(filters=filters,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   name=f'decoder_{i}_1x1')(x)
        if i < len(filters_layers)-1:
            if use_batchnorm:
                x = BatchNormalization(name=f'decoder_{i}_bnorm')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
            
        filters //= 2
    outputs = Conv2D(filters=1,
                     kernel_size=3,
                     activation='linear',
                     padding='same',
                     name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    if gen_plots:
        plot_model(decoder, to_file='plots/vae_decoder.png', show_shapes=True)
    
    ## VAE
    vae = generate_vae(encoder, decoder, input_shape)
    if gen_plots:
        plot_model(vae, to_file='plots/vae.png', show_shapes=True)
    return encoder, decoder, vae


# loss function dummy layer
class VAELossLayer3D(Layer):
    __name__ = 'vae_loss_layer_3d'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer3D, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_mean, z_log_var):
        #rec_loss = K.mean(K.square(x_true[:,3,3,:] - x_pred[:,3,3,:]))
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        return rec_loss + 0.01*kl_loss

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        loss = self.lossfun(x_true, x_pred, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x_pred
    


# concatenate label to latent dimension
class VAEOneHotLayer(Layer):
    __name__ = 'vae_onehot_layer'
    
    def __init__(self, vmin, vmax, **kwargs):
        self.vmin = vmin
        self.vmax = vmax        
        self.is_placeholder = True
        super(VAEOneHotLayer, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs)
        # get middle hrtf
        x = inputs[:,3,3,:]
        # normalize
        xn = (x - self.vmin) / (self.vmax - self.vmin)
        # quantize
        xq = K.cast(xn*255, 'int32')
        # one-hot
        y = K.one_hot(xq, 256)
        print(y)
        return y


# construct a 3d conv model
def create_model_3d(input_shape, filters_layers, n_latent):
    f_kernel = 7
    ## ENCODER
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Reshape((*input_shape,1))(inputs)
    for i, filters in enumerate(filters_layers):
        x = ZeroPadding3D((0, 0, 1), name=f'encoder_{i}_zpad')(x)
        x = Conv3D(filters=filters,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding='valid',
                   use_bias=False,
                   name=f'encoder_{i}_3dconv')(x)
        x = Activation('relu', name=f'encoder_{i}_act')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)[1:]
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    
#     ## DECODER
#     # build decoder model
#     latent_inputs = Input(shape=(n_latent,), name='decoder_input')
#     x = Dense(np.prod(shape), activation='relu')(latent_inputs)
#     x = Reshape(shape)(x)
#     for i, filters in enumerate(filters_layers[::-1]):
#         x = Conv3DTranspose(filters=filters,
#                    kernel_size=(3, 3, 3),
#                    strides=1,
#                    padding='valid',
#                    use_bias=False,
#                    name=f'decoder_{i}_3dconv')(x)
#         x = Cropping3D((0, 0, 1), name=f'decoder_{i}_crop')(x)
#         x = Activation('relu', name=f'decoder_{i}_act')(x)
#     x = Conv3D(filters=1,
#                 kernel_size=3,
#                 activation='linear',
#                 padding='same',
#                 use_bias=False,
#                 name='decoder_output')(x)
#     outputs = Reshape(input_shape)(x)
#     # instantiate decoder model
#     decoder = Model(latent_inputs, outputs, name='decoder')
#     decoder.summary()
    
    ## DECODER
    # build decoder model
    latent_inputs = Input(shape=(n_latent,), name='decoder_input')
    x = Dense(np.prod(shape), activation='relu')(latent_inputs)
    x = Reshape((*shape[2:], 1))(x)
    for i, filters in enumerate(filters_layers[::-1]):
        x = UpSampling2D(size=(1,4), name=f'decoder_{i}_upsamp')(x)
        x = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   strides=1,
                   padding='same',
                   #use_bias=False,
                   name=f'decoder_{i}_conv')(x)
        x = Activation('relu', name=f'decoder_{i}_act')(x)
    x = Conv2D(filters=1,
                kernel_size=3,
                activation='linear',
                padding='same',
                use_bias=False,
                name='decoder_output')(x)
    outputs = Reshape((128,256))(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    ## VAE
    x_true = Input(shape=input_shape, name='vae_input')
    z_mean, z_log_var = encoder(x_true)
    z = VaeSamplingLayer(name='vae_sampling')([z_mean, z_log_var])
    x_pred = decoder(z)
    
    x_true_oh = VAEOneHotLayer(vmin=-60, vmax=20, name='vae_onehot')(x_true)
    loss_tensors = [x_true_oh, x_pred, z_mean, z_log_var]
    vae_loss = VAELossLayer3D(name='vae_loss')(loss_tensors)
    model_vae = Model(inputs=[x_true], outputs=[vae_loss], name='vae')
    
    model_vae.summary()
    model_vae.compile(optimizer='adam')
    # add metrics
    #model_vae.metrics_tensors.append(K.mean(K.square(x_true[:,3,3,:] - x_pred[:,3,3,:])))
    model_vae.metrics_tensors.append(K.mean(K.square(x_true_oh - x_pred)))
    model_vae.metrics_names.append("mse_loss")
    model_vae.metrics_tensors.append(K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)))
    model_vae.metrics_names.append("kl_loss")

    return encoder, decoder, model_vae 


