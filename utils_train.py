# system modules
import os
import re
# extra modules
import numpy as np
import pandas as pd
from tqdm import tqdm
# specific stuff
from keras.utils import plot_model
from keras.callbacks import Callback, \
        EarlyStopping, TerminateOnNaN, \
        ModelCheckpoint, TensorBoard
from keras_tqdm import TQDMNotebookCallback

# bare-bone tqdm callback
class TQDMCustomCallback(Callback):
    def __init__(self, tot_epochs, **kwargs):
        self.tot_epochs = tot_epochs
        super().__init__(**kwargs)
        
    def on_train_begin(self, logs={}):
        self.tqdm = tqdm(total=self.tot_epochs)

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.tqdm.set_description(desc=f'loss: {loss:.4f}, val_loss: {val_loss:.4f}', refresh=False)
        self.tqdm.update(n=1)
    
    def on_train_end(self, logs={}):
        self.tqdm.close()
                

# train model with all the fancy stuff (checkpoint, tensorbord, tqdm...)
def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, 
                epochs_range, validation_freq, 
                tqdm_bar, earlystopping_patience, checkpoint_destination, tensorboard_destination, cuda_device):
    #training callback functions
    callbacks = [
        TerminateOnNaN()
    ]
    # show fancy progressbar
    if tqdm_bar:
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()
        n_epochs = epochs_range[1] - epochs_range[0]
        callbacks.append(TQDMCustomCallback(tot_epochs=n_epochs))
    # conclude training if no improvement after N epochs
    if earlystopping_patience is not None:
        callbacks.append(EarlyStopping(monitor='val_loss' if earlystopping_patience > 0 else 'loss', patience=np.abs(earlystopping_patience)))
    # save model after each epoch if improved
    if checkpoint_destination is not None:
        callbacks.append(ModelCheckpoint(
            filepath=model_destination,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1))
    # show tensorboard
    if tensorboard_destination is not None:
        callbacks.append(TensorBoard(
            log_dir=tensorboard_destination))
    # input data
    data_train = {'vae_input_hrtf': x_train}
    data_valid = {'vae_input_hrtf': x_valid}
    if y_train is not None:
        data_train['vae_input_position'] = y_train
        data_valid['vae_input_position'] = y_valid
    # train the autoencoder
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    trainig_obj = model.fit(data_train,
            epochs=epochs_range[1],
            initial_epoch=epochs_range[0],
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=0 if tqdm_bar else 1,
            validation_data=(data_valid,None))
    # return 
    return pd.DataFrame(trainig_obj.history)


# train model with all the fancy stuff (checkpoint, tensorbord, tqdm...)
def train_model_chen2019_dnn(model, x_train, y_train, x_valid, y_valid, batch_size, 
                epochs_range, validation_freq, 
                tqdm_bar, earlystopping_patience, checkpoint_destination, tensorboard_destination, cuda_device):
    #training callback functions
    callbacks = [
        TerminateOnNaN()
    ]
    # show fancy progressbar
    if tqdm_bar:
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()
        n_epochs = epochs_range[1] - epochs_range[0]
        callbacks.append(TQDMCustomCallback(tot_epochs=n_epochs))
    # conclude training if no improvement after N epochs
    if earlystopping_patience is not None:
        callbacks.append(EarlyStopping(monitor='val_loss' if earlystopping_patience > 0 else 'loss', patience=np.abs(earlystopping_patience)))
    # save model after each epoch if improved
    if checkpoint_destination is not None:
        callbacks.append(ModelCheckpoint(
            filepath=model_destination,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1))
    # show tensorboard
    if tensorboard_destination is not None:
        callbacks.append(TensorBoard(
            log_dir=tensorboard_destination))
    # train the autoencoder
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    trainig_obj = model.fit(x_train, y_train,
            epochs=epochs_range[1],
            initial_epoch=epochs_range[0],
            callbacks=callbacks,
            batch_size=batch_size,
            shuffle=True,
            verbose=0 if tqdm_bar else 1,
            validation_data=(x_valid, y_valid))
    # return 
    return pd.DataFrame(trainig_obj.history)


# train model with all the fancy stuff (checkpoint, tensorbord, tqdm...)
def train_model_dnn_concat(model, x_train, x_train_coords, y_train, x_valid, x_valid_coords, y_valid, batch_size, 
                epochs_range, validation_freq, 
                tqdm_bar, earlystopping_patience, checkpoint_destination, tensorboard_destination, cuda_device):
    #training callback functions
    callbacks = [
        TerminateOnNaN()
    ]
    # show fancy progressbar
    if tqdm_bar:
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()
        n_epochs = epochs_range[1] - epochs_range[0]
        callbacks.append(TQDMCustomCallback(tot_epochs=n_epochs))
    # conclude training if no improvement after N epochs
    if earlystopping_patience is not None:
        callbacks.append(EarlyStopping(monitor='val_loss' if earlystopping_patience > 0 else 'loss', patience=np.abs(earlystopping_patience)))
    # save model after each epoch if improved
    if checkpoint_destination is not None:
        callbacks.append(ModelCheckpoint(
            filepath=model_destination,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1))
    # show tensorboard
    if tensorboard_destination is not None:
        callbacks.append(TensorBoard(
            log_dir=tensorboard_destination))
    # train the autoencoder
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print(x_train.shape, x_train_coords.shape)
    trainig_obj = model.fit({
                'dnn_input_features': x_train,
                'dnn_input_coords': x_train_coords
            }, y_train,
            epochs=epochs_range[1],
            initial_epoch=epochs_range[0],
            callbacks=callbacks,
            batch_size=batch_size,
            shuffle=True,
            verbose=0 if tqdm_bar else 1,
            validation_data=({
                'dnn_input_features': x_valid,
                'dnn_input_coords': x_valid_coords
            }, y_valid))
    # return 
    return pd.DataFrame(trainig_obj.history)


# train model with all the fancy stuff (checkpoint, tensorbord, tqdm...)
def train_model_2d(model, x_train, x_valid, batch_size, 
                epochs_range, validation_freq, 
                tqdm_bar, earlystopping_patience, checkpoint_destination, tensorboard_destination, cuda_device):
    #training callback functions
    callbacks = [
        TerminateOnNaN()
    ]
    # show fancy progressbar
    if tqdm_bar:
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()
        n_epochs = epochs_range[1] - epochs_range[0]
        callbacks.append(TQDMCustomCallback(tot_epochs=n_epochs))
    # conclude training if no improvement after N epochs
    if earlystopping_patience is not None:
        callbacks.append(EarlyStopping(monitor='val_loss' if earlystopping_patience > 0 else 'loss', patience=np.abs(earlystopping_patience)))
    # save model after each epoch if improved
    if checkpoint_destination is not None:
        callbacks.append(ModelCheckpoint(
            filepath=model_destination,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1))
    # show tensorboard
    if tensorboard_destination is not None:
        callbacks.append(TensorBoard(
            log_dir=tensorboard_destination))
    # train the autoencoder
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    trainig_obj = model.fit(x_train,
            epochs=epochs_range[1],
            initial_epoch=epochs_range[0],
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=0 if tqdm_bar else 1,
            validation_data=(x_valid, None))
    # return 
    return pd.DataFrame(trainig_obj.history)