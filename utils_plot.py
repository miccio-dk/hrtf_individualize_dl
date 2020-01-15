import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# display a 2D plot of the datapoints in the latent space
def plot_latent_space(encoder, data, ax,
                      batch_size=512, dims=(0,1),
                      colorbar=False, use_pca=False):
    x_test, y_test = data
    z_mean, z_log_var = encoder.predict(x_test, batch_size=batch_size)
    
    if use_pca:
        scaler = StandardScaler()
        pca = PCA(n_components=None, svd_solver='randomized')
        zs = scaler.fit_transform(z_mean)
        z_pca = pca.fit_transform(zs)
        zd_pca = pd.DataFrame(z_pca, columns=[f'PC_{i}' for i in range(z_pca.shape[1])])
        zd_pca['target'] = y_test
        sc = ax.scatter(zd_pca[:, dims[0]], zd_pca[:, dims[1]], c=zd_pca['target'], alpha=0.75)
    else:
        zd = pd.DataFrame(z_mean, columns=[f'Z_{i}' for i in range(z_mean.shape[1])])
        zd['target'] = y_test
        sc = ax.scatter(zd.iloc[:, dims[0]], zd.iloc[:, dims[1]], c=zd['target'], alpha=0.75)
        
    if colorbar:
        plt.colorbar(sc, ax=ax)
        
    xname = 'pc' if use_pca else 'z'
    ax.set_xlabel("{}[{}]".format(xname, dims[0]))
    ax.set_ylabel("{}[{}]".format(xname, dims[1]))
    ax.set_title(dims)


    
# display a grid of plots for all combinations of latent dimensions
def plot_latent_pairs(encoder, data,
                      batch_size=512, n_pca=False, n_dim=False, height=2.5, nlvl=5, plot_kws={}):
    x_test, y_test = data
    z_mean, z_log_var = encoder.predict(x_test, batch_size=batch_size)
    
    if n_pca:
        scaler = StandardScaler()
        pca = PCA(n_components=None, svd_solver='randomized')
        zs = scaler.fit_transform(z_mean)
        z_pca = pca.fit_transform(zs)
        zd_pca = pd.DataFrame(z_pca, columns=[f'Z_PC{i}' for i in range(z_pca.shape[1])])
        zd_pca['hue'] = pd.qcut(y_test.astype(float), nlvl, labels=False)
        sns.pairplot(data=zd_pca, vars=zd_pca.columns[:n_pca], hue='hue',
                     palette=sns.color_palette('BrBG', nlvl), 
                     height=height, plot_kws=plot_kws)
    elif n_dim:
        zd = pd.DataFrame(z_mean, columns=[f'Z{i}' for i in range(z_mean.shape[1])])
        zd['hue'] = pd.qcut(y_test.astype(float), nlvl, labels=False)
        sns.pairplot(data=zd, vars=zd.columns[:n_dim], hue='hue',
                     palette=sns.color_palette('GnBu_d', nlvl), 
                     height=height, plot_kws=plot_kws)
    else:
        print('Pass n_dim or n_pca!!')


        
# display a grid of plots for all combinations of latent dimensions
def plot_xcorr(encoder, data, ax,
               batch_size=512, pca=False,
               labels=False):
    x_test, y_test = data
    z_mean, z_log_var = encoder.predict(x_test, batch_size=batch_size)
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    if pca:
        scaler = StandardScaler()
        pca = PCA(n_components=None, svd_solver='randomized')
        zs = scaler.fit_transform(z_mean)
        z_pca = pca.fit_transform(zs)
        zd_pca = pd.DataFrame(z_pca, columns=[f'PC{i}' for i in range(z_pca.shape[1])])
        x = zd_pca
    else:
        zd = pd.DataFrame(z_mean, columns=[f'Z{i}' for i in range(z_mean.shape[1])])
        x = zd
    
    corr = pd.concat([x, y_test], axis=1, keys=['df1', 'df2']).corr().loc['df1', 'df2']
    sns.heatmap(data=corr, cmap=cmap, vmin=-0.9, vmax=0.9,
                center=0, square=True, annot=labels, fmt='.2f',
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_ylim(len(corr)+0.5, -0.5)
    return corr


# display a grid of plots for all combinations of latent dimensions
def plot_xcorr_cvae(encoder, data, ax,
               batch_size=512, pca=False,
               labels=False):
    x_test, y_test = data
    x_input = np.concatenate((x_test, y_test[['azimuth_norm', 'elevation_norm']]), axis=1)
    z_mean, z_log_var = encoder.predict(x_input, batch_size=batch_size)
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    if pca:
        scaler = StandardScaler()
        pca = PCA(n_components=None, svd_solver='randomized')
        zs = scaler.fit_transform(z_mean)
        z_pca = pca.fit_transform(zs)
        zd_pca = pd.DataFrame(z_pca, columns=[f'PC{i}' for i in range(z_pca.shape[1])])
        x = zd_pca
    else:
        zd = pd.DataFrame(z_mean, columns=[f'Z{i}' for i in range(z_mean.shape[1])])
        x = zd
    
    corr = pd.concat([x, y_test], axis=1, keys=['df1', 'df2']).corr().loc['df1', 'df2']
    sns.heatmap(data=corr, cmap=cmap, vmin=-0.9, vmax=0.9,
                center=0, square=True, annot=labels, fmt='.2f',
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_ylim(len(corr)+0.5, -0.5)
    return corr
 
        
# display reconstructed images
def plot_reconstructions(encoder, decoder, data, axs, batch_size=512, freq_loss=False, side='h'):
    x_test, y_test = data
    img_size = x_test.shape[1:-1]
    
    n_imgs = len(axs.flatten())
    z_mean, z_log_var = encoder.predict(x_test[:n_imgs], batch_size=batch_size)
    if freq_loss:
        z_mean = np.concatenate([z_mean, y_test[:n_imgs,np.newaxis]], axis=-1)
    
    for i, ax in enumerate(axs.flatten()):
        n = int(float(len(x_test) / len(axs.flatten()) * i))
        # reconstruct
        z_sample = z_mean[np.newaxis, i]
        x_decoded = decoder.predict(z_sample)
        # generate image
        img_input = x_test[i,...,0]
        img_decoded = x_decoded[0].reshape(img_size)
        img = np.hstack((img_input, img_decoded)) if side=='h' else np.vstack((img_input, img_decoded))
        # show in plot
        im = ax.imshow(img, cmap='Greys_r')
        ax.axis('off')
        
        
# display reconstructed hrtfs
def plot_reconstruction_hrtfs(encoder, decoder, data, axs, batch_size=512, elevation=0):
    x_test, y_test = data
    img_size = x_test.shape[1:-1]
    
    configs = sio.loadmat('./data/hutubs_hrtf/configs.mat')
    freqs = configs['f'][0]
    elevs = configs['elevations'][0]
    # elevation to index
    el_index = np.where(elevs == elevation)[0][0]
    
    step = len(x_test) // len(axs.flatten())
    z_mean, z_log_var = encoder.predict(x_test[::step], batch_size=batch_size)
    
    for i, ax in enumerate(axs.flatten()):
        n = i * step
        # reconstruct
        z_sample = z_mean[np.newaxis, i]
        x_decoded = decoder.predict(z_sample)
        # show in plot
        hrtf_true = x_test[n,el_index,:,0]
        hrtf_pred = x_decoded[0,el_index,:,0]
        line_true, = ax.plot(freqs, hrtf_true, label='true')
        line_pred, = ax.plot(freqs, hrtf_pred, label='pred')
        ax.legend(handles=[line_true, line_pred])
        ax.set_title('#{:02}{} - ({:.0f}° ; {:.0f}°)'.format(
                y_test['id'].iloc[n], 
                y_test['ear'].iloc[n][0].upper(),
                y_test['azimuth'].iloc[n],
                elevation))
        ax.set_ylim([-60, 20])
        ax.set_yticks(np.arange(-60, 21, 20))
        ax.yaxis.grid()
        #ax.axis('off')

        
# display reconstructed hrtfs
def plot_reconstructions_3d(encoder, decoder, data, axs, batch_size=512):
    x_test, y_test = data
    img_size = x_test.shape[1:-1]
    
    configs = sio.loadmat('./data/hutubs_hrtf/configs.mat')
    freqs = configs['f'][0]
    elevs = configs['elevations'][0]
    
    step = len(x_test) // len(axs.flatten())
    z_mean, z_log_var = encoder.predict(x_test[::step], batch_size=batch_size)
    
    for i, ax in enumerate(axs.flatten()):
        n = i * step
        # reconstruct
        z_sample = z_mean[np.newaxis, i]
        x_decoded = decoder.predict(z_sample)
        # show in plot
        hrtf_true = x_test[n,3,3,:]
        hrtf_pred = x_decoded[0,3,3,:]
        line_true, = ax.plot(freqs, hrtf_true, label='true')
        line_pred, = ax.plot(freqs, hrtf_pred, label='pred')
        ax.legend(handles=[line_true, line_pred])
        ax.set_title('#{:02}{} ({:.0f}° ; {:.0f}°)'.format(
                y_test['id'].iloc[n], 
                y_test['ear'].iloc[n][0].upper(),
                y_test['azimuth'].iloc[n],
                y_test['elevation'].iloc[n]))
        ax.set_ylim([-60, 20])
        ax.set_xlim([0, 18000])
        ax.set_yticks(np.arange(-60, 21, 20))
        ax.yaxis.grid()
        #ax.axis('off')
        
                
# display reconstructed hrtfs from one-hot encoding
def plot_reconstructions_3d_oh(encoder, decoder, data, axs, batch_size=512):
    x_test, y_test = data
    img_size = x_test.shape[1:-1]
    
    configs = sio.loadmat('./data/hutubs_hrtf/configs.mat')
    freqs = configs['f'][0]
    elevs = configs['elevations'][0]
    
    step = len(x_test) // len(axs.flatten())
    z_mean, z_log_var = encoder.predict(x_test[::step], batch_size=batch_size)
    
    for i, ax in enumerate(axs.flatten()):
        n = i * step
        # reconstruct
        z_sample = z_mean[np.newaxis, i]
        x_decoded = decoder.predict(z_sample)
        # show in plot
        hrtf_true = x_test[n,3,3,:]
        vmin = -60
        vmax = 20
        hrtf_pred = (np.argmax(x_decoded[0], axis=1)/255*(vmax-vmin))+vmin
        line_true, = ax.plot(freqs, hrtf_true, label='true')
        line_pred, = ax.plot(freqs, hrtf_pred, label='pred')
        ax.legend(handles=[line_true, line_pred])
        ax.set_title('#{:02}{} ({:.0f}° ; {:.0f}°)'.format(
                y_test['id'].iloc[n], 
                y_test['ear'].iloc[n][0].upper(),
                y_test['azimuth'].iloc[n],
                y_test['elevation'].iloc[n]))
        #ax.set_ylim([-60, 20])
        ax.set_xlim([0, 18000])
        #ax.set_yticks(np.arange(-60, 21, 20))
        ax.yaxis.grid()
        #ax.axis('off')
        
        
# display reconstructed hrtfs (chen et al. 2019 paper)
def plot_reconstructions_chen2019(encoder, decoder, data, axs, batch_size=512, x_train_mean=0, x_train_std=1, show_axes=False):
    x_test, y_test = data
    step = len(x_test) // len(axs.flatten())
    x = x_test[::step]
    y = y_test.iloc[::step]
    
    # load configs
    configs = sio.loadmat('./data/hutubs_hrtf/configs.mat')
    freqs = configs['f'][0]
    
    # encode data
    z_mean, z_log_var = encoder.predict(x, batch_size=batch_size)
        
    for i, ax in enumerate(axs.flatten()):
        # reconstruct
        z_sample = np.concatenate((z_mean[i], y[['azimuth_norm', 'elevation_norm']].iloc[i]))
        x_decoded = decoder.predict(z_sample[np.newaxis,:])
        # show in plot
        hrtf_true = (x[i] * x_train_std) + x_train_mean
        hrtf_pred = (x_decoded[0] * x_train_std) + x_train_mean
        line_true, = ax.plot(freqs, hrtf_true, label='true')
        line_pred, = ax.plot(freqs, hrtf_pred, label='pred')
        
        ax.set_title('#{:02}{} ({:.0f}° ; {:.0f}°)'.format(
                y['id'].iloc[i], 
                y['ear'].iloc[i][0].upper(),
                y['azimuth'].iloc[i],
                y['elevation'].iloc[i]))
        ax.set_ylim([-40, 20])
        ax.set_xlim([0, 18000])
        if show_axes:
            ax.legend(handles=[line_true, line_pred])
            ax.set_yticks(np.arange(-40, 21, 10))
            ax.yaxis.grid()
        else:
            ax.axis('off')
        

# display reconstructed hrtfs (CVAE)
def plot_reconstructions_cvae(encoder, decoder, data, axs, batch_size=512, x_train_mean=0, x_train_std=1, show_axes=False):
    x_test, y_test = data
    step = len(x_test) // len(axs.flatten())
    x = x_test[::step]
    y = y_test.iloc[::step]
    print(x.shape, y[['azimuth_norm', 'elevation_norm']].shape)
    x_input = np.concatenate((x, y[['azimuth_norm', 'elevation_norm']]), axis=1)
    print(x_input.shape)
    
    # load configs
    configs = sio.loadmat('./data/hutubs_hrtf/configs.mat')
    freqs = configs['f'][0]
    
    # encode data
    z_mean, z_log_var = encoder.predict(x_input, batch_size=batch_size)
        
    for i, ax in enumerate(axs.flatten()):
        # reconstruct
        z_sample = np.concatenate((z_mean[i], y[['azimuth_norm', 'elevation_norm']].iloc[i]))
        x_decoded = decoder.predict(z_sample[np.newaxis,:])
        # show in plot
        hrtf_true = (x[i] * x_train_std) + x_train_mean
        hrtf_pred = (x_decoded[0] * x_train_std) + x_train_mean
        line_true, = ax.plot(freqs, hrtf_true, label='true')
        line_pred, = ax.plot(freqs, hrtf_pred, label='pred')
        
        ax.set_title('#{:02}{} ({:.0f}° ; {:.0f}°)'.format(
                y['id'].iloc[i], 
                y['ear'].iloc[i][0].upper(),
                y['azimuth'].iloc[i],
                y['elevation'].iloc[i]))
        ax.set_ylim([-40, 20])
        ax.set_xlim([0, 18000])
        if show_axes:
            ax.legend(handles=[line_true, line_pred])
            ax.set_yticks(np.arange(-40, 21, 10))
            ax.yaxis.grid()
        else:
            ax.axis('off')
        

        
## OLD FUNCTION (TODO put sandom sampling into new function)
def plot_results(models,
                 data,
                 batch_size=128,
                 digit_size=100,
                 figsize=(12, 4),
                 grid_range=(-1, 1),
                 n=10,
                 ndims=2,
                 dims=[0,1],
                 model_name="vae_mnist"):
    
    reconstruction = False
    
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    encoder, decoder = models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, z_log_var, z = encoder.predict(x_test, batch_size=batch_size)
        
    sc = ax[0].scatter(z_mean[:, dims[0]], z_mean[:, dims[1]], c=y_test, alpha=0.75)
    plt.colorbar(sc, ax=ax)
    ax[0].set_xlabel("z[{}]".format(dims[0]))
    ax[0].set_ylabel("z[{}]".format(dims[1]))

    
    # display a 30x30 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(*grid_range, n)
    grid_y = np.linspace(*grid_range, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if not reconstruction:
                z_sample = np.zeros((1, ndims))
                z_sample[:, dims[0]] = xi
                z_sample[:, dims[1]] = yi
            else:
                z_sample = z_mean[np.newaxis, i*n+j]
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    if not reconstruction:
        ax[1].set_xticks(pixel_range)
        ax[1].set_yticks(pixel_range)
        ax[1].set_xticklabels(sample_range_x)
        ax[1].set_yticklabels(sample_range_y)
    ax[1].set_xlabel("z[{}]".format(dims[0]))
    ax[1].set_ylabel("z[{}]".format(dims[1]))
    ax[1].imshow(figure, cmap='Greys_r')
    
    
## TODO use same visualization as `ear_img_dimred`