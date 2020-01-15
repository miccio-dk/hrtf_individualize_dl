# system modules
import re
import json
import time
import os.path as osp
from glob import glob
# extra modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import scipy.io as sio


blacklist = [1, 88, 96]
blacklist.extend([18, 79, 92])
ids = [i for i in range(1, 97) if i not in blacklist]


# apply salt-pepper to image
# https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#a7b0
def add_salt_pepper(X_img):
    # Need to produce a copy as to not modify the original image
    X_img_copy = X_img.copy()
    row, col = X_img_copy.shape
    salt_vs_pepper = 0.5
    amount = 0.01
    num_salt = np.ceil(amount * X_img_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_img_copy.size * (1.0 - salt_vs_pepper))
    
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1]] = 255

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img_copy.shape]
    X_img_copy[coords[0], coords[1]] = 0
    return X_img_copy


## load enthropometric measurement data into dataframe
def load_anthropometrics(data_path):
    df = pd.read_csv(data_path, index_col=0).dropna()
    # split in two sets
    l_cols = [c for c in df.columns if 'R_' not in c]
    r_cols = [c for c in df.columns if 'L_' not in c]
    df_l = df[l_cols].copy()
    df_r = df[r_cols].copy()
    # remove R_ and L_ from column name
    df_l.rename(columns={c: c if 'L_' not in c else c[2:] for c in df_l.columns}, inplace=True)
    df_r.rename(columns={c: c if 'R_' not in c else c[2:] for c in df_r.columns}, inplace=True)
    # add left/right to index
    df_l.index = pd.MultiIndex.from_tuples([(i, 'left') for i in df_l.index])
    df_r.index = pd.MultiIndex.from_tuples([(i, 'right') for i in df_r.index])
    # merge into one df
    df = pd.concat([df_l, df_r])
    df.index.names = ['id', 'ear']
    # add targets (left notch freq)
    target_path_l = osp.join(osp.dirname(data_path), 'n1_l.txt')
    trgt_l = pd.read_csv(target_path_l, header=None, names=['n1'])
    trgt_l.index = pd.MultiIndex.from_tuples([(i+1, 'left') for i in trgt_l.index])
    trgt_l.index.names = ['id', 'ear']
    # add targets (right notch freq)     
    target_path_r = osp.join(osp.dirname(data_path), 'n1_r.txt')
    trgt_r = pd.read_csv(target_path_r, header=None, names=['n1'])
    trgt_r.index = pd.MultiIndex.from_tuples([(i+1, 'right') for i in trgt_r.index])
    trgt_r.index.names = ['id', 'ear']
    df['n1'] = pd.concat([trgt_l, trgt_r])['n1']
    return df


## load elevation-azimuth pictures from the HUTUBS dataset, for each freq
def load_hutubs_hrtf(dataset_path, anthropometrics_path, data_content='hrtfs', user_filters={}):
    # load params
    configs = sio.loadmat(osp.join(dataset_path, 'configs.mat'))
    freqs = configs['f'][0]
    dshape = (len(configs['elevations'][0]), len(configs['azimuths'][0]))
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        'freqs': freqs,
        **user_filters
    }    
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    df = df.reindex(pd.MultiIndex.from_product(
        [ids, df.index.levels[1]], 
        names=df.index.names))
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'])
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['freqs'])
    n_train = n_variations * len(ids_train)
    n_test = n_variations * len(ids_test)
    y_cols = ['id', 'ear', 'freq'] + list(df.columns)
    # init placeholders
    x_train = np.zeros((n_train, *dshape))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, *dshape))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through filters
    i_train=0
    i_test=0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n_train+n_test)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate path
            filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            p = osp.join(dataset_path, filename)   
            # load data
            mat = sio.loadmat(p)
            content = mat[data_content]
            for i, f in enumerate(freqs):
                # filter by freq
                if f not in filters['freqs']:
                    continue
                # collect target data
                adata = df.loc[(sid, ear)]
                if float(f) != float(f):
                    print(f)
                tdata = {
                    'id': sid,
                    'ear': ear,
                    'freq': float(f),
                    **adata
                }
                # store data
                data = content[i].T if ear=='left' else content[i].T[:,::-1]
                if sid in ids_train:
                    x_train[i_train] = data
                    y_train.loc[i_train] = tdata
                    i_train += 1
                    pbar.update(1)
                elif sid in ids_test:
                    x_test[i_test] = data
                    y_test.loc[i_test] = tdata
                    i_test += 1
                    pbar.update(1)
    pbar.close()
    return (x_train, y_train), (x_test, y_test)


## load elevation-frequency pictures from the HUTUBS dataset
def load_hutubs_hrtf_alt(dataset_path, anthropometrics_path, data_content='hrtfs', user_filters={}):
    # load params
    configs = sio.loadmat(osp.join(dataset_path, 'configs.mat'))
    azimuths = configs['azimuths'][0]
    dshape = (len(configs['elevations'][0]), len(configs['f'][0]))
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        'azimuths': azimuths,
        **user_filters
    }    
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    df = df.reindex(pd.MultiIndex.from_product(
        [ids, df.index.levels[1]], 
        names=df.index.names))
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'])
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['azimuths'])
    n_train = n_variations * len(ids_train)
    n_test = n_variations * len(ids_test)
    y_cols = ['id', 'ear', 'azimuth'] + list(df.columns)
    # init placeholders
    x_train = np.zeros((n_train, *dshape))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, *dshape))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through filters
    i_train=0
    i_test=0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n_train+n_test)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate path
            filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            p = osp.join(dataset_path, filename)   
            # load data
            mat = sio.loadmat(p)
            content = mat[data_content]
            for i, az in enumerate(azimuths):
                # filter by freq
                if az not in filters['azimuths']:
                    continue
                # collect target data
                adata = df.loc[(sid, ear)]
                tdata = {
                    'id': sid,
                    'ear': ear,
                    'azimuth': float(az),
                    **adata
                }
                # store data
                data = content[:,i].T[::-1]
                if sid in ids_train:
                    x_train[i_train] = data
                    y_train.loc[i_train] = tdata
                    i_train += 1
                    pbar.update(1)
                elif sid in ids_test:
                    x_test[i_test] = data
                    y_test.loc[i_test] = tdata
                    i_test += 1
                    pbar.update(1)
    pbar.close()
    return (x_train, y_train), (x_test, y_test)


## load "HRTF patches" from the HUTUBS dataset, as per `yamamoto_fully_2017`
def load_hutubs_yamo(dataset_path, anthropometrics_path, data_content='hrtfs', user_filters={}):
    # load params
    configs = sio.loadmat(osp.join(dataset_path, 'configs.mat'))
    azimuths = configs['azimuths'][0]
    elevations = configs['elevations'][0]
    dshape = (5, 5, len(configs['f'][0]))
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        'azimuths': azimuths,
        'elevations': elevations,
        **user_filters
    }    
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    df = df.reindex(pd.MultiIndex.from_product(
        [ids, df.index.levels[1]], 
        names=df.index.names))
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'], random_state=1337)
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['azimuths']) * len(filters['elevations'])
    print(n_variations)
    n_train = n_variations * len(ids_train)
    n_test = n_variations * len(ids_test)
    y_cols = ['id', 'ear', 'ear_n', 'azimuth', 'elevation'] + list(df.columns)
    # init placeholders
    x_train = np.zeros((n_train, *dshape))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, *dshape))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through filters
    i_train=0
    i_test=0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n_train+n_test)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate path
            filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            p = osp.join(dataset_path, filename)   
            # load data
            mat = sio.loadmat(p)[data_content]
            # for each azimuth and elevation..
            xxx = 0
            for i_az, az in enumerate(azimuths):
                if az not in filters['azimuths']:
                    continue
                for i_el, el in enumerate(elevations):
                    if el not in filters['elevations']:
                        continue
                    if i_el < 2 or i_el > len(elevations)-3:
                        continue
                    xxx += 1
                    #print(xxx)
                    # collect target data
                    adata = df.loc[(sid, ear)]
                    tdata = {
                        'id': sid,
                        'ear': ear,
                        'ear_n': {'left': 1, 'right': 2}[ear],
                        'azimuth': float(az),
                        'elevation': float(el),
                        **adata
                    }
                    # extract data
                    az_ind = range(i_az-2, i_az+3)
                    el_ind = range(i_el-2, i_el+3)
                    data = mat.take(az_ind, axis=1, mode='wrap')
                    data = data.take(el_ind, axis=2, mode='wrap')
                    data = np.moveaxis(data, 0, -1)
                    # store data
                    if sid in ids_train:
                        x_train[i_train] = data
                        y_train.loc[i_train] = tdata
                        i_train += 1
                        pbar.update(1)
                    elif sid in ids_test:
                        x_test[i_test] = data
                        y_test.loc[i_test] = tdata
                        i_test += 1
                        pbar.update(1)
    pbar.close()
    return (x_train[:i_train], y_train.iloc[:i_train]), (x_test[:i_test], y_test.iloc[:i_test])


## load HRTFs from the HUTUBS dataset, 1 hrtf per datapoint, 1 set
def load_hutubs_1(dataset_path, anthropometrics_path, data_content='hrtfs', user_filters={}):
    # load params
    configs = sio.loadmat(osp.join(dataset_path, 'configs.mat'))
    azimuths = configs['azimuths'][0]
    elevations = configs['elevations'][0]
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        'azimuths': azimuths,
        'elevations': elevations,
        **user_filters
    }    
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    df = df.reindex(pd.MultiIndex.from_product(
        [ids, df.index.levels[1]], 
        names=df.index.names))
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['azimuths']) * len(filters['elevations'])
    n = n_variations * len(filters['ids'])
    y_cols = ['id', 'ear', 'ear_n', 'azimuth', 'elevation'] + list(df.columns)
    # init placeholders
    X = np.zeros((n, len(configs['f'][0])))
    y = pd.DataFrame(columns=y_cols, index=np.arange(n))
    # loop through filters
    i=0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate path
            filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            p = osp.join(dataset_path, filename)   
            # load data
            mat = sio.loadmat(p)[data_content]
            # for each azimuth and elevation..
            for i_az, az in enumerate(azimuths):
                if az not in filters['azimuths']:
                    continue
                for i_el, el in enumerate(elevations):
                    if el not in filters['elevations']:
                        continue
                    # collect target data
                    adata = df.loc[(sid, ear)]
                    tdata = {
                        'id': sid,
                        'ear': ear,
                        'ear_n': {'left': 1, 'right': 2}[ear],
                        'azimuth': float(az),
                        'elevation': float(el),
                        **adata
                    }
                    # extract data
                    data = mat[:,i_az,i_el]
                    # store data
                    X[i] = data
                    y.iloc[i] = tdata
                    i += 1
                    pbar.update(1)
    pbar.close()
    return (X, y)



## load HRTFs, anthropometrics, and ear pictures from the HUTUBS dataset, 1 hrtf per datapoint, 1 set, as dataframe
def load_hutubs_1_ears(user_filters={}):
    # paths
    images_path = './data/hutubs_img3/'
    hrtfs_path = './data/hutubs_hrtf/'
    anthropometrics_path = './data/hutubs_measures.csv'
    # load params
    configs = sio.loadmat(osp.join(hrtfs_path, 'configs.mat'))
    azimuths = configs['azimuths'][0]
    elevations = configs['elevations'][0]
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        'azimuths': azimuths,
        'elevations': elevations,
        **user_filters
    }
    # load anthropometrics
    df_anthro = load_anthropometrics(anthropometrics_path)
    df_anthro = df_anthro.reindex(pd.MultiIndex.from_product(
        [ids, df_anthro.index.levels[1]], 
        names=df_anthro.index.names))
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['azimuths']) * len(filters['elevations'])
    n = n_variations * len(filters['ids'])
    cols = ['id', 'ear', 'ear_n', 'azimuth', 'elevation'] + list(df_anthro.columns) + ['depthmap', 'hrtf']
    # init placeholders
    data = pd.DataFrame(columns=cols, index=np.arange(n))
    # loop through filters
    i=0
    # reset tqdm state
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate hrtf path, load data
            hrtf_filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            mat = sio.loadmat(osp.join(hrtfs_path, hrtf_filename))['hrtfs']
            # generate depthmap path, load data
            depth_filename = f'pp{sid}_3DheadMesh.png'
            depth_filepath = osp.join(images_path, ear, '0_0_0_0', depth_filename)
            data_depth = np.asarray(Image.open(depth_filepath))
            # for each azimuth and elevation..
            for i_az, az in enumerate(azimuths):
                if az not in filters['azimuths']:
                    continue
                for i_el, el in enumerate(elevations):
                    if el not in filters['elevations']:
                        continue
                    # collect target data
                    data_anthro = df_anthro.loc[(sid, ear)]
                    data_meta = {
                        'id': sid,
                        'ear': ear,
                        'ear_n': {'left': 1, 'right': 2}[ear],
                        'azimuth': float(az if az<=180 else (az-360)),
                        'elevation': float(el),
                    }
                    # extract data
                    data_hrtf = mat[:,i_az,i_el]
                    # store data
                    data.iloc[i] = {
                        **data_anthro, 
                        **data_meta, 
                        'depthmap': data_depth,
                        'hrtf': data_hrtf
                    }
                    i += 1
                    pbar.update(1)
    pbar.close()
    return data


## load depthmap pictures from the HUTUBS dataset
def load_hutubs_3d(dataset_path, anthropometrics_path, data_content='hrtfs', user_filters={}):
    # load params
    configs = sio.loadmat(osp.join(dataset_path, 'configs.mat'))
    dshape = sio.loadmat(osp.join(dataset_path, 'subj_1_ear_1.mat'))['hrtfs'].shape
    # assemble filters
    filters = {
        'ids': ids,
        'ears': ['left', 'right'],
        **user_filters
    }    
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    df = df.reindex(pd.MultiIndex.from_product(
        [ids, df.index.levels[1]], 
        names=df.index.names))
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'])
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_train = len(filters['ears']) * len(ids_train)
    n_test = len(filters['ears']) * len(ids_test)
    y_cols = ['id', 'ear', 'freq'] + list(df.columns)
    # init placeholders
    x_train = np.zeros((n_train, *dshape))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, *dshape))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through filters
    i_train=0
    i_test=0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    pbar = tqdm(total=n_train+n_test)
    # for each subject and ear...
    for sid in filters['ids']:
        for ear in filters['ears']:
            # generate path
            filename = 'subj_{}_ear_{}.mat'.format(
                sid, 
                {'left': 1, 'right': 2}[ear])
            p = osp.join(dataset_path, filename)   
            # load data, anthro, and target
            mat = sio.loadmat(p)
            content = mat[data_content]
            adata = df.loc[(sid, ear)]
            tdata = {
                'id': sid,
                'ear': ear,
                'ear_n': {'left': 1, 'right': 2}[ear],
                **adata
            }
            # store data
            data = content if ear=='left' else content[:,::-1]
            if sid in ids_train:
                x_train[i_train] = data
                y_train.loc[i_train] = tdata
                i_train += 1
                pbar.update(1)
            elif sid in ids_test:
                x_test[i_test] = data
                y_test.loc[i_test] = tdata
                i_test += 1
                pbar.update(1)
    pbar.close()
    return (x_train, y_train), (x_test, y_test)


## load depthmap pictures from the HUTUBS dataset
def load_hutubs_depth(dataset_path, anthropometrics_path, user_filters={}, saltpepper=0):
    # load dataset config
    with open(osp.join(dataset_path, 'dataset_cfg.json')) as f:
        cfg = json.load(f)
    print('Loaded dataset configs.')
    # generate list of all files
    img_glob = osp.join(dataset_path, '**', '*.png')
    img_paths = glob(img_glob, recursive=True)
    img_paths.sort()        
    # generate list of subjects
    subjects = list(set([osp.splitext(osp.basename(p))[0] for p in img_paths]))
    subjects = [int(re.search('pp(.*)_', n).group(1)) for n in subjects]
    print(f'Found {len(subjects)} ids and {len(img_paths)} images in total.')
    # assemble filters
    filters = {
        'ids': subjects,
        'ears': ['left', 'right'],
        'azimuths': cfg['azimuths'],
        'elevations': cfg['elevations'],
        'xoffs': cfg['xoffs'],
        'yoffs': cfg['yoffs'],
        **user_filters
    }
    # load anthropometrics
    df = load_anthropometrics(anthropometrics_path)
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'])
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_variations = len(filters['ears']) * len(filters['elevations']) * len(filters['azimuths']) * len(filters['xoffs']) * len(filters['yoffs'])   
    n_train = n_variations * len(ids_train) * (saltpepper if saltpepper else 1)
    n_test = n_variations * len(ids_test)
    w = cfg['size']
    y_cols = ['id', 'ear', 'elevation', 'azimuth', 'xoffs', 'yoffs'] + list(df.columns)
    # init placeholders
    x_train = np.zeros((n_train, w, w))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, w, w))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through filters
    i_train=0
    i_test=0
    time.sleep(0.2)
    pbar = tqdm(total=n_train+n_test)
    for sid in filters['ids']:
        for ear in filters['ears']:
            for elevation in filters['elevations']:
                for azimuth in filters['azimuths']:
                    for x in filters['xoffs']:
                        for y in filters['yoffs']:
                            # generate path
                            var_dir = '{}_{}_{}_{}'.format(
                                int(elevation),
                                int(azimuth),
                                int(x*10000),
                                int(y*10000)                            
                            )
                            filename = f'pp{sid}_3DheadMesh.png'
                            p = osp.join(dataset_path, ear, var_dir, filename)   
                            # load file and collect target data
                            img = np.asarray(Image.open(p))
                            adata = df.loc[(sid, ear)]
                            tdata = {
                                'id': sid,
                                'ear': ear,
                                'elevation': elevation,
                                'azimuth': azimuth,
                                'xoffs': x,
                                'yoffs': y,
                                **adata
                            }
                            # store data
                            if sid in ids_train:
                                if saltpepper:
                                    for i in range(saltpepper):
                                        img_sp = add_salt_pepper(img)
                                        x_train[i_train] = img_sp
                                        y_train.loc[i_train] = tdata
                                        i_train += 1
                                        pbar.update(1)
                                else:
                                    x_train[i_train] = img
                                    y_train.loc[i_train] = tdata
                                    i_train += 1
                                    pbar.update(1)
                            elif sid in ids_test:
                                x_test[i_test] = img
                                y_test.loc[i_test] = tdata
                                i_test += 1
                                pbar.update(1)
    pbar.close()
    return (x_train, y_train), (x_test, y_test)


## load pictures from the AMI dataset
def load_ami(dataset_path, size=None, user_filters={}, saltpepper=0):
    fsize = size if size else (96, 96) # TODO read size from file?
    # generate list of all files
    img_glob = osp.join(dataset_path, '*.jpg')
    img_paths = glob(img_glob)
    img_paths.sort()
    # generate list of subjects
    ids = list(set([int(osp.basename(p).split('_')[0]) for p in img_paths]))
    print(f'Found {len(ids)} ids and {len(img_paths)} images in total.')
    # assemble filters
    filters = {
        'ids': ids,
        'variations': ['back', 'down', 'front', 'left', 'right', 'up', 'zoom'],
        **user_filters
    }
    # split train-test subjects
    ids_train, ids_test = train_test_split(filters['ids'])
    #print(ids_train, ids_test)
    print(f'Train/test split: {len(ids_train)}/{len(ids_test)} ids')
    # calculate useful parameters
    n_train = len(filters['variations']) * len(ids_train) * (saltpepper if saltpepper else 1)
    n_test  = len(filters['variations']) * len(ids_test)
    y_cols = ['id', 'variation']
    # init placeholders
    x_train = np.zeros((n_train, *fsize))
    y_train = pd.DataFrame(columns=y_cols, index=np.arange(n_train))
    x_test = np.zeros((n_test, *fsize))
    y_test = pd.DataFrame(columns=y_cols, index=np.arange(n_test))
    # loop through files
    i_train=0
    i_test=0
    pbar = tqdm(total=n_train+n_test)
    for p in img_paths:
        # load file and collect target data
        picture = Image.open(p).convert(mode='L').transpose(Image.FLIP_LEFT_RIGHT)
        width, height = picture.size   # Get dimensions
        left = (width - fsize[0])/2
        top = (height - fsize[1])/2
        right = (width + fsize[0])/2
        bottom = (height + fsize[1])/2
        picture = picture.crop().resize(fsize)
        img = np.asarray(picture)
        sid = int(osp.basename(p).split('_')[0])
        variation = osp.basename(p).split('_')[1]
        #print(sid, variation)
        tdata = {
            'id': sid,
            'variation': variation
        }
        # filter data
        if variation not in filters['variations']:
            continue
        # store data
        if sid in ids_train:
            if saltpepper:
                for i in range(saltpepper):
                    img_sp = add_salt_pepper(img)
                    x_train[i_train] = img_sp
                    y_train.loc[i_train] = tdata
                    i_train += 1
                    pbar.update(1)
            else:
                x_train[i_train] = img
                y_train.loc[i_train] = tdata
                i_train += 1
                pbar.update(1)
        elif sid in ids_test:
            x_test[i_test] = img
            y_test.loc[i_test] = tdata
            i_test += 1
            pbar.update(1)
    pbar.close()
    return (x_train, y_train), (x_test, y_test)


## load and split images by subject (OLD FUNCTION)
def load_data(dir_path, img_size=None, filters={}, target='name', rem_corners=False, rotations=[0], saltpepper=0):
    all_filters = {
        'name': None,
        'ear': None,
        'azimuth': None,
        'elevation': None,
        'xoffs': None,
        'yoffs': None,
        **filters
    }
    
    # list img files
    img_glob = osp.join(dir_path, '**', '*.png')
    img_paths = glob(img_glob, recursive=True)
    img_paths.sort()
    print(f'Found {len(img_paths)} images.')
    
    # filter and split data
    subjects = list(set([osp.splitext(osp.basename(p))[0] for p in img_paths]))
    subjects_kept = all_filters['name'] if all_filters['name'] is not None else [int(re.search('pp(.*)_', n).group(1)) for n in subjects]
    sub_train, sub_test = train_test_split(subjects_kept)
    #print(f'Train set: {len(sub_train)}')
    #print(f'Test set: {len(sub_test)}')

    # calculate params
    s = np.asarray(Image.open(img_paths[0])).shape
    offs = (0, 0) if img_size is None else ((s[0]-img_size[0])//2, (s[1]-img_size[1])//2)
    size = s if img_size is None else img_size
    
    # init placeholders
    n_elevations = 11 if all_filters['elevation'] is None else len(all_filters['elevation'])
    n_azimuths   = 11 if all_filters['azimuth'] is None else len(all_filters['azimuth'])
    n_xoffs      = 11 if all_filters['xoffs'] is None else len(all_filters['xoffs'])
    n_yoffs      = 11 if all_filters['yoffs'] is None else len(all_filters['yoffs'])
    n_ears       =  2 if all_filters['ear'] is None else len(all_filters['ear'])
    n_variations = n_elevations * n_azimuths * n_xoffs * n_yoffs * n_ears * len(rotations) * (saltpepper if saltpepper else 1)
    print(n_variations)
    # (len(img_paths) // len(subjects) * len(sub_train) * )    
    n_train = len(sub_train) * n_variations
    n_test  = len(sub_test)  * n_variations
    x_train = np.zeros((n_train, *size))
    x_test = np.zeros((n_test, *size))
    y_train = np.zeros((n_train), dtype=object)
    y_test = np.zeros((n_test), dtype=object)
    i_train = 0
    i_test = 0
    print(x_train.shape)
    print(x_test.shape)

    # create mask
    bg = Image.new("L", size, 0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, *size), fill=255)
        
    # load and add to matrix
    for p in tqdm(img_paths):
        coord = osp.basename(osp.dirname(p)).split('_')
        name = osp.splitext(osp.basename(p))[0]
        name = int(re.search('pp(.*)_', name).group(1))
        targets = {
            'name': name,
            'ear': osp.basename(osp.dirname(osp.dirname(p))),
            'azimuth': int(coord[1]),
            'elevation': int(coord[0]),
            'xoffs': int(coord[2]),
            'yoffs': int(coord[3])
        }
        # apply other filters
        filter_out = False
        for k in ['ear', 'azimuth', 'elevation', 'xoffs', 'yoffs']:
            if all_filters[k] is not None:
                if targets[k] not in all_filters[k]:
                    filter_out = True
        if filter_out:
            continue
        
        # place in correct set
        if name in sub_train:
            img = Image.open(p).crop([offs[0], offs[1], s[0]-offs[0], s[1]-offs[1]])
            if rem_corners:
                img = Image.composite(img, bg, mask)
            for r in rotations:
                img_r = img.rotate(r)
                targets['angle'] = r
                if saltpepper:
                    for i in range(saltpepper):
                        img_sp = add_salt_pepper(np.asarray(img_r))
                        x_train[i_train] = img_sp
                        y_train[i_train] = targets[target]
                        i_train += 1
                else:
                    x_train[i_train] = np.asarray(img_r)
                    y_train[i_train] = targets[target]
                    i_train += 1 
        elif name in sub_test:
            img = Image.open(p).crop([offs[0], offs[1], s[0]-offs[0], s[1]-offs[1]])
            if rem_corners:
                img = Image.composite(img, bg, mask)
            for r in rotations:
                img_r = img.rotate(r)
                targets['angle'] = r
                x_test[i_test] = np.asarray(img_r)
                y_test[i_test] = targets[target]
                i_test += 1
        else:
            #print('#### WHOS THIS IMG FROM??? ', name)
            pass
    return (x_train[:i_train], y_train[:i_train]), (x_test[:i_test], y_test[:i_test])
