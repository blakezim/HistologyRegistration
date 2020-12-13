import os
import sys
import glob
import torch
import numpy as np
# import PyCA.Core as ca
from tkinter import Tk
# import PyCA.Common as common
import CAMP.FileIO as io
import CAMP.Core as core
from scipy.io import loadmat
import CAMP.StructuredGridOperators as so
from avocado.RabbitCommon import Common as rc
from avocado.RabbitCommon import Config
# import PyCACalebExtras.Common as cc
from tkinter.filedialog import askopenfilename


def _valspec(_):
    """Hook to do extra validation"""
    pass


def _getFilePath(name, initialdir):
    Tk().withdraw()
    return askopenfilename(title=name, initialdir=initialdir)


RigidConfigSpec = {
    'sourcePointsFile':
        Config.Param(default='None',
                     comment='Text file that contains the rigid registration landmark points for the source volume.'),
    'targetPointsFile':
        Config.Param(default='None',
                     comment='Text file that contains the rigid registration landmark points for the target volume.'),
    'rigidMatrix':
        Config.Param(default=None,
                     comment='Affine matrix that takes source landmark points to target landmarks points with least squares fit.'),
    'outputPath':
        Config.Param(default=None,
                     comment='Path to the rigidly registered volumes. '),
    'rerun':
        Config.Param(default=False,
                     comment='Allows the user to rerun the rigid registration if the points are updated.'),
    '_validation_hook': _valspec
}


def _get_center_pos(d):

    try:
        pos = d['slicePos'][0, 0]
        if pos.size == 1:
            num_slice = 1
        else:
            num_slice = pos.shape[-1]
    except IndexError:
        raise Exception('Unknown dictionary for extracting center position')

    try:
        if num_slice == 1:
            cor_pos = [float(pos['dCor'])]
        else:
            cor_pos = np.concatenate(pos['dCor'][0]).squeeze().tolist()
    except ValueError:
        cor_pos = [0.0] * num_slice

    try:
        if num_slice == 1:
            tra_pos = [float(pos['dTra'])]
        else:
            tra_pos = np.concatenate(pos['dTra'][0]).squeeze().tolist()
    except ValueError:
        tra_pos = [0.0] * num_slice

    try:
        if num_slice == 1:
            sag_pos = [float(pos['dSag'])]
        else:
            sag_pos = np.concatenate(pos['dSag'][0]).squeeze().tolist()
    except ValueError:
        sag_pos = [0.0] * num_slice

    positions = []
    for p in range(0, num_slice):
        positions.append([sag_pos[p], tra_pos[p], cor_pos[p]])

    return positions


def _get_resolution(d):

    try:
        resolution = d['res']
    except IndexError:
        raise Exception('Unknown dictionary for extracting resolution')

    while len(resolution) == 1:
        resolution = resolution[0]

    return resolution.tolist()


def _generate_stir_vol(stir_center, stir_space, stir_data):

    # z_space = []
    # for i, c in enumerate(stir_center[1:]):
    #     z_space.append(c[2] - stir_center[i][2])

    # z_space = 2.0

    # z_slab = np.round(np.array(z_space).mean(), 3)
    z_slab = 2.0
    z_org = np.array(stir_center)[:, 2].min()
    vol_org = [stir_center[0][0] - ((stir_data.shape[0] / 2) * stir_space[0]),
               stir_center[0][1] - ((stir_data.shape[1] / 2) * stir_space[1]),
               z_org + z_slab / 2.0]

    vol_space = [stir_space[0], stir_space[1], z_slab]

    # stir_vol = core.StructuredGrid(
    #     [stir_data.shape[0], stir_data.shape[1], stir_data.shape[2]],
    #     spacing=vol_space,
    #     origin=vol_org,
    #     tensor=torch.tensor(stir_data).unsqueeze(0),
    #     channels=1
    # )

    stir_vol = core.StructuredGrid(
        [stir_data.shape[0], stir_data.shape[2], stir_data.shape[1]],
        spacing=[vol_space[0], vol_space[2], vol_space[1]],
        origin=[vol_org[0], vol_org[2], vol_org[1]],
        tensor=torch.tensor(stir_data.transpose(0, 2, 1)).unsqueeze(0).flip(-2),
        channels=1
    )

    return stir_vol


def _get_stir_vol(stir_dicts):

    t1s = []
    centers = []
    resolutions = []

    for d in stir_dicts:
        centers += _get_center_pos(d['param'][0, 0])
        resolutions.append(_get_resolution(d['param'][0, 0]))
        t1s.append(d['T1'])  # Scale from mSeconds to Seconds

    sorted_inds = np.argsort(np.array(centers)[:, -1])

    stacked_t1s = np.concatenate(t1s, -1)

    sorted_centers = []
    sorted_t1s = np.zeros_like(stacked_t1s)

    for i in range(t1s[0].shape[-1]):
        sorted_t1s[..., i*2] = t1s[1][..., i]
        sorted_t1s[..., (i*2)+1] = t1s[0][..., i]
        sorted_centers.append(centers[10 + i])
        sorted_centers.append(centers[i])

    # for i, ind in enumerate(sorted_inds):
    #     sorted_centers.append(centers[ind])
    #     sorted_t1s[..., i] = stacked_t1s[..., ind]

    t1_vol = _generate_stir_vol(sorted_centers, resolutions[0], sorted_t1s)

    return t1_vol


def ExternalRecon(regObj, opt):

    rerun = False

    if not os.path.exists(f'{regObj.rabbitDirectory}/externalRecons/'):
        print('No external files to recon ... skipping')
        return
    print('Reconning External Files ... ', end='')

    # Get a list of the files
    ablation_files = sorted(glob.glob(f'{regObj.rabbitDirectory}/externalRecons/AblationImaging/*'))

    pre_files = [x for x in ablation_files if 'Pre' in x]
    post_files = [x for x in ablation_files if 'Post' in x]

    pre_dicts = [loadmat(x) for x in pre_files]
    post_dicts = [loadmat(x) for x in post_files]

    pre_vol = _get_stir_vol(pre_dicts)
    post_vol = _get_stir_vol(post_dicts)

    # Get a list of the volumes in the ablation imaging already so we can assign the right number.
    files = glob.glob(f'{regObj.rabbitDirectory}/rawVolumes/AblationImaging/*')
    start_num = len(files)
    out_dir = f'{regObj.rabbitDirectory}/rawVolumes/AblationImaging/'
    pre_name = '_----_pre_ablation_t1_map.nii.gz'
    post_name = '_----_post_ablation_t1_map.nii.gz'

    if not any([pre_name in x for x in files]) or rerun:
        io.SaveITKFile(pre_vol, f'{out_dir}{start_num + 1:03d}{pre_name}')

    if not any([post_name in x for x in files]) or rerun:
        io.SaveITKFile(post_vol, f'{out_dir}{start_num + 2:03d}{post_name}')

    long_files = sorted(glob.glob(f'{regObj.rabbitDirectory}/externalRecons/PostImaging/*'))
    long_dicts = [loadmat(x) for x in long_files]
    long_vol = _get_stir_vol(long_dicts)

    files = glob.glob(f'{regObj.rabbitDirectory}/rawVolumes/PostImaging/*')
    start_num = len(files)
    out_dir = f'{regObj.rabbitDirectory}/rawVolumes/PostImaging/'
    long_name = '_----_t1_map.nii.gz'

    if not any([long_name in x for x in files]) or rerun:
        io.SaveITKFile(long_vol, f'{out_dir}{start_num + 1:03d}{long_name}')

    print('Done')

#### TO DO ####
# Make it so that this recon doesn't run every time