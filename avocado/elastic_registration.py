import os
import sys
import glob
import torch
import numpy as np

import CAMP.FileIO as io
import CAMP.Core as core
import CAMP.StructuredGridOperators as so
from CAMP import StructuredGridTools as st
# import IDiff.Matching
# import PyCA.Core as ca
# import PyCA.Common as common
# import PyCACalebExtras.Common as cc
# import PyCACalebExtras.Display as cd
# import PyCACalebExtras.SetBackend
#
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


# import PyCAApps as apps

from tkinter import Tk
from avocado.RabbitCommon import Config
from avocado.RabbitCommon import Common as rc
from tkinter.filedialog import askopenfilename


def _valspec(_):
    """Hook to do extra validation"""
    pass


def _getFilePath(name, initialdir):
    Tk().withdraw()
    return askopenfilename(title=name, initialdir=initialdir)


ElastConfigSpec = {
    'outputPath':
        Config.Param(default=None,
                     comment='Output location of the intensity based registration deformed images'),
    'sourceVarESig':
        Config.Param(default=5,
                     comment='Sigma for local variance equlization of the source image.'),
    'targetVarESig':
        Config.Param(default=5,
                     comment='Sigma for local variance equlization of the target image.'),
    'gaussSigma':
        Config.Param(default=[0.1],
                     comment='Sigma for gaussian blurring of both images.'),
    'gaussKernel':
        Config.Param(default=[3],
                     comment='Kernel size for gaussian blurring of both images.'),
    'croppedImage':
        Config.Param(default='None',
                     comment='A cropped image that only include the region of interest in real coordinates.'),
    'scales':
        Config.Param(default=[2, 1],
                     comment='A factor by which to downsample the image.'),
    'rerun':
        Config.Param(default=False,
                     comment='Allows the user to rerun the idiff registration if the parameters are updated.'),
    'I0':
        Config.Param(default='None',
                     comment='Source Image for the registration.'),
    'I1':
        Config.Param(default='None',
                     comment='Target Image for the registration.'),
    'I0_mask':
        Config.Param(default='None',
                     comment='Mask for the Source Image for the registration.'),
    'I1_mask':
        Config.Param(default='None',
                     comment='Mask for the Target Image for the registration.'),
    'medianFilterSize':
        Config.Param(default='None',
                     comment='Kernel size for the median filter'),
    'tvWeight':
        Config.Param(default=0.03,
                     comment='Kernel size for the median filter'),
    'Niter':
        Config.Param(default=[50, 50],
                     comment='Number of iterations for the registration.'),
    'stepSize':
        Config.Param(default=[0.01, 0.01],
                     comment='Step size for the registration.'),
    'incompressible':
        Config.Param(default=False,
                     comment='Volume Preserving registration.'),
    'regWeight':
        Config.Param(default=[0.01, 0.05],
                     comment='Regularization weight for the registration.'),
    'finalDeformation':
        Config.Param(default=None,
                     comment='The final deformation field output from Idiff registration'),
    '_validation_hook': _valspec
}


def MultiscaleElast(regObj, opt):

    device = opt.device

    try:
        src_name = regObj.I0
        tar_name = regObj.I1
        regObj.I1 = io.LoadITKFile(regObj.I1, device)
        regObj.I0 = io.LoadITKFile(regObj.I0, device)
    except RuntimeError:
        src_name = _getFilePath('Source Image File (ElastReg)',
                                initialdir='/home/sci/blakez/ucair/')
        tar_name = _getFilePath('Target Image File (ElastReg)',
                                initialdir='/home/sci/blakez/ucair/')
        regObj.I1 = io.LoadITKFile(tar_name, device)
        regObj.I0 = io.LoadITKFile(src_name, device)

    # Check if the croppedImage exists in the object - temperature object wont have it
    if regObj.croppedImage != 'None':
        # Load the cropped image which contains the region of interest
        try:
            crp_im = io.LoadITKFile(regObj.croppedImage, device)
        except RuntimeError:
            regObj.croppedImage = _getFilePath('Cropped Image File', initialdir='/home/sci/blakez/ucair/')
            crp_im = io.LoadITKFile(regObj.croppedImage, device)

        # Resample the images onto the ROI grid
        regObj.I1 = so.ResampleWorld.Create(crp_im, device=device)(regObj.I1)
        regObj.I0 = so.ResampleWorld.Create(crp_im, device=device)(regObj.I0)

        del crp_im

    # Make sure that both images are from 0 to 1
    regObj.I0 = regObj.I0 / regObj.I0.max()
    regObj.I1 = regObj.I1 / regObj.I1.max()

    # Median filter if the regObj has a kernel size for it
    if regObj.medianFilterSize != 'None':
        regObj.I0 = rc.MedianFilter_I(regObj.I0, regObj.medianFilterSize)
        regObj.I1 = rc.MedianFilter_I(regObj.I1, regObj.medianFilterSize)

    if regObj.tvWeight != 'None':
        regObj.I0 = rc.TVFilter_I(regObj.I0, regObj.tvWeight)
        regObj.I1 = rc.TVFilter_I(regObj.I1, regObj.tvWeight)

    # Variance Equalize volumes
    if regObj.sourceVarESig != 'None':
        kernel_size = regObj.sourceVarESig * 3
        regObj.I0 = so.VarianceEqualize.Create(kernel_size, regObj.sourceVarESig, device=device)(regObj.I0)
    if regObj.targetVarESig != 'None':
        kernel_size = regObj.sourceVarESig * 3
        regObj.I1 = so.VarianceEqualize.Create(kernel_size, regObj.sourceVarESig, device=device)(regObj.I1)

    # Check for masks: assume they mask out the water or bladder
    if regObj.I1_mask != 'None':
        try:
            mask = io.LoadITKFile(regObj.I1_mask, device)
        except RuntimeError:
            regObj.I1_mask = _getFilePath('Mask Image File (Target)', initialdir='/')
            mask = io.LoadITKFile(regObj.I1_mask, device)
        mask = so.ResampleWorld.Create(regObj.I1, device=opt.device)(mask)
        regObj.I1 = regObj.I1 * ((1 - mask) * -1)
        del mask
        torch.cuda.empty_cache()

    if regObj.I0_mask != 'None':
        try:
            mask = io.LoadITKFile(regObj.I0_mask, device)
        except RuntimeError:
            regObj.I0_mask = _getFilePath('Mask Image File (Source)', initialdir='/')
            mask = io.LoadITKFile(regObj.I0_mask, device)
        mask = so.ResampleWorld.Create(regObj.I0, device=opt.device)(mask)
        regObj.I0 = regObj.I0 * ((1 - mask) * -1)
        del mask
        torch.cuda.empty_cache()

    # Gaussian blur the images - blue after masking so the edges are not as sharp
    if regObj.gaussSigma != 'None':
        blur = so.Gaussian.Create(1, regObj.gaussKernel * 3, regObj.gaussSigma * 3,
                                  dim=3, device=device, dtype=torch.float32)
        regObj.I0 = blur(regObj.I0)
        regObj.I1 = blur(regObj.I1)

    deformation = regObj.I1.clone()
    deformation.set_to_identity_lut_()
    deformation_list = []

    # Create a grid composer
    composer = so.ComposeGrids(device=device, dtype=torch.float32, padding_mode='border')
    #
    for i, s in enumerate(regObj.scales):

        scale_source = regObj.I0.set_size(regObj.I1.size // s, inplace=False)
        scale_target = regObj.I1.set_size(regObj.I1.size // s, inplace=False)
        deformation = deformation.set_size(regObj.I1.size // s, inplace=False)

        # Apply the deformation to the source image
        scale_source = so.ApplyGrid(deformation)(scale_source)

        # Create the matching term
        similarity = so.L2Similarity.Create(dim=3, device=device)

        # Create the smoothing operator
        operator = so.FluidKernel.Create(
            scale_target,
            device=device,
            alpha=1.0,
            beta=0.0,
            gamma=0.001,
        )

        # Create the regularizer
        regularizer = so.NormGradient.Create(
            weight=regObj.regWeight[i],
            device=device,
            dtype=regObj.I1.dtype,
            dim=3
        )

        # Now register the source and the gad volume
        interday = st.IterativeMatch.Create(
            source=scale_source,
            target=scale_target,
            similarity=similarity,
            operator=operator,
            device=device,
            step_size=regObj.stepSize[i],
            regularization=regularizer,
            incompressible=regObj.incompressible
        )

        energy = [interday.initial_energy]
        print(f'Iteration: 0   Energy: {interday.initial_energy}')
        for i in range(1, regObj.Niter[i]):
            energy.append(interday.step())
            print(f'Iteration: {i}   Energy: {energy[-1]}')

            # if energy[-1] > energy[-2] - (3e-4 * energy[-2]):
            #     break

        deformation = interday.get_field()
        deformation_list.append(deformation.clone().set_size(regObj.I1.size, inplace=False))
        deformation = composer(deformation_list[::-1])

    regObj.I0 = src_name
    regObj.I1 = tar_name

    return deformation


def main(regObj, opt, mask=False):

    ### Image Matching
    # Create the image dirs if not already there
    regObj.elast.outputPath = regObj.rabbitDirectory + '/elastVolumes/' + regObj.pipelinePhase
    if not os.path.exists(regObj.elast.outputPath):
        os.makedirs(regObj.elast.outputPath)

    # Determing if the image registration has been done or if it needs to be rerun
    if regObj.elast.finalDeformation == 'None' or regObj.elast.rerun:
        h_final = MultiscaleElast(regObj.elast, opt)
        out_spot = f'{regObj.rabbitDirectory}/elastVolumes/{regObj.pipelinePhase}/final_Deformation_Is_to_It.mha'
        regObj.elast.finalDeformation = out_spot
        io.SaveITKFile(h_final, regObj.elast.finalDeformation)

    # Check if the deformation need to be loaded
    if 'h_final' not in locals():
        h_final = io.LoadITKFile(regObj.elast.finalDeformation, opt.device)

    # Apply the deformations to the source volumes
    print('Applying elast transformation to source volumes .... ', end='')
    sys.stdout.flush()

    # Try and and get the TPS volumes

    fileList = sorted(glob.glob(regObj.tps.outputPath + '/*'))

    if fileList == []:
        fileList = sorted(glob.glob(regObj.rigid.outputPath + '/*'))
    if fileList == []:
        fileList = regObj.sourceVolumes

    # Iterate through the volumes and apply the
    for volume in fileList:
        fileName = volume.split('/')[-1]
        fileName = fileName[:7] + 'e' + fileName[8:]
        if regObj.pipelinePhase == 'day0_motion' or regObj.pipelinePhase == 'day3_motion':
            fileName = fileName[:4] + 'm' + fileName[5:]
        outName = os.path.join(os.path.dirname(regObj.elast.finalDeformation), fileName)

        if not os.path.exists(outName) or regObj.elast.rerun:
            # Load the source image
            src_im = io.LoadITKFile(volume, device=opt.device)

            # Determine the subvol onto which the image needs to be resampled
            h_resample = so.ResampleWorld.Create(src_im, device=opt.device)(h_final)

            # Apply and save
            if any(x in outName for x in ['MOCO', 'ADC', 't1_map']):
                def_im = so.ApplyGrid.Create(h_resample, device=opt.device, interp_mode='nearest')(src_im, h_resample)
            else:
                def_im = so.ApplyGrid.Create(h_resample, device=opt.device)(src_im, h_resample)
            io.SaveITKFile(def_im, outName)

    print('Done')
