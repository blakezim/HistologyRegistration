import os
import sys
import glob
import torch
import tkinter as tk
import subprocess as sp
import CAMP.camp.FileIO as io
import elastic_registration as ER
import CAMP.camp.StructuredGridOperators as so
from avocado.RabbitCommon import Config
from avocado.RabbitCommon import Common as rc
import CAMP.camp.FileIO as io
import CAMP.camp.Core as core

import numpy as np
import matplotlib.pyplot as plt
# import PyCA.Common as common
# from AppUtils import Config
# import PyCACalebExtras.Common as cc
# import ElastRegistration
# import PyCAApps as apps
import scipy.io as spi

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def _valspec(_):
    """Hook to do extra validation"""
    pass


def _getFilePath(name, initialdir):
    tk().withdraw()
    return askopenfilename(title=name, initialdir=initialdir)


TempConfigSpec = {
    'rabbitDirectory':
        Config.Param(comment='Top directory for the rabbit.'),
    'pipelinePhase':
        Config.Param(comment='What phase of registration the config is responsible for (Interday, Intraday, Exvivo.'),
    'dataDirectory':
        Config.Param(default='None',
                     comment='Raid 2 Directory where the orignal data is held.'),
    'reconTempData':
        Config.Param(default=False,
                     comment='Indicator if temperature data is ready for recon.'),
    'scales':
        Config.Param(default=[1],
                     comment='A factor by which to downsample the image.'),
    'outputPath':
        Config.Param(default=None,
                     comment='Path to the TPS registered volumes.'),
    'unionSpacing':
        Config.Param(default=None,
                     comment='The largest FOV that contains all FOVs of the sonications.'),
    'unionOrigin':
        Config.Param(default=None,
                     comment='The largest FOV that contains all FOVs of the sonications.'),
    'unionSize':
        Config.Param(default=None,
                     comment='The largest FOV that contains all FOVs of the sonications.'),
    'regType':
        Config.Param(default='temperature',
                     comment='Tag for the type of registration (mask vs image).'),
    'croppedImage':
        Config.Param(default='None',
                     comment='A cropped image that only include the region of interest in real coordinates.'),
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
    'medianFilterSize':
        Config.Param(default='None',
                     comment='Kernel size for the median filter'),
    'tvWeight':
        Config.Param(default=0.02,
                     comment='Kernel size for the median filter'),
    'incompressible':
        Config.Param(default=False,
                     comment='Volume Preserving registration.'),
    'rerun':
        Config.Param(default=False,
                     comment='Allows the user to rerun the idiff registration if the parameters are updated.'),
    'I0':
        Config.Param(default='None',
                     comment='Source Image for the registration.'),
    'I1':
        Config.Param(default='None',
                     comment='Target Image for the registration.'),
    'Niter':
        Config.Param(default=[40],
                     comment='Number of iterations for the registration.'),
    'I0_mask':
        Config.Param(default='None',
                     comment='Mask for the Source Image for the registration.'),
    'I1_mask':
        Config.Param(default='None',
                     comment='Mask for the Target Image for the registration.'),
    'stepSize':
        Config.Param(default=[0.008],
                     comment='Step size for the registration.'),
    'regWeight':
        Config.Param(default=[2.5],
                     comment='Regularization weight for the registration.'),
    '_validation_hook': _valspec
}


def _convert_mat_files(Obj):
    rabbitNumber = Obj.rabbitDirectory.split('/')[-1]

    matList = sorted(glob.glob(f'{Obj.rabbitDirectory}/externalRecons/AblationImaging/*'))

    # Do some string comparison to make sure that this file is a sonication file
    matList = [x for x in matList if 'Son' in x]

    # Have to keep track of the MIN and MAX of each transfor so we can know the bounding box to resample to
    size = []
    orgn = []

    if not os.path.exists(os.path.join(Obj.rabbitDirectory + '/rawVolumes/TemperatureRecon/RegVols/')):
        os.makedirs(os.path.join(Obj.rabbitDirectory + '/rawVolumes/TemperatureRecon/RegVols/'))

    for tempFile in matList:

        # Create the output folder for all of the timepoint images
        outDir = os.path.splitext(tempFile)[0]
        outName = outDir.split('/')[-1]
        dirOutName = os.path.join(Obj.rabbitDirectory + '/rawVolumes/TemperatureRecon/', outName)
        if not os.path.exists(dirOutName):
            os.makedirs(dirOutName)

        # Have to add mroe logic here because there are varying lengths of numbers FUN
        preLength = len(outName.split('_')[0])
        outName = outName[:preLength] + '_----' + outName[preLength:]

        struct = spi.loadmat(tempFile)
        imData = struct['ims']

        # Could put some logic here to make sure that they all have a 4th dimension
        timePts = np.shape(imData)[-1]

        # Get transformation and solve for the affine
        # tform = struct['PosDCS']

        # print('min: {0} .... '.format(np.min(np.min(np.min(tform, 0), 0), 0).tolist()))
        # print('max: {0} .... '.format(np.max(np.max(np.max(tform, 0), 0), 0).tolist()))

        # dim = np.shape(tform)[0:3]
        # dim = [x - 1 for x in dim]

        # print('shape: {0} .... '.format(dim))

        # Pull points from the transformation to solve for the affine
        # landmarks = []
        # landmarks.append([[0, 0, 0], tform[0, 0, 0].tolist()])
        # landmarks.append([[dim[0], 0, 0], tform[dim[0], 0, 0].tolist()])
        # landmarks.append([[0, dim[1], 0], tform[0, dim[1], 0].tolist()])
        # landmarks.append([[dim[0], dim[1], 0], tform[dim[0], dim[1], 0].tolist()])
        # landmarks.append([[0, 0, dim[2]], tform[0, 0, dim[2]].tolist()])
        # landmarks.append([[dim[0], 0, dim[2]], tform[dim[0], 0, dim[2]].tolist()])
        # landmarks.append([[0, dim[1], dim[2]], tform[0, dim[1], dim[2]].tolist()])
        # landmarks.append([[dim[0], dim[1], dim[2]], tform[dim[0], dim[1], dim[2]].tolist()])

        # Solve for the affine
        # affine = apps.SolveAffine(landmarks)
        affine = struct['geomInfo'][0,0]['AffineDCS']

        temp = core.StructuredGrid(
            [imData.shape[0], imData.shape[1], imData.shape[2]],
            tensor=torch.tensor(np.real(imData)).permute(-1, 0, 1, 2),
            origin= [0, 0, 0],
            channels=imData.shape[-1]
        )

        # temp = ca.Image3D(cc.MakeGrid(np.shape(tform)[0:3], ca.MEM_DEVICE), ca.MEM_DEVICE)
        # temp.setOrigin(ca.Vec3Df(0, 0, 0))
        # temp.setSpacing(ca.Vec3Df(1, 1, 1))

        affGrid = rc.SolveAffineGrid(temp, affine)

        size.append(affGrid.size.tolist())
        orgn.append(affGrid.origin.tolist())

        print('Converting files for {0} time series images .... '.format(timePts), end='')
        sys.stdout.flush()
        for time in range(0, timePts):

            magIm_np = np.real(imData[:, :, :, time])
            phsIm_np = np.imag(imData[:, :, :, time])

            magIm = core.StructuredGrid(
                [magIm_np.shape[0], magIm_np.shape[1], magIm_np.shape[2]],
                tensor=torch.tensor(magIm_np).unsqueeze(0).float(),
                origin=[0, 0, 0],
                channels=1
            )
            phsIm = core.StructuredGrid(
                [phsIm_np.shape[0], phsIm_np.shape[1], phsIm_np.shape[2]],
                tensor=torch.tensor(phsIm_np).unsqueeze(0).float(),
                origin=[0, 0, 0],
                channels=1
            )

            affFilt = so.AffineTransform.Create(affine=torch.tensor(affine).float())
            affMagIm = affFilt(magIm, out_grid=affGrid)
            affPhsIm = affFilt(phsIm, out_grid=affGrid)

            io.SaveITKFile(affMagIm, dirOutName + '/' + outName + '_real_{0}.nii.gz'.format(str(time).zfill(2)))
            io.SaveITKFile(affPhsIm, dirOutName + '/' + outName + '_imag_{0}.nii.gz'.format(str(time).zfill(2)))

            if time == 0:
                # We need to have the magnitude image for registration
                regIm_np = np.abs(imData[:, :, :, time])
                regIm = core.StructuredGrid(
                    [regIm_np.shape[0], regIm_np.shape[1], regIm_np.shape[2]],
                    tensor=torch.tensor(regIm_np).unsqueeze(0).float(),
                    origin=[0, 0, 0],
                    channels=1
                )
                affRegIm = affFilt(regIm, out_grid=affGrid)
                io.SaveITKFile(affRegIm, dirOutName + '/../RegVols/' + outName[0:5] + '.nii.gz')

        # print('size: {0} .... '.format(magIm_np.shape), end='')
        print('Done')

    return size, orgn


def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def tempRecon(Obj, opt):

    if not os.path.exists(f'{Obj.rabbitDirectory}/rawVolumes/TemperatureRecon/'):
        os.makedirs(f'{Obj.rabbitDirectory}/rawVolumes/TemperatureRecon/')
        size, orgn = _convert_mat_files(Obj)

        unionOrigin = np.min(orgn, 0).tolist()
        adj = np.abs(np.max(orgn, 0) - np.min(orgn, 0))
        unionSize = (np.max(size, 0) + adj).tolist()
        unionSize = [int(x) for x in unionSize]

    # We are going to create a config file for each registration
    sonicationList = natural_sort(glob.glob(Obj.rabbitDirectory + '/rawVolumes/TemperatureRecon/*'))

    # Need to remove the resample from the list
    sonicationList = [x for x in sonicationList if '_resample' not in x]
    sonicationList = [x for x in sonicationList if 'RegVols' not in x]

    lastName = sonicationList[-1].split('/')[-1]
    preLength = len(lastName.split('_')[0])
    lastName = lastName[:preLength] + '_----' + lastName[preLength:]

    # Set up the Config directory
    if not os.path.exists(Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/'):
        os.makedirs(Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/')

    # Construct teh common Grid: at this point assuming isotropic spacing of 1
    if 'unionSize' in locals():
        unionSpacing = [1.0, 1.0, 1.0]
        unionIm = core.StructuredGrid(
            size=unionSize,
            origin=unionOrigin,
            spacing=unionSpacing,
            channels=1,
            device=device
        )
        def_im = unionIm.clone()

    else:
        configList = glob.glob(Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/*')
        tmpObj = Config.Load(TempConfigSpec, configList[0])
        unionSize = tmpObj.unionSize
        unionOrigin = tmpObj.unionOrigin
        unionSpacing = tmpObj.unionSpacing
        unionIm = core.StructuredGrid(
            size=unionSize,
            origin=unionOrigin,
            spacing=unionSpacing,
            channels=1,
            device=device
        )
        def_im = unionIm.clone()

    # We need to register the last image to the VIBE
    son = sonicationList[-1]
    try:
        regObj = Config.Load(TempConfigSpec,
                             Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/{0}_config.yaml'.format(
                                 son.split('/')[-1]))
    except IOError:
        regObj = Config.SpecToConfig(TempConfigSpec)
        regObj.Niter = [40]
        regObj.gaussSigma = [0.1]
        regObj.gaussKernel = [3]
        regObj.regWeight = [2.0]
        regObj.stepSize = [0.004]

        regObj.I0 = 'None'
        regObj.I1 = 'None'
        regObj.unionSize = unionSize
        regObj.unionSpacing = unionSpacing
        regObj.unionOrigin = unionOrigin

    # Need to get the name of the volume
    outName = son.split('/')[-1]
    preLength = len(outName.split('_')[0])
    outName = outName[:preLength] + '_----' + outName[preLength:]

    regObj.outputPath = Obj.rabbitDirectory + '/elastVolumes/TemperatureRecon/' + son.split('/')[-1]

    if not os.path.exists(regObj.outputPath):
        os.makedirs(regObj.outputPath)

    plt.close('all')
    # Get the deformation field that takes the first of the sonication images to the final
    h_post = ER.MultiscaleElast(regObj, opt)
    io.SaveITKFile(h_post, regObj.outputPath + '/final_deformation_Is_to_It.mha')
    rc.WriteConfig(TempConfigSpec, regObj,
                   Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/{0}_config.yaml'.format(son.split('/')[-1]))

    print('Applying elast transformation to {0} volumes .... '.format(son.split('/')[-1]), end='')
    sys.stdout.flush()

    for image in glob.glob('{0}/rawVolumes/TemperatureRecon/'.format(Obj.rabbitDirectory) + son.split('/')[-1] + '/*'):
        # Update the name
        outName = image.split('/')[-1]
        outName = outName[:preLength + 4] + 'e' + outName[preLength + 5:]

        im = io.LoadITKFile(image, device=device)
        temp = so.ResampleWorld.Create(unionIm, device=device)(im)
        def_im = so.ApplyGrid.Create(h_post, device=device)(temp)
        io.SaveITKFile(def_im, regObj.outputPath + '/' + outName)

    # common.DebugHere()

    # # Need to resample the final image onto the new grid
    # for image in glob.glob('{0}/rawVolumes/TemperatureRecon/'.format(Obj.rabbitDirectory) + sonicationList[-1].split('/')[-1] + '/*'):

    # 	outputPath = Obj.rabbitDirectory + '/rawVolumes/TemperatureRecon/' + sonicationList[-1].split('/')[-1] + '_resample'

    # 	if not os.path.exists(outputPath):
    # 		os.makedirs(outputPath)

    # 	outName = image.split('/')[-1]
    # 	im = common.LoadITKImage(image, ca.MEM_DEVICE)

    # 	cc.ResampleWorld(unionIm, im, bg=3)
    # 	common.SaveITKImage(unionIm, outputPath + '/' + outName)

    for son in sonicationList[:-1]:

        # Create the registration object for elastic
        try:
            regObj = Config.Load(TempConfigSpec,
                                 Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/{0}_config.yaml'.format(
                                     son.split('/')[-1]))
        except IOError:
            regObj = Config.SpecToConfig(TempConfigSpec)

        # Need to get the name of the volume
        outName = son.split('/')[-1]
        preLength = len(outName.split('_')[0])
        outName = outName[:preLength] + '_----' + outName[preLength:]

        # Always set the target image to the last file - have to get the first mag image!
        regObj.I0 = son + '/../RegVols/' + outName[0:5] + '.nii.gz'

        # Do we have to get the file that has been registered to the vibe??
        regObj.I1 = sonicationList[-1] + '/../RegVols/' + lastName[0:5] + '.nii.gz'

        # Set the grid parameters of the unionFOV
        regObj.unionSize = unionSize
        regObj.unionSpacing = unionSpacing
        regObj.unionOrigin = unionOrigin

        # if sonicationList.index(son) < 8:
        # 	regObj.stepSize = [0.000]
        # else:
        # 	regObj.stepSize = [0.004]

        regObj.outputPath = Obj.rabbitDirectory + '/elastVolumes/TemperatureRecon/' + son.split('/')[-1]

        if not os.path.exists(regObj.outputPath):
            os.makedirs(regObj.outputPath)

        plt.close('all')
        # Get the deformation field that takes the first of the sonication images to the final
        h = ER.MultiscaleElast(regObj, opt)

        # Need to compose the h field to get the full deformation
        h_comp = so.ComposeGrids.Create(device=device)([h, h_post])
        io.SaveITKFile(h_comp, regObj.outputPath + '/final_deformation_Is_to_It.mha')

        # Write out the config file
        if not os.path.exists(Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs'):
            os.makedirs(Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs')

        rc.WriteConfig(TempConfigSpec, regObj,
                       Obj.rabbitDirectory + '/ConfigFiles/temperatureConfigs/{0}_config.yaml'.format(
                           son.split('/')[-1]))

        # Need to itterate through all the mag and phase images and apply the deformation
        print('Applying elast transformation to {0} volumes .... '.format(son.split('/')[-1]), end='')
        sys.stdout.flush()

        for image in glob.glob(
                '{0}/rawVolumes/TemperatureRecon/'.format(Obj.rabbitDirectory) + son.split('/')[-1] + '/*'):
            # Update the name
            outName = image.split('/')[-1]
            outName = outName[:preLength + 4] + 'e' + outName[preLength + 5:]

            im = io.LoadITKFile(image, device=device)
            temp = so.ResampleWorld.Create(unionIm, device=device)(im)
            def_im = so.ApplyGrid.Create(h_post, device=device)(temp)
            io.SaveITKFile(def_im, regObj.outputPath + '/' + outName)

#### TO DO ####
# Logic for making sure that the sonications have 4 dimensions
# Add a way to rerun only specific sonications
# Add rigid option
# Need to only run if the image is not yet created or there is a rerun
