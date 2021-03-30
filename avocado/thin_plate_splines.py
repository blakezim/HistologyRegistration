import os
import sys
import glob
import torch
import numpy as np
# import PyCA.Core as ca
from tkinter import Tk
# import PyCA.Common as common
import CAMP.camp.FileIO as io
import CAMP.camp.Core as core
import CAMP.camp.StructuredGridOperators as so
from avocado.RabbitCommon import Common as rc
from avocado.RabbitCommon import Config
# import PyCACalebExtras.Common as cc
from tkinter.filedialog import askdirectory, askopenfilename
# import PyCAApps as apps


def _valspec(_):
    """Hook to do extra validation"""
    pass


def _getFilePath(name, initialdir):
    Tk().withdraw()
    return askopenfilename(title=name, initialdir=initialdir)


def _getPath(name, initialdir):
    Tk().withdraw()
    return askdirectory(title=name, initialdir=initialdir)


TPSConfigSpec = {
    'sourcePointsDir':
        Config.Param(default=None,
                     comment='Text file that contains the TPS registration landmark points for the source volume.'),
    'targetPointsDir':
        Config.Param(default=None,
                     comment='Text file that contains the TPS registration landmark points for the target volume.'),
    'landmarks':
        Config.Param(default=None,
                     comment='Landmarks that are used for TPS registration.'),
    'source_image':
        Config.Param(default='None',
                     comment='Source Image for the registration.'),
    'target_image':
        Config.Param(default='None',
                     comment='Target Image for the registration.'),
    'outputPath':
        Config.Param(default=None,
                     comment='Path to the TPS registered volumes. '),
    'incompressible':
        Config.Param(default=True,
                     comment='Volume Preserving registration.'),
    'rerun':
        Config.Param(default=False,
                     comment='Allows the user to rerun the tps registration if the points are updated.'),
    '_validation_hook': _valspec
}


def TPS(regObj, opt):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()
    # Load the points or ask where they are

    if not os.path.exists(f'{regObj.tps.outputPath}/tps_deformation.nii.gz') or regObj.tps.rerun:

        regObj.tps.outputPath = regObj.rabbitDirectory + '/tpsVolumes/' + regObj.pipelinePhase
        if not os.path.exists(regObj.tps.outputPath):
            os.makedirs(regObj.tps.outputPath)

        try:
            src_points = rc.load_slicer_points(regObj.tps.sourcePointsDir)
            tar_points = rc.load_slicer_points(regObj.tps.targetPointsDir)
        except IOError:
            regObj.tps.sourcePointsDir = _getFilePath('Source Points File ({0} TPS)'.format(regObj.pipelinePhase),
                                                         initialdir=regObj.rabbitDirectory)
            regObj.tps.targetPointsDir = _getFilePath('Target Points File ({0} TPS)'.format(regObj.pipelinePhase),
                                                         initialdir=regObj.rabbitDirectory)
            src_points = rc.load_slicer_points(regObj.tps.sourcePointsDir)
            tar_points = rc.load_slicer_points(regObj.tps.targetPointsDir)

        try:
            source_image = io.LoadITKFile(regObj.tps.source_image, opt.device)
            target_image = io.LoadITKFile(regObj.tps.target_image, opt.device)
        except RuntimeError:
            regObj.tps.source_image = _getFilePath('Source Image File (TPS)',
                                                   initialdir=regObj.rabbitDirectory)
            regObj.tps.target_image = _getFilePath('Target Image File (TPS)',
                                                   initialdir=regObj.rabbitDirectory)
            source_image = io.LoadITKFile(regObj.tps.source_image, opt.device)
            target_image = io.LoadITKFile(regObj.tps.target_image, opt.device)

        rbf_filter = so.RadialBasis.Create(
            tar_points,
            src_points,
            sigma=1.0,
            device=opt.device
        )

        # Solve for the rigid transform for determining the grid to sample onto
        rigidMat = rc.SolveRigidAffine(np.array(src_points), np.array(tar_points))

        # Solve for the radial basis funtion deformation
        if regObj.tps.incompressible:
            rbf_source, rbf_grid = rbf_filter.filter_incompressible(in_grid=source_image, out_grid=target_image,
                                                                    t_step=10000, conv=0.8, step=0.03)
        else:
            rbf_source, rbf_grid = rbf_filter(in_grid=source_image, out_grid=target_image)

        io.SaveITKFile(rbf_grid, f'{regObj.tps.outputPath}/tps_deformation.nii.gz')

    else:
        rbf_grid = io.LoadITKFile(f'{regObj.tps.outputPath}/tps_deformation.nii.gz', device=opt.device)

    # Check if there are rigid volumes, if not, get them from the source
    volumes = sorted(glob.glob(regObj.rigid.outputPath + '/*'))
    if volumes == []:
        volumes = regObj.sourceVolumes

    print('Applying TPS transformation to source volumes .... ', end='')
    sys.stdout.flush()

    for volume in volumes:
        # Tag the file with 't' to indicate it had thin plate splines applied
        fileName = volume.split('/')[-1]
        fileName = fileName[:6] + 't' + fileName[7:]
        outName = os.path.join(regObj.tps.outputPath, fileName)

        if not os.path.exists(outName) or regObj.tps.rerun:
            # Load the source image
            src_im = io.LoadITKFile(volume, device=opt.device)
            aff_im = rc.SolveAffineGrid(src_im, np.array(rigidMat))

            # Determine the subvol onto which the image needs to be resampled
            try:
                h_resample = so.ResampleWorld.Create(aff_im, device=opt.device)(rbf_grid)
            except RuntimeError:
                h_resample = so.ResampleWorld.Create(target_image, device=opt.device)(rbf_grid)

            # Apply and save
            if any(x in outName for x in ['MOCO', 'ADC', 't1_map']):
                def_im = so.ApplyGrid.Create(h_resample, device=opt.device, interp_mode='nearest')(src_im, h_resample)
            else:
                def_im = so.ApplyGrid.Create(h_resample, device=opt.device)(src_im, h_resample)
            io.SaveITKFile(def_im, outName)

    print('Done')


def oldTPS(regObj, opt):
    # Get a list of all the files in the directory
    srcPointList = sorted(glob.glob(regObj.tps.sourcePointsDir + '/*'))
    tarPointList = sorted(glob.glob(regObj.tps.targetPointsDir + '/*'))

    if srcPointList == [] and tarPointList == []:
        regObj.tps.sourcePointsDir = _getPath('Source Points File ({0} TPS)'.format(regObj.pipelinePhase),
                                              initialdir=regObj.rabbitDirectory)
        regObj.tps.targetPointsDir = _getPath('Target Points File ({0} TPS)'.format(regObj.pipelinePhase),
                                              initialdir=regObj.rabbitDirectory)
        srcPointList = sorted(glob.glob(regObj.tps.sourcePointsDir + '/*'))
        tarPointList = sorted(glob.glob(regObj.tps.targetPointsDir + '/*'))

    src_points = []
    tar_points = []

    # Iterate through the files in the directory and stack them
    for num in range(len(srcPointList)):
        srcPts = np.loadtxt(srcPointList[num], delimiter=' ')
        tarPts = np.loadtxt(tarPointList[num], delimiter=' ')

        src_points += srcPts.tolist()
        tar_points += tarPts.tolist()

    # Turn the points back to an array
    src_points = np.array(src_points)
    tar_points = np.array(tar_points)

    # Convert the points into a format that apps.SolveSpline can use
    landmarks = []
    for pt in range(0, len(tar_points)):
        landmarks.append([tar_points[pt].tolist(), src_points[pt].tolist()])

    regObj.tps.landmarks = landmarks
    spline = apps.SolveSpline(regObj.tps.landmarks)

    regObj.tps.outputPath = regObj.rabbitDirectory + '/tpsVolumes/' + regObj.pipelinePhase
    if not os.path.exists(regObj.tps.outputPath):
        os.makedirs(regObj.tps.outputPath)

    # Check if there are rigid volumes, if not, get them from the source
    volumes = sorted(glob.glob(regObj.rigid.outputPath + '/*'))
    if volumes == []:
        volumes = regObj.sourceVolumes

    print('Applying TPS transformation to source volumes .... ', end='')
    sys.stdout.flush()

    for volume in volumes:
        # Tag the file with 't' to indicate it had thin plate splines applied
        fileName = volume.split('/')[-1]
        fileName = fileName[:6] + 't' + fileName[7:]
        outName = os.path.join(regObj.tps.outputPath, fileName)

        if not os.path.exists(outName) or regObj.tps.rerun:
            src_im = common.LoadITKImage(volume, ca.MEM_DEVICE)
            tps_im = src_im.copy()
            h = apps.SplineToHField(spline, tps_im.grid())

            if any(x in outName for x in ['MOCO', 'ADC']):
                cc.ApplyHReal(tps_im, src_im, h, interp=ca.INTERP_NN)
            else:
                cc.ApplyHReal(tps_im, src_im, h)

            common.SaveITKImage(tps_im, outName)
            del src_im, tps_im, h

    print('Done')

#### TO DO ####
# Can't choose single point for TPS - make it so you can
