import os
import sys
import glob
import torch
import numpy as np
# import PyCA.Core as ca
from tkinter import Tk
# import PyCA.Common as common
import CAMP.camp.FileIO as io
import CAMP.camp.StructuredGridOperators as so
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


def RigidMatch(regObj, opt):
    # Load the points or ask where they are
    try:
        src_points = rc.load_slicer_points(regObj.rigid.sourcePointsFile)
        tar_points = rc.load_slicer_points(regObj.rigid.targetPointsFile)
    except IOError:
        regObj.rigid.sourcePointsFile = _getFilePath('Source Points File ({0} rigid)'.format(regObj.pipelinePhase),
                                                     initialdir=regObj.rabbitDirectory)
        regObj.rigid.targetPointsFile = _getFilePath('Target Points File ({0} rigid)'.format(regObj.pipelinePhase),
                                                     initialdir=regObj.rabbitDirectory)
        src_points = rc.load_slicer_points(regObj.rigid.sourcePointsFile)
        tar_points = rc.load_slicer_points(regObj.rigid.targetPointsFile)

    # Use the points to find a rigid transformation
    rigidMat = rc.SolveRigidAffine(src_points, tar_points)
    # Change to a torch tensor to use with CAMP and inverse for camp
    rigidMat = torch.tensor(rigidMat, device=opt.device, dtype=torch.float32).inverse()
    regObj.rigid.rigidMatrix = rigidMat.cpu().numpy().tolist()
    regObj.rigid.outputPath = regObj.rabbitDirectory + '/rigidVolumes/' + regObj.pipelinePhase

    # Create the output directory if it doesn't exist
    if not os.path.exists(regObj.rigid.outputPath):
        os.makedirs(regObj.rigid.outputPath)

    print('Applying rigid transformation to source volumes .... ', end='')
    sys.stdout.flush()

    # If the pipeline phase is interday, need to check for motion corrected volumes
    if regObj.pipelinePhase == 'interday':
        # Do this in a try just in case there are no motion corrected files, then it will just skip
        try:
            # Glob the directory that would contain the motion corrected volumes
            motionList = sorted(glob.glob(regObj.rabbitDirectory + '/elastVolumes/day3_motion/*.nii.gz'))
            motionList += sorted(glob.glob(regObj.rabbitDirectory + '/elastVolumes/day3_motion/*.nrrd'))
            origNames = [x.split('/')[-1][:4] + '----' + x.split('/')[-1][8:] for x in
                         motionList]  # these names have to exist in the source files
            # Create a list of paths as these original names would appear
            sorcNames = [os.path.dirname(regObj.sourceVolumes[0]) + '/' + x for x in origNames]
            regObj.sourceVolumes = sorted(list(set(regObj.sourceVolumes) - set(sorcNames)))

            # Add on the motion list to form the complete list of the source volumes
            regObj.sourceVolumes += motionList
        except IOError:
            pass

    for volume in regObj.sourceVolumes:
        # Tage the file with an 'r' to indicate rigid reg
        fileName = volume.split('/')[-1]
        fileName = fileName[:5] + 'r' + fileName[6:]
        outName = os.path.join(regObj.rigid.outputPath, fileName)

        if not os.path.exists(outName) or regObj.rigid.rerun:
            src_im = io.LoadITKFile(volume, opt.device)
            aff_im = rc.SolveAffineGrid(src_im, np.array(regObj.rigid.rigidMatrix))

            # Apply and save
            if any(x in outName for x in ['MOCO', 'ADC']):
                aff_im = so.AffineTransform.Create(affine=rigidMat,
                                                   device=opt.device, interp_mode='nearest')(src_im, aff_im)
            else:
                aff_im = so.AffineTransform.Create(affine=rigidMat,
                                                   device=opt.device)(src_im, aff_im)

            io.SaveITKFile(aff_im, outName)

    print('Done')

#### TO DO ####
