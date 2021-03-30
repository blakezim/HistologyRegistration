import os
import sys
import glob
import argparse
import avocado.data_transfer as DT
import tkinter as tk
# import AggregateData
import rigid_registration as RR
import elastic_registration as ER
import thin_plate_splines as TPS
import external_recons as EXT
import temperature_registration as TPR
# import ElastRegistration
# import ThinPlateSplines
# import TemperatureRecon
# import PyCACalebExtras.Common as cc
from avocado.RabbitCommon import Common as rc
from avocado.RabbitCommon import Config

parser = argparse.ArgumentParser(description='PyTorch Patch Based Super Deep Interpolation Example')

parser.add_argument('-d', '--raw_dir', type=str,
                    default='/v/raid10/animal_data/IACUC20-10004/',
                    help='Path to raw dicom directory')
parser.add_argument('-b', '--base_dir', type=str, default='/hdscratch2/',
                    help='Base directory for where to rabiit folder to contain the data.')
parser.add_argument('-p', '--phase', type=str, default='temperature',
                    help='Phase of registration to run.')
parser.add_argument('-r', '--rabbit', type=str, default='20_091',
                    help='Rabbit number to process.')
parser.add_argument('-t', '--transfer_timepoints', type=str, default=['AblationImaging', 'PostImaging'],
                    help='Which imaging timepoints to transfer', nargs='*')
parser.add_argument('-dev', '--device', type=str, default='cuda:1',
                    help='Torch device identifier for variable location')
parser.add_argument('--transfer_only', action='store_true', help='Only transfer files?')
opt = parser.parse_args()


def _valspec(_):
    """Hook to do extra validation"""
    # This is currently unused, but necessary
    pass


# Set up secifications for the different phases of registration. Temperature gets it's own specification
phaseSpec = {
    'rabbitDirectory':
        Config.Param(comment='Top directory for the rabbit.'),
    'pipelinePhase':
        Config.Param(comment='What phase of registration the config is responsible for (Interday, Intraday, Exvivo.'),
    'sourceVolumes':
        Config.Param(comment='A list of the source volumes to be transformed for this config file.'),
    'useRigid':
        Config.Param(default=False, comment='Binary to use rigid registration.'),
    'useTPS':
        Config.Param(default=False, comment='Binary to use Thin Plate Splines.'),
    'useElast':
        Config.Param(default=False, comment='Binary to use Elastic registration.'),
    'aggData':
        Config.Param(default=False,
                     comment='Binary if all data has been processed and needs to be placed in the same folder'),
    'rigid': RR.RigidConfigSpec,
    'tps': TPS.TPSConfigSpec,
    'elast': ER.ElastConfigSpec,
    '_validation_hook': _valspec
}

phaseSpecTemp = {
    'rabbitDirectory':
        Config.Param(comment='Top directory for the rabbit.'),
    'pipelinePhase':
        Config.Param(comment='What phase of registration the config is responsible for (Interday, Intraday, Exvivo.'),
    'dataDirectory':
        Config.Param(default='None',
                     comment='Raid 2 Directory where the orignal data is held.'),
    'aggData':
        Config.Param(default=False,
                     comment='Binary if all data has been processed and needs to be placed in the same folder'),
    '_validation_hook': _valspec
}


def main(opt):
    # Ask for the rabbit number and the registration phase
    topDir = f'{opt.base_dir}/{opt.rabbit}/Longitudinal/'
    if not os.path.exists(topDir):
        os.makedirs(topDir)
    phase = opt.phase

    # Make sure the phase is one of the known ones
    if phase not in ['interday', 'day0_motion', 'day3_motion', 'exvivo', 'temperature']:
        print('Phase not recognized, so you may NOT proceed.')
        sys.exit()

    # Load the configuration file depending on the phase
    try:
        if phase == 'temperature':
            phaseObj = Config.Load(phaseSpecTemp, topDir + '/ConfigFiles/{0}_config.yaml'.format(phase))
        else:
            phaseObj = Config.Load(phaseSpec, topDir + '/ConfigFiles/{0}_config.yaml'.format(phase))

        # Need to know if this is the initial run or not so we can prompt the user to edit the config file.
        # If the file loads, then it is not the initial run
        init = False

    except IOError:
        # If we can't load config specs, then it is assumed that this is the inital run
        init = True
        if not os.path.exists(topDir + 'ConfigFiles/'):
            os.makedirs(topDir + 'ConfigFiles/')
        if phase == 'temperature':
            phaseObj = Config.SpecToConfig(phaseSpecTemp)
        else:
            phaseObj = Config.SpecToConfig(phaseSpec)

        # Set some object parameters
        phaseObj.rabbitDirectory = topDir
        phaseObj.pipelinePhase = phase

    # Transfer the data from Raid2: Always check
    DT.data_transfer(phaseObj, opt)

    # Generate volumes from the STIR: Always check
    EXT.ExternalRecon(phaseObj, opt)

    if opt.transfer_only:
        return

    # Data Selection - Seclect the files to be used/registered from the ones transferred
    # If the phase is interday
    if phaseObj.pipelinePhase == 'interday' and phaseObj.sourceVolumes is None:
        SelectVolumes = rc.PathSelector(parent=tk,
                                        paths=sorted(glob.glob(phaseObj.rabbitDirectory + '/rawVolumes/Post*/*')),
                                        label='Interday Source Volumes')
        phaseObj.sourceVolumes = SelectVolumes.get_files()
        SelectVolumes.root.destroy()

    # If the phase is correcting day0 motion
    if phaseObj.pipelinePhase == 'day0_motion' and phaseObj.sourceVolumes is None:
        SelectVolumes = rc.PathSelector(parent=tk,
                                        paths=sorted(glob.glob(phaseObj.rabbitDirectory + '/rawVolumes/Ablation*/*')),
                                        label='Day 0 Pre-motion Volumes')
        phaseObj.sourceVolumes = SelectVolumes.get_files()
        SelectVolumes.root.destroy()

    # If the phase is correcting day3 motion
    if phaseObj.pipelinePhase == 'day3_motion' and phaseObj.sourceVolumes == None:
        SelectVolumes = rc.PathSelector(parent=tk,
                                        paths=sorted(glob.glob(phaseObj.rabbitDirectory + '/rawVolumes/Post*/*')),
                                        label='Day 3 Pre-motion Volumes')
        phaseObj.sourceVolumes = SelectVolumes.get_files()
        SelectVolumes.root.destroy()

    # If the phase is exvivo
    if phaseObj.pipelinePhase == 'exvivo' and phaseObj.sourceVolumes == None:
        SelectVolumes = rc.PathSelector(parent=tk,
                                        paths=sorted(glob.glob(phaseObj.rabbitDirectory + '/rawVolumes/ExVivo*/*')),
                                        label='Exvivo Source Volumes')
        phaseObj.sourceVolumes = SelectVolumes.get_files()
        SelectVolumes.root.destroy()

    # Write out the specifications
    if phaseObj.pipelinePhase != 'temperature':
        rc.WriteConfig(phaseSpec, phaseObj,
                       phaseObj.rabbitDirectory + '/ConfigFiles/{0}_config.yaml'.format(phaseObj.pipelinePhase))
    else:
        rc.WriteConfig(phaseSpecTemp, phaseObj,
                       phaseObj.rabbitDirectory + '/ConfigFiles/{0}_config.yaml'.format(phaseObj.pipelinePhase))

    # Perform the registration
    # If the phase is not temperature
    if phaseObj.pipelinePhase != 'temperature':
        # Preform rigid registration of the volumes
        if phaseObj.useRigid:
            RR.RigidMatch(phaseObj, opt)
            rc.WriteConfig(phaseSpec, phaseObj,
                           phaseObj.rabbitDirectory + f'/ConfigFiles/{phaseObj.pipelinePhase}_config.yaml')

        # Use TPS if required
        if phaseObj.useTPS:
            TPS.TPS(phaseObj, opt)
            rc.WriteConfig(phaseSpec, phaseObj,
                           phaseObj.rabbitDirectory + f'/ConfigFiles/{phaseObj.pipelinePhase}_config.yaml')

        # Preform elastic registration of the volumes if necessary
        if phaseObj.useElast:
            ER.main(phaseObj, opt)
            rc.WriteConfig(phaseSpec, phaseObj,
                           phaseObj.rabbitDirectory + f'/ConfigFiles/{phaseObj.pipelinePhase}_config.yaml')
    else:
        TPR.tempRecon(phaseObj, opt)
        rc.WriteConfig(phaseSpecTemp, phaseObj,
                       phaseObj.rabbitDirectory + '/ConfigFiles/{0}_config.yaml'.format(phaseObj.pipelinePhase))

    if phaseObj.aggData:
        # Need to figure out when to run the aggreagate data - Is it fine to run every time?
        AggregateData.aggregateData(phaseObj)

    if init:
        print('Please edit the Config File for registration. Config Location:',
              '/n', phaseObj.rabbitDirectory + '/ConfigFiles/{0}_config.yaml'.format(phaseObj.pipelinePhase))


if __name__ == '__main__':
    main(opt)
