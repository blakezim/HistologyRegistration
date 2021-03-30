import os
import glob
import tkinter as tk
# import PyCA.Core as ca
import subprocess as sp
# import PyCA.Common as common
import CAMP.camp.FileIO as io
from avocado.RabbitCommon import Common as rc


def data_transfer(regObj, opt):
    rabbitNumber = regObj.rabbitDirectory.split('/')[-2]

    # Try to read the list that comes from transferring the data
    # If it isn't there, then assume we need to transfer the data
    try:
        dicomList = rc.ReadList(regObj.rabbitDirectory + '/ConfigFiles/source_volume_list.txt')

    except IOError:

        # Use subprocess to get the list of files from RAID2
        p = sp.Popen(["ssh", "bzimmerman@sebulba.med.utah.edu",
                      f"find {opt.raw_dir}*{opt.rabbit[-2:]}* -type d"],
                     stdout=sp.PIPE)
        raidDirs, _ = p.communicate()

        # There is a new line at the end which puts an empty at the end of the list
        raidDirs = raidDirs.decode('utf-8').split('\n')[:-1]
        dicomList = []

    # Get a list of the already transferrred timepoints
    transfered = sorted(glob.glob(f'{regObj.rabbitDirectory}/rawVolumes/*'))

    # Get the names of the timepoints
    transfered = [x.split('/')[-1] for x in transfered]

    # If all of the timepoints have already been transferred then break
    if all([opt.transfer_timepoints[x] in transfered for x in range(len(opt.transfer_timepoints))]):
        return

    for timepoint in opt.transfer_timepoints:
        # Use subprocess to get the list of files from RAID10
        p = sp.Popen(["ssh", "bzimmerman@sebulba.med.utah.edu",
                      f"find {opt.raw_dir}*{opt.rabbit[-2:]}* -type d"],
                     stdout=sp.PIPE)
        raidDirs, _ = p.communicate()

        # There is a new line at the end which puts an empty at the end of the list
        raidDirs = raidDirs.decode('utf-8').split('\n')[:-1]
        dicomList = []
        # Initialize an empty list
        tpList = []

        # Create a selector so the ablation volumes can be selected
        Selector = rc.PathSelector(tk, sorted(raidDirs), f'Choose {timepoint} Day Files to Convert')
        convertFiles = Selector.get_files()
        Selector.root.destroy()
        tpList += convertFiles
        dicomList += convertFiles

        for i, path in enumerate(tpList, 1):
            # Get the name of the file
            filename = path.split('s0')[-1].split(' ')[-1]

            # Correct the special characters in the path string so the scp command will work
            path = path.replace("^", "\^").replace(" ", "\ ")

            rawOutName = f'{regObj.rabbitDirectory}/rawDicoms/{timepoint}/{i:03d}_----_{filename}'
            volOutName = f'{regObj.rabbitDirectory}/rawVolumes/{timepoint}/{i:03d}_----_{filename}'

            rawOutName = rawOutName.replace('\\ ', '_').replace('\ ', '_')
            volOutName = volOutName.replace('\\ ', '_').replace('\ ', '_')

            # If the volume doesn't already exist, then copy it over
            if not os.path.exists(rawOutName):
                os.makedirs(rawOutName)
                p = sp.Popen(["scp", "-r", f"bzimmerman@sebulba.med.utah.edu:{path}/*", rawOutName])
                os.waitpid(p.pid, 0)

            if len(glob.glob(rawOutName + '/*')) == 0:
                p = sp.Popen(["scp", "-r", f"bzimmerman@sebulba.med.utah.edu:{path}/*", rawOutName])
                os.waitpid(p.pid, 0)

            if not os.path.exists(os.path.dirname(volOutName)):
                os.makedirs(os.path.dirname(volOutName))

            # Load the DICOM into a CAMP object
            # if not os.path.exists(volOutName + '.nii.gz') and not os.path.exists(volOutName + '_00.nii.gz'):
            rcsVolume = rc.LoadDICOM(rawOutName, opt.device)
            if type(rcsVolume) == list:
                for im in rcsVolume:
                    io.SaveITKFile(im, volOutName + '_{0}.nii.gz'.format(str(rcsVolume.index(im)).zfill(2)))
            else:
                io.SaveITKFile(rcsVolume, volOutName + '.nii.gz')

    # Write out the dicom list that was transferred and reconstructed
    rc.WriteList(dicomList, regObj.rabbitDirectory + '/ConfigFiles/source_volume_list.txt')

    # Check if temperature data exists and if not, transfer it
    if regObj.pipelinePhase == 'temperature':
        p = sp.Popen(["ssh", "bzimmerman@sebulba.med.utah.edu",
                      "find /v/raid2/sjohnson/Data/2018_RabbitData/{0}/TemperatureRecon/ForRegistration_Current/ -type f".format(
                          rabbitNumber)],
                     stdout=sp.PIPE)
        raidDirs, _ = p.communicate()
    else:
        raidDirs = []
    #
    # if raidDirs != []:
    #     if not os.path.exists('{0}/rawDicoms/TemperatureRecon/'.format(regObj.rabbitDirectory)):
    #         os.makedirs('{0}/rawDicoms/TemperatureRecon/'.format(regObj.rabbitDirectory))
    #
    #     if sorted(glob.glob('{0}/rawDicoms/TemperatureRecon/*'.format(regObj.rabbitDirectory))) == []:
    #         p = sp.Popen(["scp", "-r",
    #                       "bzimmerman@sebulba.med.utah.edu:/v/raid2/sjohnson/Data/2018_RabbitData/{0}/TemperatureRecon/ForRegistration_Current/*".format(
    #                           rabbitNumber),
    #                       "{0}/rawDicoms/TemperatureRecon/".format(regObj.rabbitDirectory)], stdout=sp.PIPE)
    #         sts = os.waitpid(p.pid, 0)
    #
    #         temperatureList = sorted(glob.glob('{0}/rawDicoms/TemperatureRecon/*'.format(regObj.rabbitDirectory)))
    #         rc.WriteList(temperatureList, regObj.rabbitDirectory + '/ConfigFiles/temerature_volume_list.txt')

#### TO DO ####
# Add code for if the connection closes and it creates the directory but doesn't copy files over
# Add try to create volume and if it doesn't work, just remove it from the list - write list after all is done
# Need to add something so that if a folder is added (like exvivo) it can be detected
