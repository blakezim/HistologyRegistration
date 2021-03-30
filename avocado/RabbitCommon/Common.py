import yaml
import csv
import json
import torch
from tkinter import *
import numpy as np
import CAMP.camp.Core as core
import CAMP.camp.StructuredGridOperators as so
import avocado.RabbitCommon.commonD2N as dc
# import PyCA.Core as ca
# import PyCA.Common as common
# import PyCACalebExtras.Common as cc
import scipy.ndimage.filters
import skimage.restoration as res
from avocado.RabbitCommon import Config


def SolveRigidAffine(src_points, tar_points, grid=None):
    '''Takes correspondance points of a source volume and a target volume
       and returns the rigid only affine matrix formatted for use with PyCA
       apply affine functions.

       src_points = points chosen in the source (moving) volume
       tar_points = points chosen in the target (fixed) volume
       type = both sets of points are assumed to be N x Dim column numpy array

       This code is an implementation of the following technical notes:
       Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." Computing 1 (2017): 1.'''

    if 'torch' in str(src_points.dtype):
        src_points = src_points.cpu().numpy()
        tar_points = tar_points.cpu().numpy()

    if grid is not None:
        if np.shape(src_points)[1] == 2:
            src_points = np.array(src_points) * grid.spacing().tolist()[0:2] + grid.origin().tolist()[0:2]
            tar_points = np.array(tar_points) * grid.spacing().tolist()[0:2] + grid.origin().tolist()[0:2]

        else:
            src_points = np.array(src_points) * grid.spacing().tolist() + grid.origin().tolist()
            tar_points = np.array(tar_points) * grid.spacing().tolist() + grid.origin().tolist()

    # Calculate the mean of each set of points
    src_mean = np.mean(src_points, 0)
    tar_mean = np.mean(tar_points, 0)

    # Subtract the mean so the points are centered at [0, 0, 0]
    src_zero = src_points - src_mean
    tar_zero = tar_points - tar_mean

    # Calculate the covariance matrix
    S = np.matmul(np.transpose(src_zero), tar_zero)

    # SVD of the covariance matrix
    [U, _, V] = np.linalg.svd(S)

    # Create the weights matrix and incorporate the determinant of the rotation matrix
    if np.shape(src_points)[1] == 2:
        W = np.eye(2)
        W[1, 1] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    else:
        W = np.eye(3)
        W[2, 2] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    # Caluclate the rotation matrix
    R = np.matmul(np.transpose(V), np.matmul(W, np.transpose(U)))

    # Calculate the translation from the rotated points
    rot_src_points = np.matmul(R, np.transpose(src_points))
    translation = tar_mean - np.mean(rot_src_points, 1)

    # Construct the affine matrix for use in PyCA
    if np.shape(src_points)[1] == 2:
        affine = np.zeros([3, 3])
        affine[0:2, 0:2] = R
        affine[0:2, 2] = translation
        affine[2, 2] = 1
    else:
        affine = np.zeros([4, 4])
        affine[0:3, 0:3] = R
        affine[0:3, 3] = translation
        affine[3, 3] = 1

    return affine


def SolveAffineGrid(source_image, input_affine):
    '''Takes a source volume and an affine matrix and solves for the necessary
       grid in the target space of the affine. Essentially calculates the bounding
       box of the source image after apply the affine transformation

       source_image = volume to be transformed by the affine (Image3D type)
       input_affine = the affine transformation (4x4)

       Returns the grid for the transformed source volume in real coordinates'''

    # Make sure we don't mess with the incoming affine
    affine = np.copy(input_affine)

    # Get parameters from the incoming source image
    in_sz = source_image.size.tolist()
    in_sp = source_image.spacing.tolist()
    in_or = source_image.origin.tolist()

    # Extract the pure rotation and ignore scaling to find the final size of the volume
    U, s, V = np.linalg.svd(affine[0:3, 0:3])
    rotMat = np.eye(4)
    rotMat[0:3, 0:3] = np.dot(U, V)

    # Get the corners of the volume in index coordinates
    inputCorners = np.array([0, 0, 0, 1])
    inputCorners = np.vstack((inputCorners, np.array([in_sz[0], 0, 0, 1])))
    inputCorners = np.vstack((inputCorners, np.array([0, in_sz[1], 0, 1])))
    inputCorners = np.vstack((inputCorners, np.array([in_sz[0], in_sz[1], 0, 1])))

    # Account for the case that the source image is 3D
    if len(source_image.size) == 3:
        inputCorners = np.vstack((inputCorners, np.array([0, 0, in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([in_sz[0], 0, in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([0, in_sz[1], in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([in_sz[0], in_sz[1], in_sz[2], 1])))

    # Define the index corners to find the final size of the transformed volume
    indexCorners = np.matrix(inputCorners)

    # Find the real corners of the input volume for finding the output origin
    realCorners = np.matrix(np.multiply(inputCorners, np.array(in_sp + [1])) + np.array(in_or + [0]))

    # Apply the transformations to the real and index corners
    # Need to subtract the mean and apply around the center
    # Translation is not going to effect the index size, but will effect the origin of the block for real coordinates
    outRealCorners = np.matmul(affine, realCorners.transpose())
    outIndxCorners = np.matmul(rotMat, indexCorners.transpose())

    # Find the size in real and index coordinates of the output volume
    realSize = (np.max(outRealCorners, 1) - np.min(outRealCorners, 1))[0:3]
    indexSize = (np.max(outIndxCorners, 1) - np.min(outIndxCorners, 1))[0:3]

    # The index size is the size of the output volume
    out_sz = np.squeeze(np.array(np.ceil(indexSize).astype(int)))
    out_sz[out_sz == 0] = 1

    # We can divide index size into real size to get the spacing of the real volume; Need to account for 2D zero
    out_sp = np.squeeze(np.array(np.divide(realSize, indexSize, where=indexSize != 0)))
    out_sp[out_sp == 0] = 1

    # Find the output origin by taking the min in each dimension of the real transformed corners
    out_or = np.squeeze(np.array(np.min(outRealCorners, 1)))[0:3]

    # Make the output structured grid
    out_tensor = core.StructuredGrid(
        size=out_sz.copy(),
        origin=out_or.copy(),
        spacing=out_sp.copy(),
        device=source_image.device,
        channels=1
    )
    return out_tensor


def LoadDICOM(dicomDirectory, device):
    '''Takes a directory that contains DICOM files and returns a PyCA Image3D

    dicomDirectory = Full path to the folder with the dicoms

    Returns an Image3D in the Reference Coordiante System (RCS)
    '''

    # Read the DICOM files in the directory

    if type(dicomDirectory) == list:
        dicoms = dicomDirectory
    else:
        dicoms = dc.read_dicom_directory(dicomDirectory)

    # Sort the loaded dicoms
    sort_dcms = dc.sort_dicoms(dicoms)

    acqList = [sort_dcms[x].AcquisitionNumber for x in range(len(sort_dcms))]

    # Need to account for if there are multiple acquisitions in a single folder
    if np.min(acqList) != np.max(acqList):
        volList = []
        for acq in range(np.max(acqList)):
            dcm_list = [x for x in sort_dcms if x.AcquisitionNumber == (acq + 1)]
            volList += [LoadDICOM(dcm_list, device)]
        return volList

    # Extract the actual volume of pixels
    pixel_vol = dc.get_volume_pixeldata(sort_dcms)

    # Generate the affine from the dicom headers (THIS CODE WAS MODIFIED FROM dicom2nifti)
    affine, spacing, pp = dc.create_affine(sort_dcms)

    if type(affine) == str:
        return 'Not A Volume'

    # Convert the dicom volume to an Image3D - numpy is X, Y, Z and CAMP is Z, Y, X
    rawDicom = core.StructuredGrid(
        size=pixel_vol.shape,
        origin=[0.0, 0.0, 0.0],
        tensor=torch.tensor(pixel_vol.astype(np.float)).unsqueeze(0),
        device=device,
        channels=1
    )

    rcs_grid = SolveAffineGrid(rawDicom, affine)

    wcsTrans = np.eye(4)
    if pp == 'HFS':
        wcsTrans[0, 0] *= -1
        wcsTrans[1, 1] *= -1
    if pp == 'FFS':
        wcsTrans[1, 1] *= -1
        wcsTrans[2, 2] *= -1
    if pp == 'FFP':
        wcsTrans[0, 0] *= -1
        wcsTrans[2, 2] *= -1

    world_grid = SolveAffineGrid(rcs_grid, wcsTrans)

    rcs_grid = so.AffineTransform.Create(affine=torch.tensor(affine, dtype=torch.float, device=device))(
        rawDicom, rcs_grid, xyz_affine=False)
    world_grid = so.AffineTransform.Create(affine=torch.tensor(wcsTrans, dtype=torch.float, device=device))(
        rcs_grid, world_grid, xyz_affine=False)

    return world_grid


class PathSelector:

    def __init__(self, parent, paths, label):
        self.root = parent.Tk()
        self.root.title(label)
        self.root.geometry("1500x500")
        self.listbox = parent.Listbox(self.root, selectmode='multiple')
        self.listbox.pack(fill='both', expand=YES)

        for i, p in enumerate(paths):
            self.listbox.insert(END, p)
            self.listbox.itemconfig(i, bg="yellow" if i % 2 == 0 else "cyan")

        self.cvtBtn = parent.Button(self.root, text='Convert Files')
        self.cvtBtn.pack()
        self.cvtBtn.bind("<Button-1>", self.convert)

        parent.Button(self.root, text='Done', command=self.root.quit).pack()
        self.files = []
        self.root.mainloop()

    def convert(self, event):
        selections = self.listbox.curselection()
        for selection in selections:
            self.files.append(self.listbox.get(selection))

    def get_files(self):
        return self.files


def ReadList(filename):
    f = open(filename, 'r')
    readList = []
    for line in f:
        readList.append(line.strip())
    f.close()
    return readList


def WriteList(output_list, filename):
    f = open(filename, 'w')
    for i in output_list:
        f.write(str(i) + "\n")
    f.close()


def load_slicer_points(file):
    with open(file, 'r') as f:
        data = json.load(f)
        coord = []
        for i, fd in enumerate(data['markups'][0]['controlPoints']):
            pos = fd['position']
            orient = fd['orientation']
            # cord = np.dot(np.array(orient).reshape(3, 3), np.array(pos))
            coord.append(np.array(pos)[::-1].copy())

    np_coords = np.stack(coord, 0)
    return torch.as_tensor(np_coords.copy())


# def write_fcsv(data, file):
#     # Assume that we need to flip the data and what not
#     data = np.array(data.tolist())[:, ::-1]
#
#     # Now need to flip the dimensions to be consistent with slicer
#     data[:, 0] *= -1
#     data[:, 1] *= -1
#
#     with open(file, 'w') as f:
#         w = csv.writer(f, delimiter=',')
#         w.writerow(['# Markups fiducial file version = 4.10'])
#         w.writerow(['# CoordinateSystem = 0'])
#         w.writerow(['# columns = id', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel', 'lock', 'label', 'desc',
#                     'associatedNodeID'])
#
#         for i, point in enumerate(data):
#             prefix = [f'vtkMRMLMarkupsFiducialNode_{i}']
#             post = ['0.000', '0.000', '0.000', '1.000', '1', '1', '0', f'F-{i + 1}', '', 'vtkMRMLScalarVolumeNode1']
#             point = [str(x) for x in point]
#             w.writerow(prefix + point + post)


def WriteConfig(spec, d, filename):
    """Write a Config Object to a YAML file"""
    with open(filename, 'w') as f:
        f.write(Config.ConfigToYAML(spec, d))
    f.close()


def MedianFilter_I(Im, kernel_size):
    """ Median filter for 3D Images """

    # Convert to a numpy array
    Im_np = Im.data.cpu().numpy().squeeze(0)
    Im_np = scipy.ndimage.filters.median_filter(Im_np, kernel_size)

    Im_data = torch.tensor(Im_np, device=Im.device, dtype=Im.dtype)
    Im.data = Im_data.unsqueeze(0).clone()
    return Im


def TVFilter_I(Im, tv_weight):
    """ In-place total variation filter for Image3D's """

    Im_np = Im.data.cpu().numpy().squeeze(0)
    Im_np = res.denoise_tv_chambolle(Im_np, weight=tv_weight)

    Im_data = torch.tensor(Im_np, device=Im.device, dtype=Im.dtype)
    Im.data = Im_data.unsqueeze(0).clone()
    return Im


def resampleGrid(Im, h):
    """Determines the grid for Im to be resampled to assuming h is a subvol of Im (or close to it).
    The returned grid is in real coordinates so cc.ResampleWorld can be used. """
    # Extract the grid information for ease of use

    hSz = np.array(h.size().tolist())
    hSp = np.array(h.spacing().tolist())
    hOr = np.array(h.origin().tolist())

    iSz = np.array(Im.size().tolist())
    iSp = np.array(Im.spacing().tolist())
    iOr = np.array(Im.origin().tolist())

    # Need to find the size of the output image
    oSz = np.ceil((hSz * hSp) / iSp).astype('int').tolist()
    oSp = iSp.tolist()
    oOr = hOr.tolist()

    return cc.MakeGrid(oSz, oSp, oOr)

# def FieldResampleWorld(f_out, f_in, bg=0):

# 	# abs wont work
# 	# Just assume it doesn 2 pixels and dialte the mask you get for == 0; have to do ==0 otherwise the mid-point will likely get selected
# 	# actually just errode by a bit to get rid of the center

# 	i_out = ca.Image3D(f_out.grid(), f_out.memType())
# 	i_in = ca.Image3D(f_in.grid(), f_in.memType())
# 	resamp = ca.Image3D(f_out.grid(), f_out.memType())

# 	inside = i_out.copy()
# 	otside = i_out.copy()

# 	eye = f_out.copy()
# 	ca.SetToIdentity(eye)
# 	cc.HtoReal(eye)

# 	for i in xrange(3):
# 		ca.Copy(i_in, f_in, i)
# 		ca.ResampleWorld(resamp, i_in, bg=3)


# def FieldResampleWorld(f_out, f_in, bg=0):
# 	"""Resamples fields with partial ID background strategy.
# 	   Assumes the input field is in real coordinates.

# 	   Returns: Resampled H Field in real coordinates."""

# 	# # Set the f_out to identity
# 	# ca.SetToIdentity(f_out)
# 	# common.DebugHere()
# 	# Create temp variable so that the orignal field is not modified
# 	temp = f_in.copy()
# 	# Change to index coordinates
# 	cc.HtoIndex(temp)

# 	# Convert the h field to a vector field
# 	# ca.ResampleH(f_out, f_in)
# 	ca.HtoV_I(temp)
# 	# Resample the field onto f_out grid
# 	ca.ResampleV(f_out, temp, bg=bg)
# 	# Convert back to h field
# 	ca.VtoH_I(f_out)
# 	# Convert the resampled field back to real coordinates
# 	cc.HtoReal(f_out)
# 	del temp
