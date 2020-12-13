def create_affine(sorted_dicoms):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = numpy.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
    image_orient2 = numpy.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]

    delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    delta_c = float(sorted_dicoms[0].PixelSpacing[1])

    image_pos = numpy.array(sorted_dicoms[0].ImagePositionPatient)

    last_image_pos = numpy.array(sorted_dicoms[-1].ImagePositionPatient)

    if len(sorted_dicoms) == 1:
        # Single slice
        step = [0, 0, -1]
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

    # check if this is actually a volume and not all slices on the same location
    if numpy.linalg.norm(step) == 0.0:
        return 'Not A Volume', '', ''

    affine = numpy.matrix([[image_orient1[0] * delta_r, image_orient2[0] * delta_c, step[0], image_pos[0]],
                           [image_orient1[1] * delta_r, image_orient2[1] * delta_c, step[1], image_pos[1]],
                           [image_orient1[2] * delta_r, image_orient2[2] * delta_c, step[2], image_pos[2]],
                           [0, 0, 0, 1]])

    spacing = numpy.array([delta_r, delta_c, numpy.sqrt(numpy.sum(step ** 2))])

    # Adjust for the patient orientation
    dcsTrans = numpy.eye(4)
    if sorted_dicoms[0].PatientPosition == 'HFS':
      dcsTrans[0,0] *= -1
      dcsTrans[1,1] *= -1
    if sorted_dicoms[0].PatientPosition == 'FFS':
      dcsTrans[1,1] *= -1
      dcsTrans[2,2] *= -1
    if sorted_dicoms[0].PatientPosition == 'FFP':
      dcsTrans[0,0] *= -1
      dcsTrans[2,2] *= -1

    affine = numpy.matmul(affine, dcsTrans)

    return affine, spacing, sorted_dicoms[0].PatientPosition