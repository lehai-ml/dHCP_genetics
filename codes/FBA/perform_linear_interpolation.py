import argparse
import nibabel as nb
import os
import os.path as op
import shutil
import tempfile
import subprocess
from tqdm import tqdm
import numpy as np
from collections import Counter
import re
###Here is the work flow####
### First the average FODs have been registered to the common FOD template to generate non linear warps.
### Then binary mask of each tract from the WM parcellation is generated, and each tract is transformed
### to the average group sapce using the yielded non linear warps.
### The transformed binary masks are then concatenate together, and in voxels, that are overlapping
### in more than tract - the voxel is assigned to the tract that has the highest interpollated value.

def find_program(program:str):
    def is_exe(fpath):
        return op.exists(fpath) and os.access(fpath, os.X_OK)
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = op.join(path, program)
        if is_exe(exe_file):
            return program
    return None


def mif_to_nifti2(mif_file):
    if not mif_file.endswith(".nii"):
        dirpath = tempfile.mkdtemp()
        mrconvert = find_program(program='mrconvert')
        if mrconvert is None:
            raise Exception("The mrconvert executable could not be found")
        nii_file = op.join(dirpath,'mif.nii')
        proc = subprocess.Popen([mrconvert, mif_file, nii_file],stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        _, err = proc.communicate()
    else:
        nii_file = mif_file
        dirpath = None
    if not op.exists(nii_file):
        raise Exception(err)
    nifti_img = nb.load(nii_file)
    data = nifti_img.get_fdata()
    if dirpath:
        shutil.rmtree(dirpath)
    return nifti_img,data

### generate binary mask
def generate_binary_mask(WM_parcellation:str):
    if isinstance(WM_parcellation,str):
        ### first identify the available labels ####
        _,parcellation = mif_to_nifti2(WM_parcellation)
        unique_labels = np.unique(parcellation[parcellation > 0])
        unique_labels = [str(int(i)) for i in unique_labels]
        assert all(count==1 for count in Counter(unique_labels).values()), 'Some of the label may not be an integer, i.e., 1.5. In this case 1.5 is labeled as 1. If that is not a problem, then you can comment out this assertion line'

        ### generate individual binary mask
        mrcalc = find_program(program='mrcalc')
        if mrcalc is None:
            raise Exception("The mrcaclc executable could not be found")
        dirpath = tempfile.mkdtemp()
        print('Generating binary mask')
        for label in tqdm(unique_labels):
            temp_mask_file = op.join(dirpath,f'WM_{label}.mif')
            proc = subprocess.Popen([mrcalc,WM_parcellation,f'{label}','-eq',temp_mask_file],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            _, err = proc.communicate()
            if not op.exists(temp_mask_file):
                raise Exception(err)
        print(f'the temporary directory is : {dirpath}')
        return dirpath


def transform_binary_mask(dirpath,non_linear_warp):
    if isinstance(dirpath,str):
        all_WM_labels = os.listdir(dirpath)
        print('Transforming binary mask of each tract to the group average space')
        for label in tqdm(all_WM_labels):
            if re.match(r'WM_\d+.mif',label):
                label_number = re.findall(r'\d+',label)[0]
                temp_mask_file = op.join(dirpath,label)
                temp_transformed_file = op.join(dirpath,f'WM_{label_number}_transformed.nii')
                mrtransform = find_program('mrtransform')
                if mrtransform is None:
                    raise Exception("The mrtransform executable could not be found")
                proc = subprocess.Popen([mrtransform,temp_mask_file,'-warp',non_linear_warp,'-interp','linear',temp_transformed_file],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                _, err = proc.communicate()
                if not op.exists(temp_transformed_file):
                    raise Exception(err)
        return dirpath

def generate_parcellation_in_group_space(group_image,
                                         dirpath,
                                         output_file):
    nb_group_image,nb_group_image_data = mif_to_nifti2(group_image)
    if nb_group_image_data.ndim == 4:
        nb_group_image_data = nb_group_image_data[:,:,:,0]
    assert nb_group_image_data.ndim == 3
    parcellation_in_group_space = np.zeros(nb_group_image_data.shape)

    if isinstance(dirpath,str):
        label_number_list = []
        all_WM_labels = os.listdir(dirpath)
        for transformed_label in all_WM_labels:
            if re.match(r'WM_\d+_transformed.nii',transformed_label):
                label_number_list.append(re.findall(r'\d+',transformed_label)[0])
    label_number_list = [int(i) for i in label_number_list]
    lowest_label_number = min(label_number_list)
    
    ###Remember you need to sort the labels because when you stack them, the order matter.
    stacked_interpolated_labels = []
    label_number_list = sorted(label_number_list)
    for label_number in label_number_list:
        transformed_label = f'WM_{label_number}_transformed.nii'
        stacked_interpolated_labels.append(nb.load(op.join(dirpath,transformed_label)).get_fdata())
        
    stacked_interpolated_labels = np.stack(stacked_interpolated_labels,axis=-1)
    assert stacked_interpolated_labels[:,:,:,0].shape == nb_group_image_data.shape, 'The group average has different shape to the interpolated labels, check the arguments. The interpolated labels are transformed labels to the group average space'
    non_background_voxels_coords = np.column_stack(np.nonzero(nb_group_image_data))
    
    print('Assigning overlapped voxels to the highest interpolated tract')
    for coords in tqdm(non_background_voxels_coords):
        i,j,k = coords
        if np.max(stacked_interpolated_labels[i,j,k,:]) == 0:
            parcellation_in_group_space[i,j,k] = 0
            continue
        else:
            highest_label = np.argmax(stacked_interpolated_labels[i,j,k,:])
            parcellation_in_group_space[i,j,k] = highest_label+lowest_label_number
                
    output_parcellation = nb.Nifti1Image(parcellation_in_group_space,
              affine=nb_group_image.affine,
              header=nb_group_image.header)
    
    output_parcellation.to_filename(output_file)
    print('Done.')
    
    
    
def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Perform linear interpolation'
    )
    parser.add_argument(
        "--WM-parcellation","--WM_pacellation",help='the WM parccelation',
        required=True
    )
    parser.add_argument(
        "--group-FOD","--group_FOD",help='the average group warped FOD',
        required=True
    )
    parser.add_argument(
        "--nl-warp","--nl_warp",help='The non linear wrap that transfrom template to group average space',
        required=True
    )
    parser.add_argument(
        "--output-file","--output_file",
        help="The name of the WM parcellation after transformation",
        required=True
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dirpath = generate_binary_mask(WM_parcellation=args.WM_parcellation)
    dirpath = transform_binary_mask(dirpath=dirpath,non_linear_warp=args.nl_warp)
    generate_parcellation_in_group_space(group_image=args.group_FOD,
                                         dirpath=dirpath,
                                         output_file=args.output_file)
    
    
    if dirpath:
        shutil.rmtree(dirpath)
        print('Deleted the temporary path')
        
if __name__ == "__main__":
    main()

    