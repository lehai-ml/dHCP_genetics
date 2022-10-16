#!/bin/bash


# Relevant files to access
src="$(pwd)"
dwi_data=data
dwi=postmc_dstriped-dwi300.mif
bet_mask=mask_T2w_brainmask_processed.nii.gz
warps=dhcp_neo_dMRI_derived
warps_in_40wk=from-dmrishard_to-extdhcp40wk_mode-image.mif.gz
wm_response=dHCP_atlas_v2.1_rf_wm.dhsfa015_44 # this response function is generated from 21 WM response functions from subjects aged 44.1 weeks
csf_response=dHCP_atlas_v2.1_rf_csf.dhsfa015 # this response function is generated by averaging all CSF response functions.
subjects_list=subjects_list.txt
output_folder=output

# Relevant files for generate_ID_list.sh
rf_data="dhcp_neo_dMRI_derived"
participants_info="dHCP_participant_info.csv"
usable_subjects="usable_subj.txt"
all_available_IDs=all_available_IDs.txt
usable_subj=usable_subj.txt
euro_SCZ_PRS_term="euro_SCZ_PRS_term.csv"
# Relevant files for calculate_fods.sh
mask=mask.mif
wm_fod=wm_fod.mif
csf_fod=csf_fod.mif
wm_norm_fod=wm_norm_fod.mif
csf_norm_fod=csf_norm_fod.mif
warped_mask_in_dHCP_40wk=warped_mask_in_dHCP_40wk.mif
warped_wm_fod_in_dHCP_40wk=warped_wm_fod_in_dHCP_40wk.mif

# Relevant files for compute_average_masks_and_fods.sh
warped_mask_average=warped_mask_average.mif
warped_wm_fod_average=warped_wm_fod_average.mif
fixel_mask=fixel_mask

# Relevant files for calculate_fixel_metrics.sh
native2average_warp=native2average_warp.mif
average2native_warp=average2native_warp.mif

fod_in_template_space_NOT_REORIENTED=fod_in_template_space_NOT_REORIENTED.mif
fixel_in_template_space_NOT_REORIENTED=fixel_in_template_space_NOT_REORIENTED
fd=fd.mif
fixel_in_template_space=fixel_in_template_space

all_subj_fd=all_subj_fd 
all_subj_fc=all_subj_fc
all_subj_log_fc=all_subj_log_fc
all_subj_fdc=all_subj_fdc

# Relevant files for 5tt.sh
templates="${src}/atlas/templates"
KANA_in_template_space="${templates}/KANA_in_template_space_2.nii.gz"
Tissue_segmented="${templates}/week40_tissue_dseg.nii.gz"
#6 = cerbellum 7 and 8 = brain stem and deep gray matter
KANA_DGM=KANA_DGM.mif
regrid_KANA_in_template=regrid_KANA_in_template_space.mif
regrid_Tissue_segmented=regrid_Tissue_segmented.mif
output_5TT=5TT
wm=wm.mif
gm=gm.mif
dgm=dgm.mif
csf=csf.mif
image_5TT=5TT.mif

# Relevant files for tractography.sh
output_tractography=tractography
number_of_streamlines=10000000
tracts="tracts_${number_of_streamlines}.tck"
reduced_number_of_streamlines=1000000
reduced_tracts=reduced_tracts_${reduced_number_of_streamlines}.tck
fba_output=fba
ffixel_matrix=ffixel_matrix

#Relevant files for perform_fba.sh

all_subj_fd_smooth=all_subj_fd_smooth
all_subj_log_fc_smooth=all_subj_log_fc_smooth
all_subj_fdc_smooth=all_subj_fdc_smooth
id_file=id_file.txt
design_matrix=design_matrix.txt
contrast_matrix=contrast_matrix.txt

stats_fd=stats_fd
stats_log_fc=stats_log_fc
stats_fdc=stats_fdc

#Relevant files for calculate_dti.sh
diffusion_tensor=diffusion_tensor.mif
dt_fa=dt_fa.mif
dt_adc=dt_adc.mif
dt_rd=dt_rd.mif
dt_ad=dt_ad.mif

set -e

. support_functions.sh

. generate_ID_list.sh

if [ ! -d data ]; then
    echo "dhcp-pipeline-data not yet mounted"
    sudo mount -t cifs //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/ -o username=lh20,domain=isd,iocharset=utf8,vers=2.1
    echo "dhcp-pipeline-data mounted"
fi


# build list of subjects:
ID_list=()
while read subj; do
  # skip if empty:
  if [ "x$subj" == "x" ]; then continue; fi

  # split line into components to get folder name:
  IFS=',' read -ra subj <<< $subj
  ID_list+=(${subj[0]})
done < $subjects_list

cd $src
#. calculate_fods.sh
cd $src
#. compute_average_masks_and_fods.sh
cd $src
#. calculate_fixel_metrics.sh
cd $src
#. 5tt.sh
cd $src
#. tractography.sh
cd $src
. perform_fba.sh
cd $src
#. calculate_dti.sh
