This folder contains several scripts and folders, they are summarised as follows:

# Script files
1. <code>process.sh</code> : This is where all the variables are defined. It is also the only executable sh file. All the individual scripts are called in here.
2. <code>support_functions.sh</code> : This file contains all the support functions used in the scripts. For instance, run function can be used to check if the command should be executed or not.
3. FBA-related files: See individual files for description of each command

|File| Description |
|----|-------------|
|<code>calculate_fods.sh</code>| Calculate FODs | 
|<code>compute_average_masks_and_fods.sh</code>| Create a mean FOD map and mask and convert FODs to fixel |
|<code>calculate_fixel_metrics.sh</code>| Register to FODs to common template and calculate fixel measures |
|<code>gen5tt.sh</code>| Generate 5 tissue map |
|<code>tractography.sh</code>| Perform tractography and calculate fixel2fixel connectivity|
|<code>perform_fba.sh</code>| Perform fixelcfestats whole-brain|
|<code>perform_fba_wm.sh</code>| Perform fixelcfestats individual tracts see below |
 

# Folders: Input and output files
To keep everything tidy. Make sure every file is defined in the <code>process.sh</code> file and only the <code>process.sh</code> is executed.
The output for each script command is more or less shown in the <code>process.sh</code> or in their individual scripts. Generally, main output for a particular analysis is defined in as <code>output</code> in the same directory where <code>process.sh</code> is executed.

The main folders are ordered as follows (example only, there are many other folders in output that are not mentioned, but are results of the FBA pipeline)

```
process.sh <-execute .sh here
output/
|____5tt/
|____tractography/
|    |____fba/
|    |    |____SCZPRSPCA/ : the results of whole-brain FBA
|    |____individual_tracts/
|    |    |____corpus-callosum/cc.tck
|    |    |    |    cc.tck
|    |    |    |____cc_statsfd/ : results of individual tract FBA
|____glass_brain/
|____tbss/
|____|____DTI_TK_processed/
|____|____stats/
|____sub-CC00*/
```
The main input files that are symbolicly linked are

| Folder |Linked from |Description |
|--------|------------|------------|
|<code>data</code>|<code>dhcp-pipeline-data/kcl/diffusion/ShardRecond04_dstriped/</code>|contains DWI data <code>postmc_dstriped-dwi300.mif</code> and bet mask <code>mask_T2w_brainmask_processed.nii.gz</code>|
|<code>dhcp_neo_dMRI_derived</code>|<code>/projects/perinatal/peridata/Hai/dhcp_neo_dMRI_derived</code>| contains warps in 40 weeks <code>fron-dmirshard_to-extdhcp40wk_mode-image.mif.gz</code>, wm and csf response function <code>dHCP_atlas_v2.1_rf_wm.dhsfa015_44</code> and <code>dHCP_atlas_v2.1_rf_csf.dhsfa015</code>|
|<code>atlas</code>|<code>projects/perinatal/peridata/Hai/atlas/</code>|contains 40 weeks extended templates and warps |


