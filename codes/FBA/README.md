Read this markdown on [Github](https://github.com/lehai-ml/dHCP_genetics/tree/main/codes/FBA)

This folder contains several scripts and folders, they are summarised as follows:

# Script files
1. ```process.sh``` : This is where all the variables are defined. It is also the only executable sh file. All the individual scripts are called in here.
2. ```support_functions.sh``` : This file contains all the support functions used in the scripts. For instance, run function can be used to check if the command should be executed or not.
3. FBA-related files: See individual files for description of each command

|File| Description |
|----|-------------|
|```generate_ID_list.sh```|Used to call ```generate_ID_list.py```|
|```calculate_fods.sh```| Calculate FODs | 
|```compute_average_masks_and_fods.sh```| Create a mean FOD map and mask and convert FODs to fixel |
|```calculate_fixel_metrics.sh```| Register to FODs to common template and calculate fixel measures |
|```gen5tt.sh```| Generate 5 tissue map |
|```tractography.sh```| Perform tractography and calculate fixel2fixel connectivity|
|```perform_fba.sh```| Perform fixelcfestats whole-brain|
 
 See the [MRtrix3 FBA pipeline](https://mrtrix.readthedocs.io/en/0.3.16/workflows/fixel_based_analysis.html)

4. TBSS-related or FSL randomise related files

|File| Description|
|----|------------|
|```perform_tbss.sh``` | Use to perform TBSS in babies. Uses DTI-TK and FSL |
|```calculate_dti.sh```|Calculate tensor metrics using MRtrix3|

See [DTI-TK preprocessing and registration tutorial](https://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage)
See [FSL TBSS User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide)

5. Tract-based or Atlas-based statistics files

|File| Description |
|----|-------------|
|```perform_aba.sh``` | Register individual FODs map to neonatal WM atlases and calculate mean value for each ROI|
|```genwmtracts.sh``` | Generate WM tracts |
|```perform_fba_wm.sh```| Perform fixelcfestats individual tracts|

See [Alena Uss Paper](https://www.frontiersin.org/articles/10.3389/fnins.2021.661704/full)
The atlas is downloaded [here](https://gin.g-node.org/alenaullauus/4d_multi-channel_neonatal_brain_mri_atlas)

6. Other files

|File| Description |
|----|-------------|
|```generate_ID_list.py```|Python command used to generate ID list as well as design and contrast matrices|

# Folders: Input and output files
To keep everything tidy. Make sure every file is defined in the ```process.sh``` file and only the ```process.sh``` is executed.
The output for each script command is more or less shown in the ```process.sh``` or in their individual scripts. Generally, main output for a particular analysis is defined in as ```output``` in the same directory where ```process.sh``` is executed.

**NB: for obvious reasons these files are not commited to github.**

The main folders are ordered as follows (example only, there are many other folders in output that are not mentioned, but are results of the FBA pipeline)

```
process.sh <-execute .sh here
output/
|____5TT/
|____all_subj_{fc,fdc,fd,log_fc}/
|    |    sub-CC00*.mif <- calculated fixel metrics from FODs
|____fixel_mask/
|    |    {directions,index}.mif <- whole brain fixel mask
|____tractography/
|    |    tracts_20000000.tck <- whole brain tractography
|    |____fba/
|    |    |____SCZPRSPCA/ <- target of interest
|    |    |    |____stats_{fd,fc,log_fc}/
|    |    |    |    |    fwe_1mpvalue.mif <- results of whole-brain FBA
|    |____individual_tracts/
|    |    |____corpus-callosum/cc.tck
|    |    |    |    cc.tck
|    |    |    |____cc_stats_{fd,fc,log_fc}/ 
|    |    |    |    |    fwe_1mpvalue.mif <- results of individual tract FBA
|____sub-CC00*/ses*/
|    |    mask.mif
|    |    {csf,wm}_{norm}_fod.mif
|    |    warped_{mask,wm_fod}_in_dHCP_40wk.mif
|    |    fod_in_template_space_NOT_REORIENTED.mif
|    |    wm_parc2native_warp.mif
|    |    wm_parc_in_subject_space.mif
|    |____fixel_in_template_space_{NOT_REORIENTED}/
|____tbss/
|____|____sub-CC00*/ses*/
|    |    |     dti_FA.nii.gz <- DTI calculated by FSL
|____|____DTI_TK_processed/
|    |    |     *{aff,diffeo}.nii.gz <- registration by DTI-TK
|____|____stats/ 
|    |    |    *_tfce_corrp_tstat1.nii.gz <-result of TBSS
|____glass_brain/
```

1. Symbolically linked folders and files

| Folder |Linked from |Description |
|--------|------------|------------|
|```data```|```dhcp-pipeline-data/kcl/diffusion/ShardRecond04_dstriped/```|contains DWI data ```postmc_dstriped-dwi300.mif``` and bet mask ```mask_T2w_brainmask_processed.nii.gz```|
|```dhcp_neo_dMRI_derived```|```/projects/perinatal/peridata/Hai/dhcp_neo_dMRI_derived```| contains warps in 40 weeks ```fron-dmirshard_to-extdhcp40wk_mode-image.mif.gz```, wm and csf response function ```dHCP_atlas_v2.1_rf_wm.dhsfa015_44``` and ```dHCP_atlas_v2.1_rf_csf.dhsfa015```|
|```atlas```|```projects/perinatal/peridata/Hai/atlas/```|contains 40 weeks extended templates and warps |
|```wm_parcellation```|Downloaded from [here](https://gin.g-node.org/alenaullauus/4d_multi-channel_neonatal_brain_mri_atlas)||

2. User-defined text files

|File|Description|
|----|-----------|
|```subjects.txt```|comma separated file, where the first column is sub-id/ses, can be generated with ```generate_ID_list.sh```,required for ```process.sh```|
|```wm_tract.txt```|use with function ```generate_wm_tract```, see below|
|```output/tbss/DTI_TK_processed/ID_template```|List of ```dti_res.nii.gz``` files that will be used to create template|

3. Output files in individual folders

|File|Found in|
|----|--------|
|```mask.mif```|```calculate_fods.sh```|
|```{csf,wm}_fod.mif```|```calculate_fods.sh```|
|```warped_{mask,wm_fod}_in_dHCP_40wk.mif```|```compute_average_masks_and_fods.sh```|
|```average2native_warp.mif```|```calculate_fixel_metrics.sh```|
|```native2average_warp.mif```|```calculate_fixel_metrics.sh```|
|```fixel_in_template_space```|```calculate_fixel_metrics.sh```|
|```fixel_in_template_space_NOT_REORIENTED```|```calculate_fixel_metrics.sh```|
|```fod_in_template_space_NOT_REORIENTED```|```calculate_fixel_metrics.sh```|
|```wm_parc2native_warp.mif```|```perform_aba.sh```|
|```wm_parc_in_subject_space.mif```|```perfomr_aba.sh```|
|```diffusion_tensor.mif```|```calculate_dti.sh```|
|```dt_{fa,adc,rd,ad}```|

# Support functions explained

Described here are convenience functions (found in ```support_functions.sh```) used in the above mentioned scripts.

|Function|Description|Usage|
|--------|-----------|----|
|```run```|Check if the specified command need to be run. If the output is missing or the input has been updated, the command is run, else skip.|```run description command IN:input OUT:output```, ```run description command IN:input OUT-name:output_prefix```|
|```update_folder_if_needed```|wrapper function of ```run``` if the output is a folder, then check if inside components have been updated|```update_folder_if_needed run ...```|
|```sanity_check```|check if two folders have the same set of IDs|Use at your own risk, it will delete the whole folder if condition is not met|
|```generate_wm_tract```|Applying ```tckgen``` or ```tckedit``` to rows or identifier in the ```wm_tracts.txt```.|Each row in the ```wm_tracts.txt``` define the following ```IDENTIFIER:{tckoperation}``` for example ```CING_D_R-seed_image``` means for identifier ```CING_D_R``` use whatever follows as the argument in ```tckgen```|