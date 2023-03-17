Read this markdown on [Github](https://github.com/lehai-ml/dHCP_genetics/tree/main/codes/FBA)
This folder contains several scripts and folders, they are summarised as follows:

# Script files
1. <code>process.sh</code> : This is where all the variables are defined. It is also the only executable sh file. All the individual scripts are called in here.
2. <code>support_functions.sh</code> : This file contains all the support functions used in the scripts. For instance, run function can be used to check if the command should be executed or not.
3. FBA-related files: See individual files for description of each command

|File| Description |
|----|-------------|
|```generate_ID_list.sh```|Used to call ```generate_ID_list.py```|
|<code>calculate_fods.sh</code>| Calculate FODs | 
|<code>compute_average_masks_and_fods.sh</code>| Create a mean FOD map and mask and convert FODs to fixel |
|<code>calculate_fixel_metrics.sh</code>| Register to FODs to common template and calculate fixel measures |
|<code>gen5tt.sh</code>| Generate 5 tissue map |
|<code>tractography.sh</code>| Perform tractography and calculate fixel2fixel connectivity|
|<code>perform_fba.sh</code>| Perform fixelcfestats whole-brain|
 
 See the [MRtrix3 FBA pipeline](https://mrtrix.readthedocs.io/en/0.3.16/workflows/fixel_based_analysis.html)

4. TBSS-related files
|File| Description|
|----|------------|
|<code>perform_tbss.sh</code> | Use to perform TBSS in babies. Uses DTI-TK and FSL |

See [DTI-TK preprocessing and registration tutorial](https://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage)
See [FSL TBSS User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide)

5. Tract-based or Atlas-based statistics files
|File| Description |
|----|-------------|
|```perform_aba.sh``` | Register individual FODs map to neonatal WM atlases and calculate mean value for each ROI|
|```perform_fba_wm.sh```| Perform fixelcfestats individual tracts|

See [Alena Uss Paper](https://www.frontiersin.org/articles/10.3389/fnins.2021.661704/full)
The atlas is downloaded [here](https://gin.g-node.org/alenaullauus/4d_multi-channel_neonatal_brain_mri_atlas)

6. Other files
|File| Description |
|----|-------------|
|```generate_ID_list.py```|Python command used to generate ID list as well as design and contrast matrices|

# Folders: Input and output files
To keep everything tidy. Make sure every file is defined in the <code>process.sh</code> file and only the <code>process.sh</code> is executed.
The output for each script command is more or less shown in the <code>process.sh</code> or in their individual scripts. Generally, main output for a particular analysis is defined in as <code>output</code> in the same directory where <code>process.sh</code> is executed.

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
|<code>data</code>|<code>dhcp-pipeline-data/kcl/diffusion/ShardRecond04_dstriped/</code>|contains DWI data <code>postmc_dstriped-dwi300.mif</code> and bet mask <code>mask_T2w_brainmask_processed.nii.gz</code>|
|<code>dhcp_neo_dMRI_derived</code>|<code>/projects/perinatal/peridata/Hai/dhcp_neo_dMRI_derived</code>| contains warps in 40 weeks <code>fron-dmirshard_to-extdhcp40wk_mode-image.mif.gz</code>, wm and csf response function <code>dHCP_atlas_v2.1_rf_wm.dhsfa015_44</code> and <code>dHCP_atlas_v2.1_rf_csf.dhsfa015</code>|
|<code>atlas</code>|<code>projects/perinatal/peridata/Hai/atlas/</code>|contains 40 weeks extended templates and warps |

2. User-defined text files
|File|Description|```variable```|Note|
|----|-----------|--------------|----|
|```subjects.txt```|comma separated file, where the first column is sub-id/ses, can be generated with ```generate_ID_list.sh```,required for ```process.sh```|
|```wm_tract.txt```|use with function ```generate_wm_tract```, see below|

# Support functions explained

Described here are convenience functions (found in ```support_functions.sh```) used in the above mentioned scripts.

|Function|Description|Usage|
|--------|-----------|----|
|```run```|Check if the specified command need to be run. If the output is missing or the input has been updated, the command is run, else skip.|```run description command IN:input OUT:output```, ```run description command IN:input OUT-name:output_prefix```|
|```update_folder_if_needed```|wrapper function of ```run``` if the output is a folder, then check if inside components have been updated|```update_folder_if_needed run ...```|
|```sanity_check```|check if two folders have the same set of IDs|Use at your own risk, it will delete the whole folder if condition is not met|
|```generate_wm_tract```|Applying ```tckgen``` or ```tckedit``` to rows or identifier in the ```wm_tracts.txt```.|Each row in the ```wm_tracts.txt``` define the following ```IDENTIFIER:{tckoperation}``` for example ```CING_D_R-seed_image``` means for identifier ```CING_D_R``` use whatever follows as the argument in ```tckgen```|