Read this markdown on [Github](https://github.com/lehai-ml/dHCP_genetics/tree/main/codes/DrawEMVol)

This folder contains scripts necessary for volumetric analysis

# Script files

|File|Description|
|----|-----------|
|```calculate_volumes.sh```| Calculate the DrawEM volumes |
|```support_functions.sh```| Symbolically linked from [here](../FBA/README.md) |
|```do_randomise_Jacob.sh```| Perform FSL randomise on the Jacobians |

# Folders

1. Symbolically linked folders and files

|Folder|Linked from|Description|
|------|-----------|-----------|
|```neonatal_release3```|``` //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/neonatal_release3/BIDS_public/rel3_dhcp_anat_pipeline```|File containing all the anatomical data and parcellation|
|```Jacobians```|```/data/projects/dHCP/Jacobians/``` on **Nan***|The Jacobians matrices; best to contact Dafnis or Oliver-Grant|

2. Files used

|File|Description|
|```week40_T2w.nii.gz```|Downloaded from ```atlas/template/week40_T2w.nii.gz```, template of ext40_dhcp See [here](../FBA/README.md)|

3. Folder order

```
do_randomise_Jacob.sh
output
|    design.mat
|    design.con
|    *_tfce_corrp_tstat1.nii.gz <- FSL randomise output
```

