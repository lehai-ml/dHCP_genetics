#!/bin/bash

#symbolic link to get the necessary folders
#diffusion dataset is found in /isi01/dhcp-pipeline-data/kcl/diffusion/ShardRecon04_dstriped/
#dhcp_neo_dMRI_derived -> this is where the map from individual to 40 weeks sub-CC00416XX12_ses-128900_from-dmrishard_to-extdhcp40wk_mode-image.mif.gz - found in /projects/perinatal/peridata/Hai/dhcp_neo_dMRI_derived
#atlas are found in /projects/perinatal/peridata/Hai/atlas/ - these are used to perform 5TT

if [ ! -d data ]; then
	echo "dhcp-pipeline-data not yet mounted"
	sudo mount -t cifs //isi01/dhcp-pipeline-data /home/lh20/dhcp-pipeline-data/-o username=lh20,domain=isd,iocharset=utf8,vers=2.1
	echo "dhcp-pipeline-data mounted"
fi

if [ ! -d output ]; then
	echo "dhcp-reconstructions not yet mounted"
	sudo mount -t cifs //pnraw01/dhcp-reconstructions /home/lh20/dhcp-reconstructions/-o username=lh20,domain=isd,iocharset=utf8,vers=2.1
	echo "dhcp-reconstructions mounted"
fi
