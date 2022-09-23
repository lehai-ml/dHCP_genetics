#!bin/bash

dwi_data="data"
rf_data="dhcp_neo_dMRI_derived"
participants_info="dHCP_participant_info.csv"
usable_subjects="usable_subj.txt"

if [ ! -f "$usable_subjects" ]; then
(
echo "list of usable subjects has not yet been generated"
#PREPROCESSING
#first find how many available subjects are in the preprocessed DWI file (ShardRecon04_dstriped), and in the response file (dhcp_neo_dMRI_derived).

#folder=$1

#find $folder -maxdepth 2 -type d | grep /ses 

echo "finding $dwi_data and $rf_data available subjects"
find $dwi_data/ -maxdepth 2 -type d | grep /ses | sed -e "s/$dwi_data\///" > tmp_${dwi_data}_avail_subj.txt
find $rf_data/ -maxdepth 2 -type d | grep /ses | sed -e "s/$rf_data\///" > tmp_${rf_data}_avail_subj.txt

#compare between the two files using the comm comand, i.e. comm -12 <(sort file1) <(sort file2) > all_dHCP_subj.txt
echo "find total available subjects"
comm -12 <(sort tmp_${dwi_data}_avail_subj.txt) <(sort tmp_${rf_data}_avail_subj.txt) > tmp_all_avail_subj.txt

#open the participants file and see which ones can be used in our case
# match the participants with the available list of subjects.

echo "selecting subjects based on $participants_info"
cat dHCP_participant_info.csv | while read subj; do
(
ID=$(cut -d, -f1 <<< $subj)
SES=$(cut -d, -f3 <<< $subj)
Gender=$(cut -d, -f6 <<< $subj)
GA=$(cut -d, -f4 <<< $subj)
PMA=$(cut -d, -f5 <<< $subj)
if [ $ID == "ID" ]; then continue; fi
if (( $(echo "$PMA < 37" | bc -l))); then 
#   echo "skipping $ID with this $GA and $PMA"	
continue; fi

if (( $(echo "$GA < 37" | bc -l ))) ; then
    echo "sub-$ID/ses-$SES,$Gender,$GA,$PMA,preterm"
else
    echo "sub-$ID/ses-$SES,$Gender,$GA,$PMA,term"
fi
)
done > tmp_usable_subj.txt

#join the tmp_usable_subj with the tmp_available_subj to get the list and covariates of the usable subjects
#-F, separate the filed
echo "generating usable_subj.txt"
awk -F, 'NR==FNR{a[$1];next} $1 in a{print $0}' tmp_all_avail_subj.txt tmp_usable_subj.txt > tmp_all_usable_subj.txt

cat tmp_all_usable_subj.txt | sed -e 's/\/ses/,ses/' | sort -t, -u -k1 > usable_subj.txt

rm tmp*.txt
)
fi
# usable_subj.txt contains the name of the subj, the session, the Gender, GA, PMA, and termness

if [ ! -f subjects_list.txt ]; then
    shuf -n 2 $usable_subjects > subjects_list.txt
fi

#shuf -n 3 $usable_subjects | while read subj; do
#(
#ID=$(cut -d, -f1 <<< $subj)
#SES=$(cut -d, -f3 <<< $subj)
#Gender=$(cut -d, -f6 <<< $subj)
#GA=$(cut -d, -f4 <<< $subj)
#PMA=$(cut -d, -f5 <<< $subj)
#)
#done

