#!/bin/bash

## Give the Job a descriptive name
#PBS -N lists

## Output and error files
#PBS -o results/list_8192_serial.out
#PBS -e results/list_8192_serial.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:30:00

## Start 
## Run make in the src folder (modify properly)

#module load openmp
cd /home/parallel/parlab16/a4/conc_ll

for percentages in 100-0-0 80-10-10 20-40-40 0-50-50
do
    echo "#############################################################################################################################"
    for cores in 0
    do
        IFS='-'
        read -a starr <<< $percentages
        export MT_CONF=$cores
        ./x.serial 8192 ${starr[0]} ${starr[1]} ${starr[2]}
        echo ""
        #echo "8192 ${starr[0]} ${starr[1]} ${starr[2]}"
    done
done
