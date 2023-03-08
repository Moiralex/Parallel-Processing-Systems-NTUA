#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_fw

## Output and error files
#PBS -o output/make_fw.out
#PBS -e output/make_fw.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=1

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/FW/tiled
make fw_tiled
