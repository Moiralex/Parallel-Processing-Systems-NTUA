#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_tiled

## Output and error files
#PBS -o output/run_tiled.out
#PBS -e output/run_tiled.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:01:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/FW/tiled
export OMP_NUM_THREADS=64
##export GOMP_CPU_AFFINITY="0-4"
./fw_tiled 2048 16
##./fw 2048
## ./fw_sr <SIZE> <BSIZE>
## ./fw_tiled <SIZE> <BSIZE>
