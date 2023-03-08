#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_tiled

## Output and error files
#PBS -o output/run_all_tiled.out
#PBS -e output/run_all_tiled.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:30:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/FW/tiled
for BLOCKSIZE in 16 32 64 128 256
do
for THREADS in 1 2 4 8 16 32 64
do
export OMP_NUM_THREADS=$THREADS
./fw_tiled 4096 $BLOCKSIZE
done
done
