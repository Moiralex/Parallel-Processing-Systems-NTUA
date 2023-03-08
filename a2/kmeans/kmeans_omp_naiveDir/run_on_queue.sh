#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o outdir/run_kmeans.out
#PBS -e errdir/run_kmeans.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/kmeans
export OMP_NUM_THREADS=8
./kmeans_seq -s 256 -n 16 -c 16 -l 10
