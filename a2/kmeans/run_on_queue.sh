#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_keams

## Output and error files
#PBS -o outdir/smallConfiguration/run_numa_aware_16_new.out
#PBS -e errdir/smallConfiguration/run_numa_aware_16_new.err

## How many machines should we get? 
##PBS -l nodes=1:ppn=8
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:02:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/kmeans
export OMP_NUM_THREADS=16

export GOMP_CPU_AFFINITY="0-15"

./kmeans_omp_reduction_numa_aware -s 256 -n 1 -c 4 -l 10
