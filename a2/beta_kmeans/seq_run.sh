#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o output/kmeans_seq.out
#PBS -e output/kmeans_seq.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:02:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/beta_kmeans
export OMP_NUM_THREADS=64
export GOMP_CPU_AFFINITY="0-63"

./kmeans_seq -s 256 -n 16 -c 16 -l 10
##./kmeans_omp_naive -s 256 -n 16 -c 16 -l 10
##./kmeans_omp_reduction -s 256 -n 16 -c 16 -l 10
