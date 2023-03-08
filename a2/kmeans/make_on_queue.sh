#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_omp_naive_kmeans

## Output and error files
#PBS -o outdir/make_omp_naive_kmeans.out
#PBS -e errdir/make_omp_naive_kmeans.err

## How many machines should we get? 
##PBS -l nodes=1:ppn=1
#PBS -l nodes=sandman:ppn=1

##How long should the job run for?
#PBS -l walltime=00:01:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a2/kmeans
make
