#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_omp_Game_Of_Life

## Output and error files
#PBS -o outdir/run_omp_GOL_8_4096_new.out
#PBS -e errdir/run_omp_GOL_8_4096_new.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab16/a1b
export OMP_NUM_THREADS=8
./omp_Game_Of_Life 4096 1000

