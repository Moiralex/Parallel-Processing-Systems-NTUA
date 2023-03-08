#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_omp_helloworld

## Output and error files
#PBS -o results.out
#PBS -e errors.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:30:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab20/ex1
export OMP_NUM_THREADS=1
./exec 64 1000
./exec 1024 1000
./exec 4096 1000
echo "2 threads"
export OMP_NUM_THREADS=2
./exec 64 1000
./exec 1024 1000
./exec 4096 1000
echo "4 threads"
export OMP_NUM_THREADS=4
./exec 64 1000
./exec 1024 1000
./exec 4096 1000
echo "6 threads"
export OMP_NUM_THREADS=6
./exec 64 1000
./exec 1024 1000
./exec 4096 1000
echo "8 threads"
export OMP_NUM_THREADS=8
./exec 64 1000
./exec 1024 1000
./exec 4096 1000
