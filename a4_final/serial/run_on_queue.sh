#!/bin/bash

## Give the Job a descriptive name
#PBS -N runjob

## Output and error files
#PBS -o outdir/run_6144.out
#PBS -e errdir/run.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8   

#PBS -l walltime=00:05:00

## Start
## Run make in the src folder (modify properly)
cd /home/parallel/parlab16/a4_final/serial
./jacobi 6144
./seidelsor 6144
./redblacksor 6144
