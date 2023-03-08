#!/bin/bash

## Give the Job a descriptive name
#PBS -N get_gpu_info

## Output and error files
#PBS -o get_gpu_info.out
#PBS -e get_gpu_info.err

## How many machines should we get? 
#PBS -l nodes=dungani:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

cd /home/parallel/parlab16/a5/kmeans
export CUDA_VISIBLE_DEVICES=2

./get_gpu_info
