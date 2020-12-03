#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet
#PBS -j oe
#PBS -o log/output.log

cd ${PBS_O_WORKDIR}
mkdir -p model log

python main.py
