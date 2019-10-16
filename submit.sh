#!/bin/bash

#$ -N HGdecomp
#$ -cwd
#$ -M varunvai@stanford.edu
#$ -m besan
#$ -j y

python HGparallel.py
