#! /bin/bash

echo -e "\nRunning 40 toys with w_clip = 7 \n"
python run_toys.py -p toy.py -t 100 -l True -w 7.6
python run_toys.py -p toy.py -t 100 -l True -w 9.2
