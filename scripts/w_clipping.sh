#! /bin/bash

echo -e "\nRunning 40 toys with w_clip = 7 \n"
python run_toys.py -p toy.py -t 50 -l True -w 7.4
python run_toys.py -p toy.py -t 50 -l True -w 7.5
python run_toys.py -p toy.py -t 50 -l True -w 7.6
python run_toys.py -p toy.py -t 50 -l True -w 8.6
# python run_toys.py -p toy.py -t 50 -l True -w 8.8
# python run_toys.py -p toy.py -t 50 -l True -w 9.0
# python run_toys.py -p toy.py -t 50 -l True -w 9.2
