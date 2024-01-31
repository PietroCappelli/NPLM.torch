#! /bin/bash

echo -e "\nRunning 40 toys with w_clip = 5 \n"
python run_toys.py -p toy.py -t 40 -l True -w 4.0
python run_toys.py -p toy.py -t 40 -l True -w 4.5
python run_toys.py -p toy.py -t 40 -l True -w 5.0
python run_toys.py -p toy.py -t 40 -l True -w 5.5
python run_toys.py -p toy.py -t 40 -l True -w 6.0
python run_toys.py -p toy.py -t 40 -l True -w 6.5
python run_toys.py -p toy.py -t 40 -l True -w 7
python run_toys.py -p toy.py -t 40 -l True -w 7.5
python run_toys.py -p toy.py -t 40 -l True -w 8
python run_toys.py -p toy.py -t 40 -l True -w 8.5
python run_toys.py -p toy.py -t 40 -l True -w 9

echo -e "\nRunning 40 toys with w_clip = 30 \n"
python run_toys.py -p toy.py -t 40 -l True -w 29
python run_toys.py -p toy.py -t 40 -l True -w 29.5
python run_toys.py -p toy.py -t 40 -l True -w 30
python run_toys.py -p toy.py -t 40 -l True -w 30.5
python run_toys.py -p toy.py -t 40 -l True -w 31
python run_toys.py -p toy.py -t 40 -l True -w 31.5