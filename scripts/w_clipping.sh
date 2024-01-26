#! /bin/bash

echo -e "\nRunning 15 toys with w_clip = 5 \n"
python run_toys.py -p toy.py -t 15 -l True -w 5

echo -e "\nRunning 15 toys with w_clip = 10 \n"
python run_toys.py -p toy.py -t 15 -l True -w 10

echo -e "\nRunning 15 toys with w_clip = 15 \n"
python run_toys.py -p toy.py -t 15 -l True -w 15

echo -e "\nRunning 15 toys with w_clip = 20 \n"
python run_toys.py -p toy.py -t 15 -l True -w 20

echo -e "\nRunning 15 toys with w_clip = 25 \n"
python run_toys.py -p toy.py -t 15 -l True -w 25

echo -e "\nRunning 15 toys with w_clip = 30 \n"
python run_toys.py -p toy.py -t 15 -l True -w 30