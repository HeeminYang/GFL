#!/bin/bash
mpirun -np 11 python code/main.py --p_stage 1.0
mpirun -np 11 python code/main.py --p_stage 0.9
mpirun -np 11 python code/main.py --p_stage 0.8
mpirun -np 11 python code/main.py --p_stage 0.7
mpirun -np 11 python code/main.py --p_stage 0.6
mpirun -np 11 python code/main.py --p_stage 0.5
mpirun -np 11 python code/main.py --p_stage 0.4
mpirun -np 11 python code/main.py --p_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.2
mpirun -np 11 python code/main.py --p_stage 0.1
