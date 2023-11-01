#!/bin/bash
mpirun -np 11 python code/main.py --p_stage 1.0 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.9 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.8 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.7 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.6 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.5 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.4 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.3 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.2 --malicious_stage 0.3
mpirun -np 11 python code/main.py --p_stage 0.1 --malicious_stage 0.3
