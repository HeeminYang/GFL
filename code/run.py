import subprocess
command = ['mpirun' , '-np' , '10' , 'python' , '/home/heemin/GFL/code/mpi.py']
subprocess.run(command)
