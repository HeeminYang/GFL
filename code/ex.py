from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 초기 W 값 설정 (실제 코드에서는 이미 설정되어 있을 것입니다.)
if rank == 0:
    W = np.ones((size, size))
else:
    W = None

# 각 프로세스는 자신의 W_k 값을 수정합니다.
W_k = np.ones(size) * rank

# rank 0에서 모든 W_k를 수집합니다.
gathered_W = comm.gather(W_k, root=0)

# rank 0에서 W를 업데이트합니다.
if rank == 0:
    for i, row in enumerate(gathered_W):
        W[i] = row
    print(W)
