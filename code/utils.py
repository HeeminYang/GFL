from torch import nn

import numpy as np

import logging
from collections import OrderedDict
import copy

# set logger
def get_logger(log_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    return logger

# custom weights initialization
def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)

# Adjacency Matrix
def W_t(num_client,p,seed):
    np.random.seed(seed)
    while True:
        A = np.random.random((num_client,num_client));
        A[A<p] = 1;
        A[A<1] = 0;
        #symmetrize A, get adjacency matrix
        A = np.triu(A,1); A = A + A.T;
        #get laplacian
        L = -A;
        for k in range(num_client):
            L[k,k] = sum(A[k,:]);

        eig_L = np.linalg.eig(L)[0];
        pos_eig_0 = np.where(np.abs(eig_L) <1e-5)[0];
        if len(pos_eig_0)==1:
            break;
    
    A+=np.eye(A.shape[0])
    
    return A

def fed_avg(sampled_list, W_k):

    total = int(sum(W_k))
    # count = 0
    aggreted_model = {}
    for key in sampled_list[0].classifier.state_dict():
        summed_weight = sum([client.classifier.state_dict()[key]*int(W_k[i]) for i, client in enumerate(sampled_list)])
        # if count == 0:
        #     for i, client in enumerate(sampled_list):
        #         logger.info(f'Client {i}\n{client.classifier.state_dict()[key][0]}')
        avg_weight = summed_weight/total
        # if count == 0:
        #     logger.info(f'Aggreted\n{avg_weight[0]}')
        # count += 1
        aggreted_model[key] = avg_weight

    return aggreted_model

    # # Initialize the global model weights
    # model = OrderedDict()
    # ratio = float(1/sum(W_k))

    # for i, cli in enumerate(sampled_list):

    #     if i == 0:
    #         for key in cli.classifier.state_dict().keys():
    #             model[key] = copy.deepcopy(cli.classifier.state_dict()[key])*ratio*W_k[i]
    #     else:
    #         for key in cli.classifier.state_dict().keys():
    #             model[key] += copy.deepcopy(cli.classifier.state_dict()[key])*ratio*W_k[i]

    # return model