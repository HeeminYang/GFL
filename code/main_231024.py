from code.Data_231028 import create_federated_loaders
from client import client
from cDCGAN import Generator_MNIST, Discriminator_MNIST
from process import run, simpleFL
from model import CNN
from utils import get_logger, W_t

import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import os
import argparse
import time
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Adam_beta1', type=float, default=0.5)
    parser.add_argument('--Adam_lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--clf_epoch', type=int, default=1)
    parser.add_argument('--client', type=int, default=10)
    parser.add_argument('--d_iter', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='FashionMNIST', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--g_img_num', type=int, default=100)
    parser.add_argument('--g_epoch', type=int, default=1)
    parser.add_argument('--Ksteps', type=int, default=2)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--LF_set', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--malicious_stage', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--p_stage', type=float, default=1.0)
    parser.add_argument('--project', type=str, default="GFL", help='project name', choices=['GFL', 'FL'])
    parser.add_argument('--round', type=int, default=200)
    parser.add_argument('--size_z', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--warm_up', type=int, default=1)
    args = parser.parse_args()

    # fix random seed
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(777)
    random.seed(777)
    os.environ['PYTHONHASHSEED'] = str(777)
    
    # malicious client
    args.malicious = int(args.client * args.malicious_stage)

    # project name
    if args.project == 'GFL':
        args.project_name = f'{args.project}_{args.dataset}_c{args.client}_m{args.malicious}_p{args.p}_pstage{args.p_stage}_warmup{args.warm_up}_round{args.round}_gepoch{args.g_epoch}_diter{args.d_iter}_Ksteps{args.Ksteps}'
    elif args.project == 'FL':
        args.project_name = f'{args.project}_{args.dataset}_c{args.client}_m{args.malicious}_lf{args.LF_set}_p{args.p}_pstage{args.p_stage}_round{args.round}'

    # logger
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = f'./log/{now}_{args.project_name}.log'
    logger = get_logger(log_path)
    
    # use cuda if available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")
    logger.info(f"Using {DEVICE} backend")

    # Adjacency Matrix
    R = W_t(args.client, args.p, 777)
    logger.info(f'Adjacency Matrix:\n{R}')

    # data loader 
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'MNIST':
        args.source = 7
        args.target = 1
        logger.info('Train dataset loading...')
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loaders = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger)
        logger.info('Test dataset loading...')
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loaders = create_federated_loaders(test_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger)
        logger.info('External dataset loading...')
        external_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        external_loaders = create_federated_loaders(external_dataset, 1, args.malicious, 0, 0, args.p_stage, args.batch_size, logger)

    elif args.dataset == 'FashionMNIST':
        if args.LF_set == 1:
            args.source = 6
            args.target = 0
        elif args.LF_set == 2:
            args.source = 1
            args.target = 3
        elif args.LF_set == 3:
            args.source = 4
            args.target = 6
        logger.info('Train dataset loading...')
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        train_loaders = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger)
        logger.info('Test dataset loading...')
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        test_loaders = create_federated_loaders(test_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger)
        logger.info('External dataset loading...')
        external_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        external_loaders = create_federated_loaders(external_dataset, 1, 0, 0, 0, args.p_stage, args.batch_size, logger)


    elif args.dataset == 'CIFAR10':
        if args.LF_set == 1:
            args.source = 5
            args.target = 3
        elif args.LF_set == 2:
            args.source = 0
            args.target = 2
        elif args.LF_set == 3:
            args.source = 0
            args.target = 9

        logger.info('Train dataset loading...')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loaders = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, logger)
        
        logger.info('Test dataset loading...')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loaders = create_federated_loaders(test_dataset, args.client, 0, args.source, args.target, args.p_stage, logger)

        logger.info('External dataset loading...')
        external_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        external_loaders = create_federated_loaders(external_dataset, 1, 0, 0, 0, args.p_stage, logger)
    dataiter = iter(train_loaders[0])
    images, labels = next(dataiter)
    # MNIST image size is 1*28*28
    args.img_size = images.shape[2]

    # GAN
    gan_list = []
    for _ in range(args.client):
        G = Generator_MNIST()
        D = Discriminator_MNIST()
        gan_list.append([copy.deepcopy(G), copy.deepcopy(D)])

    # make clients
    clients = []
    for i in range(args.client):
        clients.append(copy.deepcopy(client(args, gan_list[i][0], gan_list[i][1], train_loaders[i], test_loaders[i], external_loaders[0], 
                                            DEVICE, logger, args.source, args.target, copy.deepcopy(CNN()), copy.deepcopy(CNN()))))
    
    # run
    save_path = f'./project'
    if args.project == 'GFL':
        run(args, clients, save_path, R, logger)
    elif args.project == 'FL':
        simpleFL(args, clients, save_path, R, logger)
