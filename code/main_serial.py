from client import client
from utils import fed_avg

from mpi4py import MPI

from Data_231028 import create_federated_loaders
from Data import get_MNIST_loaders, get_FMNIST_loaders
from client import client
from cDCGAN import Generator_MNIST, Discriminator_MNIST
from model import CNN
from utils import get_logger, W_t

import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pickle
import random
import os
import argparse
import time
import copy
from tqdm import tqdm
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Adam_beta1', type=float, default=0.5)
    parser.add_argument('--Adam_lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--clf_epoch', type=int, default=1)
    parser.add_argument('--client', type=int, default=10)
    parser.add_argument('--d_iter', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='FMNIST', choices=['MNIST', 'FMNIST'])
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--g_img_num', type=int, default=100)
    parser.add_argument('--g_epoch', type=int, default=80)
    parser.add_argument('--Ksteps', type=int, default=2)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--LF_set', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--malicious_stage', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--p_stage', type=float, default=1.0)
    parser.add_argument('--project', type=str, default="GFL", help='project name', choices=['GFL', 'FL'])
    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--size_z', type=int, default=100)
    parser.add_argument('--small', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--warm_up', type=int, default=10)
    args = parser.parse_args()

    # fix random seed
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(777)
    random.seed(777)
    os.environ['PYTHONHASHSEED'] = str(777)

    save_path = f'./project'
    # malicious client
    args.malicious = int(args.client * args.malicious_stage)
    # project name
    if args.project == 'GFL':
        args.project_name = f'{args.project}_{args.dataset}_c{args.client}_m{args.malicious}_ps{args.p_stage}_wrmp{args.warm_up}_rd{args.round}_g{args.g_epoch}_clf{args.clf_epoch}_s{str(args.small)}_l{args.local_epoch}'
    elif args.project == 'FL':
        args.project_name = f'{args.project}_{args.dataset}_c{args.client}_m{args.malicious}_lf{args.LF_set}_ps{args.p_stage}_rd{args.round}_s{str(args.small)}_l{args.local_epoch}'
        
    # logger
    now = time.strftime("%Y%m%d_%H%M", time.localtime())
    
    log_dir = f'./log/{args.project_name}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = f'./log/{args.project_name}/{now}_{args.project_name}.log'
    logger = get_logger(log_path)

    # use cuda if available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")
    logger.info(f"Using {DEVICE} backend")
    
    # Adjacency Matrix
    W = W_t(args.client, args.p, 777)
    logger.info(f'Adjacency Matrix:\n{W}')

    # data loader 
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'MNIST':
        args.source = 7
        args.target = 1
        # logger.info('Train dataset loading...')
        # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # train_loaders, data_size = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger, args.small)
        # logger.info('Test dataset loading...')
        # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        # test_loaders, _ = create_federated_loaders(test_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, 32, logger)
        # logger.info('External dataset loading...')
        # external_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        # external_loaders, data_size_ex = create_federated_loaders(external_dataset, 1, args.malicious, 0, 0, args.p_stage, args.batch_size, logger)
        train_loaders, test_loaders, malicious, external_loaders = get_MNIST_loaders("/home/heemin/GFL/data", args.client, args.malicious, logger, 'IID', args.batch_size, args.p_stage, False)

    elif args.dataset == 'FMNIST':
        if args.LF_set == 1:
            args.source = 6
            args.target = 0
        elif args.LF_set == 2:
            args.source = 1
            args.target = 3
        elif args.LF_set == 3:
            args.source = 4
            args.target = 6
        # logger.info('Train dataset loading...')
        # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        # train_loaders, data_size = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, args.batch_size, logger, args.small)
        # logger.info('Test dataset loading...')
        # test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        # test_loaders, _ = create_federated_loaders(test_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, 32, logger)
        # logger.info('External dataset loading...')
        # external_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        # external_loaders, data_size_ex = create_federated_loaders(external_dataset, 1, 0, 0, 0, args.p_stage, args.batch_size, logger)
        train_loaders, test_loaders, malicious, external_loaders = get_FMNIST_loaders("/home/heemin/GFL/data", args.client, args.malicious, logger, 'IID', args.batch_size, args.p_stage, False)


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
        train_loaders, data_size = create_federated_loaders(train_dataset, args.client, args.malicious, args.source, args.target, args.p_stage, logger, args.small)
        
        logger.info('Test dataset loading...')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loaders, _ = create_federated_loaders(test_dataset, args.client, 0, args.source, args.target, args.p_stage, logger)

        logger.info('External dataset loading...')
        external_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        external_loaders, data_size_ex = create_federated_loaders(external_dataset, 1, 0, 0, 0, args.p_stage, logger)
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
        clients.append(copy.deepcopy(client(args, gan_list[i][0], gan_list[i][1], train_loaders[i], test_loaders[i], 
                                            DEVICE, logger, args.source, args.target, copy.deepcopy(CNN()), copy.deepcopy(CNN()), i, external_loaders[0]))) # , external_loaders[0]
    # run
    one_round_g_epoch = int(args.g_epoch / args.warm_up)
    init_W = copy.deepcopy(W)

    for i in range(args.client):
        img_dir = f'{save_path}/{args.project_name}/img/client{i+1}'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        model_dir = f'{save_path}/{args.project_name}/model/client{i+1}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        wrong_dir = f'{save_path}/{args.project_name}/wrong/client{i+1}'
        if not os.path.exists(wrong_dir):
            os.makedirs(wrong_dir)
        result_dir = f'{save_path}/{args.project_name}/result/client{i+1}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    if args.project == 'GFL':
        # Round start
        for Round in range(args.round):
            logger.info(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            logger.info(f'Round {Round+1} start')

            # GAN training & Save model            
            count = 0

            logger.info('GAN training & Save model')
            if not count == args.g_epoch:
                for i in range(args.client):
                    clients[i].gan_train(f'{save_path}/{args.project_name}/img/client{i+1}', one_round_g_epoch, Round)
                    count += one_round_g_epoch
                    clients[i].save_model(f'{save_path}/{args.project_name}/model/client{i+1}')
            else:
                for i in range(args.client):
                    one_round_g_epoch = args.local_epoch
                    count = 0
                    clients[i].gan_train(f'{save_path}/{args.project_name}/img/client{i+1}', one_round_g_epoch, Round)            

            # Make generated data loaders

            # local training
            logger.info('Local Classifier training')
            for i in range(args.client):
                clients[i].local_train()            

            # local test
            if ((Round + 1) >= args.warm_up) & ((Round + 1) % args.d_iter == 0):

                # gathering clients

                logger.info('Generate images')
                generated_loaders = []
                for i in tqdm(range(len(clients)), desc='Generate images'):
                    generated_loaders.append(copy.deepcopy(clients[i].generate_image(f'{save_path}/{args.project_name}/img/client{i+1}', Round)))
                logger.info('Local Classifier test')
                for i in range(args.client):
                    W[i] = copy.deepcopy(init_W[i])
                    W[i] = clients[i].local_test(generated_loaders, f'{save_path}/{args.project_name}/wrong/client{i+1}', Round, W[i], i)                

                # calculate W
                logger.info(f'Changed Adjacency Matrix: \n{W}')

            # classifier training
            logger.info('Classifier training')
            for i in range(args.client):
                clients[i].clf_train()
            
            # Aggregation
            logger.info('Aggregation')
            agg_models = []
            for i in range(args.client):
                agg_model = fed_avg(clients, W[i])
                agg_models.append(agg_model)
            else:
                for i in range(args.client):
                    clients[i].modelcopy(agg_models[i], i)
                    clients[i].modify_clf()
                    clients[i].aggregation_test(Round)            
            
            logger.info('external test')
            for i in range(args.client):
                clients[i].external_test(Round)

        for i in range(args.client):
            clients[i].make_plot(f'{save_path}/{args.project_name}/result/client{i+1}', 'acc_loss(local)', 'asr(local)', 'recall(local)')
            clients[i].make_plot_ex(f'{save_path}/{args.project_name}/result/client{i+1}', 'acc_loss(ex)', 'asr(ex)', 'recall(ex)')

    elif args.project == 'FL':
        for Round in range(args.round):
            logger.info(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            logger.info(f'Round {Round+1} start')
            

            # classifier training
            logger.info('Classifier training')
            for i in range(args.client):
                clients[i].clf_train()
            
            # Aggregation
            logger.info('Aggregation')
            agg_models = []
            for i in range(args.client):
                agg_model = fed_avg(clients, W[i])
                agg_models.append(agg_model)
            else:
                for i in range(args.client):
                    clients[i].modelcopy(agg_models[i], i)
                    clients[i].modify_clf()
                    clients[i].aggregation_test(Round)            
            
            logger.info('external test')
            for i in range(args.client):
                clients[i].external_test(Round)

        for i in range(args.client):
            clients[i].make_plot(f'{save_path}/{args.project_name}/result/client{i+1}', 'acc_loss(local)', 'asr(local)', 'recall(local)')
            clients[i].make_plot_ex(f'{save_path}/{args.project_name}/result/client{i+1}', 'acc_loss(ex)', 'asr(ex)', 'recall(ex)')
