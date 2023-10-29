import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
def adjust_dataset_length(dataset, batch_size):
    """
    Adjusts the dataset length to be a multiple of batch_size.
    """
    excess = len(dataset) % batch_size
    if excess != 0:
        # Calculate number of samples to remove
        num_to_remove = len(dataset) - excess
        indices = list(range(len(dataset)))
        adjusted_indices = indices[:num_to_remove]
        dataset = Subset(dataset, adjusted_indices)
    return dataset


def MNIST_split_loader(logger, mnist_dataset, subset_indices, batch_size, p_stage=1.0, flip=False):
    # Filter out data only with label 1 and 7 for flipping
    if flip:
        # label 1 indices
        indices1_for_flip = [subset_indices[i] for i in range(len(subset_indices)) if mnist_dataset.targets[subset_indices[i]] == 1]
        samples1_flip = int(len(indices1_for_flip) * p_stage)

        # label 7 indices
        indices7_for_flip = [subset_indices[i] for i in range(len(subset_indices)) if mnist_dataset.targets[subset_indices[i]] == 7]
        samples7_flip = int(len(indices7_for_flip) * p_stage)
        
        # Randomly select samples to flip
        indices1_for_flip = np.random.choice(indices1_for_flip, samples1_flip, replace=False)
        indices7_for_flip = np.random.choice(indices7_for_flip, samples7_flip, replace=False)

        # Flip the labels
        for idx in indices1_for_flip:
            mnist_dataset.targets[idx] = 7
        for idx in indices7_for_flip:
            mnist_dataset.targets[idx] = 1
        
        # log
        logger.info(f'Flipped {samples1_flip} label 1 samples and {samples7_flip} label 7 samples.')

        # for idx in tqdm(indices_for_flip, desc='Flipping'):
        #     if mnist_dataset.targets[idx] == 7:
        #         mnist_dataset.targets[idx] = 1 
        #     elif mnist_dataset.targets[idx] == 1:
        #         mnist_dataset.targets[idx] = 7

    # Splitting the data
    train_indices, test_indices = train_test_split(subset_indices, test_size=0.2)

    # Create Datasets
    train_dataset = Subset(mnist_dataset, train_indices)
    test_dataset = Subset(mnist_dataset, test_indices)

    # Adjust the length of train_dataset and test_dataset to be a multiple of batch_size
    train_dataset = adjust_dataset_length(train_dataset, batch_size)
    test_dataset = adjust_dataset_length(test_dataset, batch_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def get_MNIST_loaders(data_dir, num_client, mali_client, logger, distribution, batch_size=32, p_stage=1.0, central_test=False):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                               ])
    mnist_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    loaders_train = []
    loaders_test = []
    malicious_info = []
    central_trainset = []
    central_testset = []
    
    indices = list(range(len(mnist_dataset)))
    targets = [label for _, label in mnist_dataset]
    
    skf = StratifiedKFold(n_splits=num_client)
    logger.info(f'data distribution: {distribution}')
    logger.info('Label distribution in the Clients:')

    # Create loaders for each client
    for i, (_, train_index) in enumerate(skf.split(indices, targets)):
        if len(loaders_train) < num_client - mali_client:
            train_loader, test_loader, train_dataset, test_dataset  = MNIST_split_loader(logger, mnist_dataset, train_index, batch_size, p_stage)
            loaders_train.append(train_loader)
            loaders_test.append(test_loader)
            malicious_info.append(False)
            central_trainset.append(train_dataset)
            central_testset.append(test_dataset)
        else:
            logger.info(f'Client {len(loaders_train)+1} is malicious.')
            train_loader, test_loader, train_dataset, test_dataset  = MNIST_split_loader(logger, mnist_dataset, train_index, batch_size, p_stage, flip=True)
            loaders_train.append(train_loader)
            loaders_test.append(test_loader)
            malicious_info.append(True)
            central_trainset.append(train_dataset)

        # Calculate label distribution in the loaders
        train_labels = [label for _, label in train_loader.dataset]
        test_labels = [label for _, label in test_loader.dataset]
        label_counts = pd.Series(train_labels).value_counts()
        label_info = ', '.join([f"{label}: {count}" for label, count in label_counts.items()])
        logger.info(f"Client {i+1} Label train Distribution: \n{label_info}")
        test_label_counts = pd.Series(test_labels).value_counts()
        test_label_info = ', '.join([f"{label}: {count}" for label, count in test_label_counts.items()])
        logger.info(f"Client {i+1} Label test Distribution: \n{test_label_info}")
    
    # Central scenario
    central_train_dataset = ConcatDataset(central_trainset)
    central_test_dataset = ConcatDataset(central_testset)
    central_train_loader = DataLoader(central_train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    central_test_loader = DataLoader(central_test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    # Test set loader
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(mnist_test, num_workers=2, batch_size=batch_size, shuffle=False)
    train_labels = [label for _, label in test_loader.dataset]
    label_counts = pd.Series(train_labels).value_counts()
    label_info = ', '.join([f"{label}: {count}" for label, count in label_counts.items()])
    logger.info(f"External Set Label Distribution: \n{label_info}")

    if central_test:
        # Calculate label distribution in the loaders
        train_labels = [label for _, label in central_train_loader.dataset]
        test_labels = [label for _, label in central_test_loader.dataset]
        train_counts = pd.Series(train_labels).value_counts()
        test_counts = pd.Series(test_labels).value_counts()
        logger.info(f"Central Scenario Train Label Distribution: \n0: {train_counts.get(0, 0)}, 1: {train_counts.get(1, 0)}")
        logger.info(f"Central Scenario Test Label Distribution: \n0: {test_counts.get(0, 0)}, 1: {test_counts.get(1, 0)}")
        return [central_train_loader], [central_test_loader], malicious_info, test_loader
    else:
        return loaders_train, loaders_test, malicious_info, [test_loader]



def FMNIST_split_loader(logger, mnist_dataset, subset_indices, batch_size, p_stage=1.0, flip=False):
    # Filter out data only with label 1 and 7 for flipping
    if flip:
        # label 1 indices
        indices1_for_flip = [subset_indices[i] for i in range(len(subset_indices)) if mnist_dataset.targets[subset_indices[i]] == 6]
        samples1_flip = int(len(indices1_for_flip) * p_stage)

        # label 7 indices
        indices7_for_flip = [subset_indices[i] for i in range(len(subset_indices)) if mnist_dataset.targets[subset_indices[i]] == 0]
        samples7_flip = int(len(indices7_for_flip) * p_stage)
        
        # Randomly select samples to flip
        indices1_for_flip = np.random.choice(indices1_for_flip, samples1_flip, replace=False)
        indices7_for_flip = np.random.choice(indices7_for_flip, samples7_flip, replace=False)

        # Flip the labels
        for idx in indices1_for_flip:
            mnist_dataset.targets[idx] = 0
        for idx in indices7_for_flip:
            mnist_dataset.targets[idx] = 6
        
        # log
        logger.info(f'Flipped {samples1_flip} label 6 samples and {samples7_flip} label 0 samples.')

        # for idx in tqdm(indices_for_flip, desc='Flipping'):
        #     if mnist_dataset.targets[idx] == 7:
        #         mnist_dataset.targets[idx] = 1 
        #     elif mnist_dataset.targets[idx] == 1:
        #         mnist_dataset.targets[idx] = 7

    # Splitting the data
    train_indices, test_indices = train_test_split(subset_indices, test_size=0.2)

    # Create Datasets
    train_dataset = Subset(mnist_dataset, train_indices)
    test_dataset = Subset(mnist_dataset, test_indices)

    # Adjust the length of train_dataset and test_dataset to be a multiple of batch_size
    train_dataset = adjust_dataset_length(train_dataset, batch_size)
    test_dataset = adjust_dataset_length(test_dataset, batch_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def get_FMNIST_loaders(data_dir, num_client, mali_client, logger, distribution, batch_size=32, p_stage=1.0, central_test=False):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                               ])
    mnist_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)

    loaders_train = []
    loaders_test = []
    malicious_info = []
    central_trainset = []
    central_testset = []
    
    indices = list(range(len(mnist_dataset)))
    targets = [label for _, label in mnist_dataset]
    
    skf = StratifiedKFold(n_splits=num_client)
    logger.info(f'data distribution: {distribution}')
    logger.info('Label distribution in the Clients:')

    # Create loaders for each client
    for i, (_, train_index) in enumerate(skf.split(indices, targets)):
        if len(loaders_train) < num_client - mali_client:
            train_loader, test_loader, train_dataset, test_dataset  = FMNIST_split_loader(logger, mnist_dataset, train_index, batch_size, p_stage)
            loaders_train.append(train_loader)
            loaders_test.append(test_loader)
            malicious_info.append(False)
            central_trainset.append(train_dataset)
            central_testset.append(test_dataset)
        else:
            logger.info(f'Client {len(loaders_train)+1} is malicious.')
            train_loader, test_loader, train_dataset, test_dataset  = FMNIST_split_loader(logger, mnist_dataset, train_index, batch_size, p_stage, flip=True)
            loaders_train.append(train_loader)
            loaders_test.append(test_loader)
            malicious_info.append(True)
            central_trainset.append(train_dataset)

        # Calculate label distribution in the loaders
        train_labels = [label for _, label in train_loader.dataset]
        test_labels = [label for _, label in test_loader.dataset]
        label_counts = pd.Series(train_labels).value_counts()
        label_info = ', '.join([f"{label}: {count}" for label, count in label_counts.items()])
        logger.info(f"Client {i+1} Label train Distribution: \n{label_info}")
        test_label_counts = pd.Series(test_labels).value_counts()
        test_label_info = ', '.join([f"{label}: {count}" for label, count in test_label_counts.items()])
        logger.info(f"Client {i+1} Label test Distribution: \n{test_label_info}")
    
    # Central scenario
    central_train_dataset = ConcatDataset(central_trainset)
    central_test_dataset = ConcatDataset(central_testset)
    central_train_loader = DataLoader(central_train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    central_test_loader = DataLoader(central_test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    # Test set loader
    mnist_test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(mnist_test, num_workers=2, batch_size=batch_size, shuffle=False)
    train_labels = [label for _, label in test_loader.dataset]
    label_counts = pd.Series(train_labels).value_counts()
    label_info = ', '.join([f"{label}: {count}" for label, count in label_counts.items()])
    logger.info(f"External Set Label Distribution: \n{label_info}")

    if central_test:
        # Calculate label distribution in the loaders
        train_labels = [label for _, label in central_train_loader.dataset]
        test_labels = [label for _, label in central_test_loader.dataset]
        train_counts = pd.Series(train_labels).value_counts()
        test_counts = pd.Series(test_labels).value_counts()
        logger.info(f"Central Scenario Train Label Distribution: \n0: {train_counts.get(0, 0)}, 1: {train_counts.get(1, 0)}")
        logger.info(f"Central Scenario Test Label Distribution: \n0: {test_counts.get(0, 0)}, 1: {test_counts.get(1, 0)}")
        return [central_train_loader], [central_test_loader], malicious_info, test_loader
    else:
        return loaders_train, loaders_test, malicious_info, [test_loader]