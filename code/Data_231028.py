import torch
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
import numpy as np
import random
from collections import defaultdict
from collections import OrderedDict

def create_federated_loaders(dataset, num_clients, num_malicious, source_label, target_label, poisoning_stage, batch_size=32, logger=None, small=False, both=True):
    # 데이터셋을 IID 분포로 분할합니다.
    def iid_partition(dataset, num_clients):
        # 모든 레이블에 대해 인덱스를 가져옵니다.
        all_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            all_indices[label].append(idx)

        # 클라이언트별로 분할된 인덱스
        trainset_indices = [[] for _ in range(num_clients)]
        testset_indices = [[] for _ in range(num_clients)]

        # 각 레이블에 대해 IID 분포를 유지하며 인덱스를 분배합니다.
        for label_indices in all_indices.values():
            random.shuffle(label_indices)
            size = len(label_indices) // num_clients
            if small:
                size = int(size * 0.1)
            train_size = int(size * 0.8)
            test_size = size - train_size
            remain = len(label_indices) % num_clients
            start = 0
            
            # make trainset
            for client in range(num_clients):
                end = start + train_size + (1 if client < remain else 0)
                trainset_indices[client].extend(label_indices[start:end])
                start = end
            
            # make testset
            for client in range(num_clients):
                testset_indices[client].extend(label_indices[start:start+test_size])
                start += test_size

        for client in range(num_clients):
            random.shuffle(trainset_indices[client])

        return trainset_indices, testset_indices, test_size

    # 악성 클라이언트의 데이터 레이블을 변경합니다.
    def label_flipping(dataset, indices, source_label, target_label, poisoning_stage, both, label_count):
        flip_count = 0
        reverse_flip_count = 0
        changed_to_target = set()

        target = int(label_count[target_label] * poisoning_stage)

        for idx in indices:
            # 소스 레이블에서 타겟 레이블로 변경
            if (flip_count < target) and (dataset.targets[idx] == source_label):
                dataset.targets[idx] = target_label
                flip_count += 1
                changed_to_target.add(idx)

            # both가 True이고, 타겟 레이블에서 소스 레이블로 변경
            elif both and (reverse_flip_count < target) and (dataset.targets[idx] == target_label) and (idx not in changed_to_target):
                dataset.targets[idx] = source_label
                reverse_flip_count += 1

            # 두 변경 작업 모두 완료되었는지 확인
            if (flip_count == target) and (not both or (reverse_flip_count == target)):
                break

        return dataset, flip_count, reverse_flip_count

    # 데이터셋 분할
    train_subsets_indices, test_subset_indices, size = iid_partition(dataset, num_clients)

    # 로깅: 각 클라이언트의 데이터 분포
    if logger:
        for client_idx, (train_indices, test_indices) in enumerate(zip(train_subsets_indices, test_subset_indices)):
            train_label_count = defaultdict(int)
            for idx in train_indices:
                label = dataset[idx][1]
                train_label_count[label] += 1
            
            test_label_count = defaultdict(int)
            for idx in test_indices:
                label = dataset[idx][1]
                test_label_count[label] += 1

            # 레이블의 개수별로 정렬합니다.
            sorted_train_label_count = OrderedDict(sorted(train_label_count.items(), key=lambda item: item[0]))
            sorted_test_label_count = OrderedDict(sorted(test_label_count.items(), key=lambda item: item[0]))
            logger.info(f" Client {client_idx} train dataset distribution:\n {dict(sorted_train_label_count)}")
            logger.info(f" Client {client_idx} test dataset distribution:\n {dict(sorted_test_label_count)}")

    # 악성 클라이언트 선택 및 레이블 변경
    # malicious_clients = random.sample(range(num_clients), num_malicious)
    malicious_clients = list(range(num_clients - num_malicious, num_clients))
    logger.info(f" Malicious clients: {malicious_clients}")
    if malicious_clients:
        for malicious in malicious_clients:
            dataset, train_flip_count, train_reverse_flip_count = label_flipping(dataset, train_subsets_indices[malicious], source_label, target_label, poisoning_stage, both, train_label_count)
            dataset, test_flip_count, test_reverse_flip_count = label_flipping(dataset, test_subset_indices[malicious], source_label, target_label, poisoning_stage, both, test_label_count)

            if logger:
                # 어떤 레이블이 어떻게 바뀌었는지 로깅합니다.
                logger.info(f" Malicious client {malicious} trainset flipped {train_flip_count} labels from {source_label} to {target_label}")
                logger.info(f" Malicious client {malicious} testset flipped {test_flip_count} labels from {source_label} to {target_label}")
                if both:
                    logger.info(f" Malicious client {malicious} trainset flipped {train_reverse_flip_count} labels from {target_label} to {source_label}")
                    logger.info(f" Malicious client {malicious} testset flipped {test_reverse_flip_count} labels from {target_label} to {source_label}")

                # 악성 클라이언트의 최종 트레인셋 데이터 분포를 로깅합니다.
                final_label_count = defaultdict(int)
                for idx in train_subsets_indices[malicious]:
                    label = dataset[idx][1]
                    final_label_count[label] += 1

                # 악성 클라이언트의 최종 테스트셋 데이터 분포를 로깅합니다.
                final_test_label_count = defaultdict(int)
                for idx in test_subset_indices[malicious]:
                    label = dataset[idx][1]
                    final_test_label_count[label] += 1

                # 레이블 값을 기준으로 오름차순 정렬합니다.
                sorted_final_label_count = OrderedDict(sorted(final_label_count.items(), key=lambda item: item[0]))
                sorted_final_test_label_count = OrderedDict(sorted(final_test_label_count.items(), key=lambda item: item[0]))
                logger.info(f" Malicious client {malicious} trainset final data distribution: \n{dict(sorted_final_label_count)}")
                logger.info(f" Malicious client {malicious} testset final data distribution: \n{dict(sorted_final_test_label_count)}")

    # 각 클라이언트의 DataLoader 생성
    train_data_loaders = []
    for subset_indices in train_subsets_indices:
        sampler = SubsetRandomSampler(subset_indices)
        train_data_loaders.append(DataLoader(dataset, num_workers=2, batch_size=batch_size, sampler=sampler))
    
    test_data_loaders = []
    for subset_indices in test_subset_indices:
        sampler = SubsetRandomSampler(subset_indices)
        test_data_loaders.append(DataLoader(dataset, num_workers=2, batch_size=batch_size, sampler=sampler))
    
    return train_data_loaders, test_data_loaders, size

def create_federated_loaders_ex(dataset, num_clients=1, batch_size=32, logger=None, small=False):
    # 데이터셋을 IID 분포로 분할합니다.
    def iid_partition(dataset, num_clients):
        # 모든 레이블에 대해 인덱스를 가져옵니다.
        all_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            all_indices[label].append(idx)

        # 클라이언트별로 분할된 인덱스
        trainset_indices = [[] for _ in range(num_clients)]

        # 각 레이블에 대해 IID 분포를 유지하며 인덱스를 분배합니다.
        for label_indices in all_indices.values():
            random.shuffle(label_indices)
            size = len(label_indices) // num_clients
            if small:
                size = int(size * 0.1)
            remain = len(label_indices) % num_clients
            start = 0
            
            # make trainset
            for client in range(num_clients):
                end = start + size + (1 if client < remain else 0)
                trainset_indices[client].extend(label_indices[start:end])
                start = end

        for client in range(num_clients):
            random.shuffle(trainset_indices[client])

        return trainset_indices, size

    # 데이터셋 분할
    train_subsets_indices, size = iid_partition(dataset, num_clients)

    # 로깅: 각 클라이언트의 데이터 분포
    if logger:
        for client_idx, train_indices in enumerate(train_subsets_indices):
            train_label_count = defaultdict(int)
            for idx in train_indices:
                label = dataset[idx][1]
                train_label_count[label] += 1
            
            # 레이블의 개수별로 정렬합니다.
            sorted_train_label_count = OrderedDict(sorted(train_label_count.items(), key=lambda item: item[0]))
            logger.info(f" Client {client_idx} train dataset distribution:\n {dict(sorted_train_label_count)}")

    # 각 클라이언트의 DataLoader 생성
    train_data_loaders = []
    for subset_indices in train_subsets_indices:
        sampler = SubsetRandomSampler(subset_indices)
        train_data_loaders.append(DataLoader(dataset, num_workers=2, batch_size=batch_size, sampler=sampler))

    return train_data_loaders, size