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
        client_indices = [[] for _ in range(num_clients)]

        # 각 레이블에 대해 IID 분포를 유지하며 인덱스를 분배합니다.
        for label_indices in all_indices.values():
            random.shuffle(label_indices)
            size = len(label_indices) // num_clients
            if small:
                size = size // 5
            remain = len(label_indices) % num_clients
            start = 0

            for client in range(num_clients):
                end = start + size + (1 if client < remain else 0)
                client_indices[client].extend(label_indices[start:end])
                start = end

        for client in range(num_clients):
            random.shuffle(client_indices[client])

        return client_indices, size

    # 악성 클라이언트의 데이터 레이블을 변경합니다.
    def label_flipping(dataset, indices, source_label, target_label, poisoning_stage, both, label_count):
        flip_count = 0
        reverse_flip_count = 0  # 타겟 레이블에서 소스 레이블로 변경된 데이터 수
        
        target = int(label_count[target_label] * poisoning_stage)
        print(target, label_count[target_label], poisoning_stage)

        for idx in indices:
            if flip_count < target:
                _, label = dataset[idx]
                if label == source_label:
                    dataset.targets[idx] = target_label
                    flip_count += 1
            if both and (reverse_flip_count < target):
                _, label = dataset[idx]
                if label == target_label:
                    dataset.targets[idx] = source_label
                    reverse_flip_count += 1
            
        return dataset, flip_count, reverse_flip_count

    # 데이터셋 분할
    subsets_indices, size = iid_partition(dataset, num_clients)
    subsets = [Subset(dataset, indices) for indices in subsets_indices]

    # 로깅: 각 클라이언트의 데이터 분포
    if logger:
        for client_idx, indices in enumerate(subsets_indices):
            label_count = defaultdict(int)
            for idx in indices:
                label = dataset[idx][1]
                label_count[label] += 1

            # 레이블의 개수별로 정렬합니다.
            sorted_label_count = OrderedDict(sorted(label_count.items(), key=lambda item: item[0]))
            logger.info(f" Client {client_idx} data distribution: {dict(sorted_label_count)}")

    # 악성 클라이언트 선택 및 레이블 변경
    # malicious_clients = random.sample(range(num_clients), num_malicious)
    malicious_clients = list(range(num_clients - num_malicious, num_clients))
    if malicious_clients:
        for malicious in malicious_clients:
            dataset, flip_count, reverse_flip_count = label_flipping(dataset, subsets_indices[malicious], source_label, target_label, poisoning_stage, both, label_count)

            if logger:
                # 어떤 레이블이 어떻게 바뀌었는지 로깅합니다.
                logger.info(f" Malicious client {malicious} flipped {flip_count} labels from {source_label} to {target_label}")
                if both:
                    logger.info(f" Malicious client {malicious} flipped {reverse_flip_count} labels from {target_label} to {source_label}")

                # 악성 클라이언트의 최종 데이터 분포를 로깅합니다.
                final_label_count = defaultdict(int)
                for idx in subsets_indices[malicious]:
                    label = dataset[idx][1]
                    final_label_count[label] += 1

                # 레이블 값을 기준으로 오름차순 정렬합니다.
                sorted_final_label_count = OrderedDict(sorted(final_label_count.items(), key=lambda item: item[0]))
                logger.info(f" Malicious client {malicious} final data distribution: {dict(sorted_final_label_count)}")

    # 각 클라이언트의 DataLoader 생성
    data_loaders = []
    for subset_indices in subsets_indices:
        sampler = SubsetRandomSampler(subset_indices)
        data_loaders.append(DataLoader(dataset, num_workers=2, batch_size=batch_size, sampler=sampler))

    return data_loaders, size
