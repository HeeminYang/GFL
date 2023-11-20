import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np

from model import CNN
from Data import get_FMNIST_loaders
from utils import get_logger
import matplotlib.pyplot as plt
import torchvision
import os

# 이미지 저장 함수
def save_images(img, filename):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)
    plt.close()

# 난수 시드 고정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 모델 초기화 및 가중치 확인 함수
def initialize_and_check_model(seed):
    set_seed(seed)
    model = CNN()
    print("Initial weights:", model.layer1[0].weight.data)
    return model

def train_and_check_model(model, train_loader, epochs, seed):
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()  # 모델을 학습 모드로 설정

    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 1 == 0:
                img_filename = f"epoch_{epoch+1}_batch_{i+1}.png"
                save_images(torchvision.utils.make_grid(images[:4]), img_filename)

                # 중간 가중치 출력 (예: 첫 번째 레이어의 가중치)
                print(f"Step [{i+1}/{len(train_loader)}], Layer1 weights: {model.layer1[0].weight.data}")

                # 배치 손실 출력
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss}")


# 메인 실행 함수
def main():
    seed = 42
    epochs = 1  # 일단 한 epoch만 실행
    model = initialize_and_check_model(seed)

    # 학습 데이터 로딩 함수
    log_path = f'test.log'
    logger = get_logger(log_path)
    train_loaders, test_loaders, malicious, external_loaders = get_FMNIST_loaders("/home/heemin/GFL/data", 10, 1, logger, 'IID', 128, 1.0, False)

    train_loader = train_loaders[0]
    train_and_check_model(model, train_loader, epochs, seed)

main()
