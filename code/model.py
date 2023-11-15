import torch.nn as nn
import torch.nn.functional as F

# simple 3 layer CNN for MNISt
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x, feature_extract=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        if feature_extract:
            return out  # 특징 추출 시 여기서 반환
        out = self.dropout(out)
        out = self.fc(out)
        return out
  
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()  # 2D 이미지를 1D 벡터로 평탄화합니다.
        
        # 숨겨진 레이어 정의
        self.hidden1 = nn.Linear(28 * 28, 256)  # 입력층: 입력 차원은 28x28, 출력은 256
        self.hidden2 = nn.Linear(256, 128)      # 은닉층: 입력 차원은 256, 출력은 128
        self.hidden3 = nn.Linear(128, 64)       # 추가 은닉층: 입력 차원은 128, 출력은 64

        # 출력 레이어 정의
        self.output_layer = nn.Linear(64, 10)   # 출력층: 입력 차원은 64, 출력은 클래스 수인 10

    def forward(self, x):
        # 데이터 평탄화
        x = self.flatten(x)

        # 숨겨진 레이어들을 통과하면서 활성화 함수(예: ReLU) 적용 
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))

        # 출력 레이어를 통과하여 최종 결과 얻기
        x = self.output_layer(x)
        return x

class Cifar10CNN(nn.Module):

    def __init__(self):
        super(Cifar10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = F.softmax(self.fc2(x))

        return x