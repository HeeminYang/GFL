from utils import weights_init

import os
import copy

from math import floor
from math import ceil
from scipy.linalg import sqrtm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt

# client recieves args, generator, discriminator, and data loaders for training and testing
class client:
    def __init__(self, args, generator, discriminator, train_loader, test_loader,  device, logger, source, target, classifier=None, name=None, external_loader=None):
        self.args = args
        self.netG = generator
        self.netD = discriminator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.external_loader = external_loader
        self.logger = logger
        self.name = name
        self.top_performance = [0 for _ in range(4)] # accuracy, accuracy_round, loss, loss_round
        self.top_performance_ex = [0 for _ in range(4)] # accuracy, accuracy_round, loss, loss_round

        self.device = device

        self.threshold = int(args.g_img_num * args.threshold)

        self.Gen_switch = True

        # flipping label
        self.source = source
        self.target = target

        # test results
        self.accuray = []
        self.loss = []
        self.flipped_recall1 = []
        self.flipped_recall2 = []
        self.asr1 = []
        self.asr2 = []

        # test results
        self.accuray_ex = []
        self.loss_ex = []
        self.flipped_recall1_ex = []
        self.flipped_recall2_ex = []
        self.asr1_ex = []
        self.asr2_ex = []

        # GAN hyperparameters
        self.netG.to(device)
        self.netD.to(device)
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

        # FL Task hyperparameters
        self.classifier = classifier
        self.classifier.to(device)
        self.clf_criterion = nn.CrossEntropyLoss()
        self.clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=1e-5)
        self.aggregated_model = copy.deepcopy(classifier)
        self.aggregated_model.to(device)

        # # local classifier
        # self.local_model = local_model
        # self.local_model.to(device)
        # self.local_criterion = nn.CrossEntropyLoss()
        # self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # We calculate Binary cross entropy loss
        self.criterion = nn.BCELoss()
        # Adam optimizer for generator
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.args.Adam_lr, betas=(self.args.Adam_beta1, 0.999))
        # Adam optimizer for discriminator
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.args.Adam_lr, betas=(self.args.Adam_beta1, 0.999))

        # # labels for training images x for Discriminator training
        self.labels_real = torch.ones((self.args.batch_size, 1)).to(device)
        # labels for generated images G(z) for Discriminator training
        self.labels_fake = torch.zeros((self.args.batch_size, 1)).to(device)
        # Fix noise for testing generator and visualization
        self.z_test = torch.randn(self.args.num_classes*self.args.g_img_num, self.args.size_z).to(device)

        # convert labels to onehot encoding
        self.onehot = torch.zeros(self.args.num_classes, self.args.num_classes).scatter_(1, torch.arange(self.args.num_classes).view(self.args.num_classes,1), 1)
        # reshape labels to image size, with number of labels as channel
        self.fill = torch.zeros([10, 10, self.args.img_size, self.args.img_size])
        #channel corresponding to label will be set one and all other zeros
        for i in range(10):
            self.fill[i, i, :, :] = 1
        # create labels for testing generator
        self.test_y = torch.tensor(list(range(self.args.num_classes))*self.args.g_img_num).type(torch.LongTensor)
        # convert to one hot encoding
        self.test_Gy = self.onehot[self.test_y].to(device)

    def save_results_ex(self, accuracy, loss, wrong_count, total_count,log,Round):
        self.accuray_ex.append(accuracy)
        self.loss_ex.append(loss)
        self.asr1_ex.append((Counter(wrong_count[self.args.source])[self.args.target])/total_count[self.args.source])
        self.asr2_ex.append((Counter(wrong_count[self.args.target])[self.args.source])/total_count[self.args.target])
        source_wrong = len(wrong_count[self.args.source])
        self.flipped_recall1_ex.append((total_count[self.source]-source_wrong)/total_count[self.source])
        target_wrong = len(wrong_count[self.args.target])
        self.flipped_recall2_ex.append((total_count[self.target]-target_wrong)/total_count[self.target])
        log = ' Client{:} Accuracy: {:.5f} Loss: {:.5f} ASR1: {:.5f} ASR2: {:.5f} Recall1: {:.5f} Recall2: {:.5f}'.format((self.name+1), accuracy, loss, self.asr1_ex[-1], self.asr2_ex[-1], self.flipped_recall1_ex[-1], self.flipped_recall2_ex[-1]) + log
        if accuracy > self.top_performance_ex[0]:
            self.top_performance_ex[0] = accuracy
            self.top_performance_ex[1] = Round
        if loss < self.top_performance_ex[2]:
            self.top_performance_ex[2] = loss
            self.top_performance_ex[3] = Round
        self.logger.info(log)

    def save_results(self, accuracy, loss, wrong_count, total_count, log,Round):
        self.accuray.append(accuracy)
        self.loss.append(loss)
        self.asr1.append((Counter(wrong_count[self.args.source])[self.args.target])/total_count[self.args.source])
        self.asr2.append((Counter(wrong_count[self.args.target])[self.args.source])/total_count[self.args.target])
        source_wrong = len(wrong_count[self.args.source])
        self.flipped_recall1.append((total_count[self.source]-source_wrong)/total_count[self.source])
        target_wrong = len(wrong_count[self.args.target])
        self.flipped_recall2.append((total_count[self.target]-target_wrong)/total_count[self.target])
        log = ' Client{:} Accuracy: {:.5f} Loss: {:.5f} ASR1: {:.5f} ASR2: {:.5f} Recall1: {:.5f} Recall2: {:.5f}'.format((self.name+1), accuracy, loss, self.asr1[-1], self.asr2[-1], self.flipped_recall1[-1], self.flipped_recall2[-1]) + log
        if accuracy > self.top_performance[0]:
            self.top_performance[0] = accuracy
            self.top_performance[1] = Round
        if loss < self.top_performance[2]:
            self.top_performance[2] = loss
            self.top_performance[3] = Round
        self.logger.info(log)

    def plot_accuracy_and_loss(self, accuracies, losses, save_path, filename, top_performance):
        rounds = list(range(1, len(accuracies) + 1))
        # 새로운 그림(figure)를 생성합니다.
        fig, ax1 = plt.subplots()

        # ax1은 왼쪽 y축을 담당하며, 정확도에 대한 플롯을 생성합니다.
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        line1, = ax1.plot(rounds, accuracies, color='tab:blue', label='Accuracy')  # legend를 위한 label 추가
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # ax2는 ax1과 동일한 x축을 공유하지만, 오른쪽 y축을 담당합니다. 손실에 대한 플롯을 생성합니다.
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:red')
        line2, = ax2.plot(rounds, losses, color='tab:red', label='Loss')  # legend를 위한 label 추가
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # 최적의 정확도 및 손실 값을 주석으로 추가합니다.
        best_accuracy = top_performance[0]
        best_accuracy_round = top_performance[1]
        best_loss = top_performance[2]
        best_loss_round = top_performance[3]

        # 이 값을 그래프에 주석으로 추가합니다.
        ax1.annotate(f'Best Acc: {best_accuracy:.3f}\nAt Round: {best_accuracy_round}', 
                    xy=(best_accuracy_round, best_accuracy), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    va='center', 
                    color='midnightblue', 
                    backgroundcolor='white', 
                    arrowprops=dict(arrowstyle='->', color='midnightblue'),
                    zorder=1)  # 선 아래에 주석을 놓습니다.

        ax2.annotate(f'Best Loss: {best_loss:.3f}\nAt Round: {best_loss_round}', 
                    xy=(best_loss_round, best_loss), 
                    textcoords="offset points", 
                    xytext=(0,-30), 
                    ha='center', 
                    va='center', 
                    color='maroon', 
                    backgroundcolor='white', 
                    arrowprops=dict(arrowstyle='->', color='maroon'),
                    zorder=1)  # 선 아래에 주석을 놓습니다.
        # 마지막 정확도 및 손실 값을 주석으로 추가합니다.
        last_accuracy = accuracies[-1]
        last_loss = losses[-1]
        ax1.annotate(f'{last_accuracy:.3f}', xy=(rounds[-1], last_accuracy), textcoords="offset points", xytext=(0,10), ha='center', color='midnightblue')
        ax2.annotate(f'{last_loss:.3f}', xy=(rounds[-1], last_loss), textcoords="offset points", xytext=(0,-20), ha='center', color='maroon')

        # 그래프가 겹치지 않도록 레이아웃을 조정합니다.
        fig.tight_layout()

        # legend 설정. 각 Line에 대한 label을 이용합니다.
        fig.legend(handles=[line1, line2], loc='center right')

        # 그래프를 파일로 저장합니다.
        plt.savefig(f"{save_path}/{filename}.png")

    def plot_asr(self, asr1, asr2, save_path, filename):
        rounds = list(range(1, len(asr1) + 1))
        # 새로운 그림(figure)를 생성합니다.
        fig, ax1 = plt.subplots()

        # ax1은 왼쪽 y축을 담당하며, ASR 값에 대한 플롯을 생성합니다.
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('ASR', color='tab:blue')
        ax1.plot(rounds, asr1, color='tab:blue', label=f'ASR {self.source}')
        ax1.plot(rounds, asr2, color='tab:orange', label=f'ASR {self.target}')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 마지막 ASR 값들을 주석으로 추가합니다.
        last_asr1 = asr1[-1]
        last_asr2 = asr2[-1]
        ax1.annotate(f'{last_asr1:.3f}', xy=(rounds[-1], last_asr1), textcoords="offset points", xytext=(0,10), ha='center', color='midnightblue' )
        ax1.annotate(f'{last_asr2:.3f}', xy=(rounds[-1], last_asr2), textcoords="offset points", xytext=(0,-20), ha='center', color='saddlebrown')

        # 그래프가 겹치지 않도록 레이아웃을 조정합니다.
        fig.tight_layout()

        # 범례(legend)를 추가합니다.
        ax1.legend()

        # 그래프를 파일로 저장합니다.
        plt.savefig(f"{save_path}/{filename}.png")

    def plot_recall(self, flipped_recall1, flipped_recall2, save_path, filename):
        rounds = list(range(1, len(flipped_recall1) + 1))
        # 새로운 그림(figure)를 생성합니다.
        fig, ax1 = plt.subplots()

        # ax1은 왼쪽 y축을 담당하며, Recall 값에 대한 플롯을 생성합니다.
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Recall', color='tab:blue')
        line1, = ax1.plot(rounds, flipped_recall1, color='tab:blue', label=f'label {self.source}')  # Recall1에 대한 라인
        line2, = ax1.plot(rounds, flipped_recall2, color='tab:orange', label=f'label {self.target}')  # Recall2에 대한 라인
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # 마지막 Recall 값들을 주석으로 추가합니다.
        last_recall1 = flipped_recall1[-1]
        last_recall2 = flipped_recall2[-1]
        ax1.annotate(f'{last_recall1:.2f}', xy=(rounds[-1], last_recall1), textcoords="offset points", xytext=(0,10), ha='center', color='midnightblue')
        ax1.annotate(f'{last_recall2:.2f}', xy=(rounds[-1], last_recall2), textcoords="offset points", xytext=(0,-20), ha='center', color='saddlebrown')

        # 그래프가 겹치지 않도록 레이아웃을 조정합니다.
        fig.tight_layout()

        # legend 설정. 각 Line에 대한 label을 이용합니다.
        ax1.legend(handles=[line1, line2])

        # 그래프를 파일로 저장합니다.
        plt.savefig(f"{save_path}/{filename}.png")

    def make_plot(self, save_path, acc_name, asr_name, recall_name):
        self.plot_accuracy_and_loss(self.accuray, self.loss, save_path, acc_name, self.top_performance)
        self.plot_asr(self.asr1, self.asr2, save_path, asr_name)
        self.plot_recall(self.flipped_recall1, self.flipped_recall2, save_path, recall_name)

    def make_plot_ex(self, save_path, acc_name, asr_name, recall_name):
        self.plot_accuracy_and_loss(self.accuray_ex, self.loss_ex, save_path, acc_name, self.top_performance_ex)
        self.plot_asr(self.asr1_ex, self.asr2_ex, save_path, asr_name)
        self.plot_recall(self.flipped_recall1_ex, self.flipped_recall2_ex, save_path, recall_name)

    def modelcopy(self, model, rank):
        self.aggregated_model.load_state_dict(copy.deepcopy(model))
    
    def modify_clf(self):
        self.classifier.load_state_dict(copy.deepcopy(self.aggregated_model.state_dict()))
        self.classifier.to(self.device)

    def gan_train(self, save_path, Round):
        self.netG.train()

        # List of values, which will be used for plotting purpose
        D_losses = []
        G_losses = []
        Dx_values = []
        DGz_values = []

        # number of training steps done on discriminator
        step = 0
        for epoch in range(self.args.g_epoch):
            epoch_D_losses = []
            epoch_G_losses = []
            epoch_Dx = []
            epoch_DGz = []
        
            # iterate through data loader generator object
            for images, y_labels in self.train_loader:
                step += 1
                
                # 현재 배치의 크기를 가져옵니다.
                current_batch_size = images.size(0)

                # '진짜'와 '가짜' 레이블을 현재 배치 크기에 맞게 동적으로 생성합니다.
                labels_real = torch.ones((current_batch_size, 1)).to(self.device)

                ############################
                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # images will be send to gpu, if cuda available
                x = images.to(self.device)
                # preprocess labels for feeding as y input
                # D_y shape will be (batch_size, 10, 28, 28)
                D_y = self.fill[y_labels].to(self.device)
                # forward pass D(x)
                x_preds = self.netD(x, D_y)
                # calculate loss log(D(x))
                D_x_loss = self.criterion(x_preds, labels_real)

                # create latent vector z from normal distribution
                z = torch.randn(self.args.batch_size, self.args.size_z).to(self.device)
                # create random y labels for generator
                y_gen = (torch.rand(self.args.batch_size, 1)*self.args.num_classes).type(torch.LongTensor).squeeze()
                # convert genarator labels to onehot
                G_y = self.onehot[y_gen].to(self.device)
                # preprocess labels for feeding as y input in D
                # DG_y shape will be (batch_size, 10, 28, 28)
                DG_y = self.fill[y_gen].to(self.device)

                # generate image
                fake_image = self.netG(z, G_y)
                # calculate D(G(z)), fake or not
                z_preds = self.netD(fake_image.detach(), DG_y)
                # loss log(1 - D(G(z)))
                D_z_loss = self.criterion(z_preds, self.labels_fake)

                # total loss = log(D(x)) + log(1 - D(G(z)))
                D_loss = D_x_loss + D_z_loss

                # save values for plots
                epoch_D_losses.append(D_loss.item())
                epoch_Dx.append(x_preds.mean().item())

                # zero accumalted grads
                self.netD.zero_grad()
                # do backward pass
                D_loss.backward()
                # update discriminator model
                self.optimizerD.step()

                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################

                # if Ksteps of Discriminator training are done, update generator
                if step % self.args.Ksteps == 0:
                    # As we done one step of discriminator, again calculate D(G(z))
                    z_out = self.netD(fake_image, DG_y)
                    # loss log(D(G(z)))
                    G_loss = self.criterion(z_out, self.labels_real)
                    # save values for plots
                    epoch_DGz.append(z_out.mean().item())
                    epoch_G_losses.append(G_loss)

                    # zero accumalted grads
                    self.netG.zero_grad()
                    # do backward pass
                    G_loss.backward()
                    # update generator model
                    self.optimizerG.step()
            else:
                # calculate average value for one epoch
                D_losses.append(sum(epoch_D_losses)/len(epoch_D_losses))
                G_losses.append(sum(epoch_G_losses)/len(epoch_G_losses))
                Dx_values.append(sum(epoch_Dx)/len(epoch_Dx))
                DGz_values.append(sum(epoch_DGz)/len(epoch_DGz))

        # # Generating images after each epoch and saving
        # # set generator to evaluation mode
        # self.netG.eval()
        # with torch.no_grad():
        #     # forward pass of G and generated image
        #     fake_test = self.netG(self.z_test, self.test_Gy).cpu()
        #     # save images in grid of 10 * 10
        #     torchvision.utils.save_image(fake_test, f"{save_path}/{Round+1}_{epoch+1}.jpg", nrow=10, padding=0, normalize=True)
        # # set generator to training mode
        # self.netG.train()

        self.logger.info(f" Client{self.name+1}: Epoch {epoch+1}/{self.args.g_epoch} Discriminator Loss {D_losses[-1]:.5f} Generator Loss {G_losses[-1]:.5f}"
                + f" D(x) {Dx_values[-1]:.5f} D(G(x)) {DGz_values[-1]:.5f}")
        
    def generate_image(self, save_path, Round):
        
        self.netG.eval()
        with torch.no_grad():
            new_Z = torch.randn(self.args.num_classes*self.args.g_img_num, self.args.size_z).to(self.device)
            new_test_y = torch.tensor(list(range(self.args.num_classes))*self.args.g_img_num).type(torch.LongTensor)
            new_test_Gy = self.onehot[new_test_y].to(self.device)
            fake_image = self.netG(new_Z, new_test_Gy).cpu()
            torchvision.utils.save_image(fake_image, f"{save_path}/data_{Round+1}.jpg", nrow=10, padding=0, normalize=True)
        dataset = torch.utils.data.TensorDataset(fake_image, new_test_y)
        generated_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        self.generated_loader = generated_loader

    def save_model(self, save_path):
        torch.save(self.netG.state_dict(), f"{save_path}/generator.pth")
        torch.save(self.netD.state_dict(), f"{save_path}/discriminator.pth")
    
    def load_model(self, load_path):
        self.netG.load_state_dict(torch.load(f"{load_path}/generator.pth"))
        self.netD.load_state_dict(torch.load(f"{load_path}/discriminator.pth"))

    # only CNN classifier training (from model.py)
    def clf_train(self):
        
        self.classifier.train()

        # List of values, which will be used for plotting purpose
        losses = []
        accuracies = []

        for epoch in range(self.args.clf_epoch):
            epoch_losses = []
            epoch_accuracies = []
            for images, labels in self.train_loader:
                # images will be send to gpu, if cuda available
                x = images.to(self.device)
                # labels will be send to gpu, if cuda available
                y = labels.to(self.device)

                # forward pass
                preds = self.classifier(x)
                # calculate loss
                loss = self.clf_criterion(preds, y)

                # calculate accuracy
                correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()

                # save values for plots
                epoch_losses.append(loss.item())
                epoch_accuracies.append(correct)

                # zero accumalted grads
                self.classifier.zero_grad()
                # do backward pass
                loss.backward()
                # update classifier model
                self.clf_optimizer.step()
            else:
                # calculate average value for one epoch
                losses.append(sum(epoch_losses)/len(epoch_losses))
                accuracies.append(sum(epoch_accuracies)/len(epoch_accuracies))

        log = f" Client{self.name+1}:\n  Train Loss {losses[-1]:.5f} Accuracy {accuracies[-1]:.5f}"
        self.clf_test(log)
 
    # test generated images by CNN classifier
    def clf_test(self, log):
        
        self.classifier.eval()

        features = []
        losses = []
        accuracies = []

        for images, labels in self.test_loader:

            x = images.to(self.device)
            y = labels.to(self.device)

            # for fid
            feature = self.classifier(x, feature_extract=True)
            features.append(feature)

            preds = self.classifier(x)
            loss = self.clf_criterion(preds, y)

            correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()

            losses.append(loss.item())
            accuracies.append(correct)

        features = torch.cat(features, dim=0)
        self.mu_real = torch.mean(features, dim=0)
        self.sigma_real = torch.cov(features.t())

        log += f"\n  Test Loss {sum(losses)/len(losses):.5f} Accuracy {sum(accuracies)/len(accuracies):.5f}"
        self.logger.info(log)

        self.clf_acc = sum(accuracies)/len(accuracies)
        self.clf_loss = sum(losses)/len(losses)
     
    # test generated images by CNN classifier
    def local_gen_test(self, Round):
        
        self.classifier.eval()

        features = []
        losses = []
        accuracies = []

        for images, labels in self.generated_loader:

            x = images.to(self.device)
            y = labels.to(self.device)
            
            # for fid
            feature = self.classifier(x, feature_extract=True)
            features.append(feature)

            preds = self.classifier(x)
            loss = self.clf_criterion(preds, y)

            correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()

            losses.append(loss.item())
            accuracies.append(correct)
            
        features = torch.cat(features, dim=0)
        mu_gen = torch.mean(features, dim=0)
        sigma_gen = torch.cov(features.t())

        log = f" Client{self.name+1} Gen Test: Loss {sum(losses)/len(losses):.5f} Accuracy {sum(accuracies)/len(accuracies):.5f}"
        self.logger.info(log)

        self.gen_acc = sum(accuracies)/len(accuracies)
        self.gen_loss = sum(losses)/len(losses)

        fid = self.calculate_fid(self.mu_real, self.sigma_real, mu_gen, sigma_gen)
        self.logger.info(f" Client{self.name+1} FID: {fid:.5f}")

        if Round > (self.args.warm_up - 1):
            if (self.gen_acc >= (floor(self.clf_acc * 100)/100)) and (self.gen_loss <= (ceil(self.clf_acc * 100)/100)):
                self.Gen_switch = False
                self.logger.info(f" Client{self.name+1}: Generator training is stopped at Round {Round+1} with accuracy {self.gen_acc:.5f} and loss {self.gen_loss:.5f}")

    def calculate_fid(self, mu_real, sigma_real, mu_gen, sigma_gen):
        # torch.Tensor를 numpy.ndarray로 변환
        mu_real = mu_real.detach().cpu().numpy() if isinstance(mu_real, torch.Tensor) else mu_real
        mu_gen = mu_gen.detach().cpu().numpy() if isinstance(mu_gen, torch.Tensor) else mu_gen
        sigma_real = sigma_real.detach().cpu().numpy() if isinstance(sigma_real, torch.Tensor) else sigma_real
        sigma_gen = sigma_gen.detach().cpu().numpy() if isinstance(sigma_gen, torch.Tensor) else sigma_gen

        # 평균 벡터 간의 차이의 제곱
        ssdiff = np.sum((mu_real - mu_gen)**2.0)
        
        # 공분산 행렬의 제곱근
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        # 수치적 안정성을 위해 복소수 체크
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # FID 계산
        fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
        return fid


    # test generated images by CNN classifier
    def aggregation_test(self, Round, logging=True):
        
        self.classifier.eval()
        wrong_count = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
        total_count = {j: 0 for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label

        losses = []
        accuracies = []

        for images, labels in self.test_loader:
                
                x = images.to(self.device)
                y = labels.to(self.device)
    
                preds = self.classifier(x)
                loss = self.clf_criterion(preds, y)
    
                correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()
    
                losses.append(loss.item())
                accuracies.append(correct)
    
                # Check for misclassified images
                for img_idx, (true_label, pred_label) in enumerate(zip(y, torch.argmax(preds, dim=1))):
                    if true_label != pred_label:
                        # Store the predicted label for this true label
                        if true_label.item() not in wrong_count:
                            wrong_count[true_label.item()] = []
                            total_count[true_label.item()] = 0
                        wrong_count[true_label.item()].append(pred_label.item())
                        total_count[true_label.item()] += 1
                    else:
                        if true_label.item() not in total_count:
                            total_count[true_label.item()] = 0
                        total_count[true_label.item()] += 1
        if logging:
            log = f"\n   Summary of misclassifications:"
            for true_label, pred_labels in wrong_count.items():
                counter = Counter(pred_labels)
                counter_dict = dict(counter)
                oreder_dict = OrderedDict(sorted(counter_dict.items(), key=lambda item: item[0]))
                log += f"\n   True label {true_label} was misclassified as: {oreder_dict}"
        else:
            log = ''
            for true_label, pred_labels in wrong_count.items():
                counter = Counter(pred_labels)
                counter_dict = dict(counter)
                oreder_dict = OrderedDict(sorted(counter_dict.items(), key=lambda item: item[0]))
        self.save_results(sum(accuracies)/len(accuracies), sum(losses)/len(losses), wrong_count, total_count, log, Round)

    # test generated images by CNN classifier
    def external_test(self, Round):
        
        self.classifier.eval()
        wrong_count = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
        total_count = {j: 0 for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
        
        losses = []
        accuracies = []

        for images, labels in self.external_loader:
                
                x = images.to(self.device)
                y = labels.to(self.device)
    
                preds = self.classifier(x)
                loss = self.clf_criterion(preds, y)
    
                correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()
    
                losses.append(loss.item())
                accuracies.append(correct)
    
                # Check for misclassified images
                for img_idx, (true_label, pred_label) in enumerate(zip(y, torch.argmax(preds, dim=1))):
                    if true_label != pred_label:
                        # Store the predicted label for this true label
                        if true_label.item() not in wrong_count:
                            wrong_count[true_label.item()] = []
                        if true_label.item() not in total_count:
                            total_count[true_label.item()] = 0
                        wrong_count[true_label.item()].append(pred_label.item())
                        total_count[true_label.item()] += 1
                    else:
                        if true_label.item() not in total_count:
                            total_count[true_label.item()] = 0
                        total_count[true_label.item()] += 1
        
        
        log = f"\n   Summary of misclassifications:"
        for true_label, pred_labels in wrong_count.items():
            counter = Counter(pred_labels)
            counter_dict = dict(counter)
            oreder_dict = OrderedDict(sorted(counter_dict.items(), key=lambda item: item[0]))
            log += f"\n   True label {true_label} was misclassified as: {oreder_dict}"
        self.save_results_ex(sum(accuracies)/len(accuracies), sum(losses)/len(losses), wrong_count, total_count, log, Round)
     
    # test generated images by local classifier
    def gen_test(self, generated_loaders, states, save_path, Round, W):
        
        self.classifier.eval()

        log = f' Client{self.name+1}:'
        for (i, generated_loader), state in zip(enumerate(generated_loaders), states):
            correct_preds = 0
            total_preds = 0
            loss_dict = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
            wrong_count = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
            
            if state == 1:
                for batch_idx, (images, labels) in enumerate(generated_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # forward pass of classifier
                    test_preds = self.classifier(images)
                    pred_labels = torch.argmax(test_preds, dim=1)
                    # max_prob, pred_labels = test_preds.max(1)

                    # loss for each label
                    for label in range(self.args.num_classes):
                        mask = labels == label
                        if mask.sum().item() > 0:
                            loss_dict[label].append(self.clf_criterion(test_preds[mask], labels[mask]).item())

                    # Count correct predictions
                    correct_preds += (pred_labels == labels).sum().item()
                    total_preds += labels.size(0)
                    
                    if not os.path.exists(f"{save_path}/{Round}/generator{i+1}"):
                        os.makedirs(f"{save_path}/{Round}/generator{i+1}")

                    # Check for misclassified images
                    for img_idx, (true_label, pred_label) in enumerate(zip(labels, pred_labels)):
                        if true_label != pred_label:
                            # Log the misclassified image's true and predicted labels
                            # self.logger.info(f"Client{self.name+1}: Batch {batch_idx}, Image {img_idx}: True label: {true_label}, Predicted: {pred_label}")
                            
                            # Save the misclassified image
                            img_path = f"{save_path}/{Round}/generator{i+1}/img{img_idx}_true{true_label}_pred{pred_label}.jpg"
                            torchvision.utils.save_image(images[img_idx].cpu(), img_path, padding=0, normalize=True)
                            
                            # Store the predicted label for this true label
                            if true_label.item() not in wrong_count:
                                wrong_count[true_label.item()] = []
                            wrong_count[true_label.item()].append(pred_label.item())

                # After processing all batches for this generator, log the summary of misclassifications
                accuracy = correct_preds / total_preds
                # loss for each label
                loss = {label: sum(loss_dict[label])/len(loss_dict[label]) for label in range(self.args.num_classes)}
                log += f"\n  Generator {i+1} Accuracy: {accuracy:.5f}, Loss: {loss}\n   Summary of misclassifications for Generator {i+1}:"
                for true_label, pred_labels in wrong_count.items():
                    counter = Counter(pred_labels)
                    counter_dict = dict(counter)
                    oreder_dict = OrderedDict(sorted(counter_dict.items(), key=lambda item: item[0]))
                    log += f" \n    True label {true_label} was misclassified as: {oreder_dict}"
                    anomaly = [number for number, count in oreder_dict.items() if count > self.threshold]
                    if anomaly:
                        for poison_label in anomaly:
                            log += f"\n     Detection: Source label {true_label} was misclassified as Target label {poison_label}"
                        if not self.name == i:
                            W[i] = 0
            else:
                if not self.name == i:
                    W[i] = 0
        self.logger.info(log)
        return W

    def log_top_performance(self):
        self.logger.info(f" Client{self.name+1}: Top Performance Accuracy: {self.top_performance[0]:.5f} At Round: {self.top_performance[1]} Loss: {self.top_performance[2]:.5f} At Round: {self.top_performance[3]}")
        self.logger.info(f" Client{self.name+1}: Top Performance External Accuracy: {self.top_performance_ex[0]:.5f} At Round: {self.top_performance_ex[1]} Loss: {self.top_performance_ex[2]:.5f} At Round: {self.top_performance_ex[3]}")
    