from utils import weights_init

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

# client recieves args, generator, discriminator, and data loaders for training and testing
class client:
    def __init__(self, args, generator, discriminator, train_loader, test_loader, device, logger, classifier=None, local_model=None):
        self.args = args
        self.netG = generator
        self.netD = discriminator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger

        self.device = device

        # make knn classifier
        self.knn = KNeighborsClassifier(n_neighbors=3)

        # GAN hyperparameters
        self.netG.to(device)
        self.netD.to(device)
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

        # FL Task hyperparameters
        self.classifier = classifier
        self.classifier.to(device)
        self.clf_criterion = nn.CrossEntropyLoss()
        self.clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
        # local classifier
        self.local_model = local_model
        self.local_model.to(device)
        self.local_criterion = nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
        
        # We calculate Binary cross entropy loss
        self.criterion = nn.BCELoss()
        # Adam optimizer for generator
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.args.Adam_lr, betas=(self.args.Adam_beta1, 0.999))
        # Adam optimizer for discriminator
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.args.Adam_lr, betas=(self.args.Adam_beta1, 0.999))

        # labels for training images x for Discriminator training
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

    def gan_train(self, save_path):
        self.netG.train()

        # List of values, which will be used for plotting purpose
        D_losses = []
        G_losses = []
        Dx_values = []
        DGz_values = []

        # number of training steps done on discriminator
        step = 0
        for epoch in range(self.args.num_epoch):
            epoch_D_losses = []
            epoch_G_losses = []
            epoch_Dx = []
            epoch_DGz = []
        
            # iterate through data loader generator object
            for images, y_labels in self.train_loader:
                step += 1

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
                D_x_loss = self.criterion(x_preds, self.labels_real)

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

                self.logger.info(f" Epoch {epoch+1}/{self.args.num_epoch} Discriminator Loss {D_losses[-1]:.3f} Generator Loss {G_losses[-1]:.3f}"
                    + f" D(x) {Dx_values[-1]:.3f} D(G(x)) {DGz_values[-1]:.3f}")

                # Generating images after each epoch and saving
                # set generator to evaluation mode
                self.netG.eval()
                with torch.no_grad():
                    # forward pass of G and generated image
                    fake_test = self.netG(self.z_test, self.test_Gy).cpu()
                    # save images in grid of 10 * 10
                    torchvision.utils.save_image(fake_test, f"{save_path}/epoch_{epoch+1}.jpg", nrow=10, padding=0, normalize=True)
                # set generator to training mode
                self.netG.train()
        
    def generate_image(self, save_path):
        self.netG.eval()
        with torch.no_grad():
            new_Z = torch.randn(self.args.num_classes*self.args.g_img_num, self.args.size_z).to(self.device)
            new_test_y = torch.tensor(list(range(self.args.num_classes))*self.args.g_img_num).type(torch.LongTensor)
            new_test_Gy = self.onehot[new_test_y].to(self.device)
            fake_image = self.netG(new_Z, new_test_Gy).cpu()
            torchvision.utils.save_image(fake_image, f"{save_path}/data.jpg", nrow=10, padding=0, normalize=True)
        dataset = torch.utils.data.TensorDataset(fake_image, new_test_y)
        generated_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        return generated_loader

    def save_model(self, save_path):
        torch.save(self.netG.state_dict(), f"{save_path}/generator.pth")
        torch.save(self.netD.state_dict(), f"{save_path}/discriminator.pth")
    
    def load_model(self, load_path):
        self.netG.load_state_dict(torch.load(f"{load_path}/generator.pth"))
        self.netD.load_state_dict(torch.load(f"{load_path}/discriminator.pth"))

    # only CNN classifier training (from model.py)
    def clf_train(self, generated_loaders, save_path):
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

                self.logger.info(f" Epoch {epoch}/{self.args.clf_epoch} Loss {losses[-1]:.3f} Accuracy {accuracies[-1]:.3f}")
            
            if (epoch+1) % self.args.d_iter == 0:
                self.clf_test(generated_loaders, save_path, epoch)
 
    # test generated images by CNN classifier
    def clf_test(self, generated_loaders, save_path, epoch):
        self.classifier.eval()

        gen_loader_sig_values = []

        for i, generated_loader in enumerate(generated_loaders):
            correct_preds = 0
            total_preds = 0
            loss_dict = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
            sigmoid_dict = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of sigmoid scores for each label
            wrong_count = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label

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
                
                if not os.path.exists(f"{save_path}/{epoch}/generator{i+1}"):
                    os.makedirs(f"{save_path}/{epoch}/generator{i+1}")

                # Check for misclassified images
                for img_idx, (true_label, pred_label) in enumerate(zip(labels, pred_labels)):
                    if true_label != pred_label:
                        # Log the misclassified image's true and predicted labels
                        # self.logger.info(f"Batch {batch_idx}, Image {img_idx}: True label: {true_label}, Predicted: {pred_label}")
                        
                        # Save the misclassified image
                        img_path = f"{save_path}/{epoch}/generator{i+1}/img{img_idx}_true{true_label}_pred{pred_label}.jpg"
                        torchvision.utils.save_image(images[img_idx].cpu(), img_path, padding=0, normalize=True)
                        
                        # Store the predicted label for this true label
                        if true_label.item() not in wrong_count:
                            wrong_count[true_label.item()] = []
                        wrong_count[true_label.item()].append(pred_label.item())
                
            gen_loader_sig_values.append(sigmoid_dict)

            # After processing all batches for this generator, log the summary of misclassifications
            accuracy = correct_preds / total_preds
            # loss for each label
            loss = {label: sum(loss_dict[label])/len(loss_dict[label]) for label in range(self.args.num_classes)}
            self.logger.info(f"Generator {i+1} Accuracy: {accuracy:.3f}")
            self.logger.info(f"Generator {i+1} Loss: {loss}")
            self.logger.info(f"Summary of misclassifications for Generator {i+1}:")
            for true_label, pred_labels in wrong_count.items():
                self.logger.info(f"True label {true_label} was misclassified as: {pred_labels}")
    
    # local classifier training
    def local_train(self, generated_loaders, save_path):
        self.local_model.train()

        # List of values, which will be used for plotting purpose
        losses = []
        accuracies = []

        for epoch in range(self.args.local_epoch):
            epoch_losses = []
            epoch_accuracies = []
            for images, labels in self.train_loader:
                # images will be send to gpu, if cuda available
                x = images.to(self.device)
                # labels will be send to gpu, if cuda available
                y = labels.to(self.device)

                # forward pass
                preds = self.local_model(x)
                # calculate loss
                loss = self.local_criterion(preds, y)

                # calculate accuracy
                correct = (torch.argmax(preds, dim=1) == y).type(torch.FloatTensor).mean().item()

                # save values for plots
                epoch_losses.append(loss.item())
                epoch_accuracies.append(correct)

                # zero accumalted grads
                self.local_model.zero_grad()
                # do backward pass
                loss.backward()
                # update classifier model
                self.local_optimizer.step()
            else:
                # calculate average value for one epoch
                losses.append(sum(epoch_losses)/len(epoch_losses))
                accuracies.append(sum(epoch_accuracies)/len(epoch_accuracies))

                self.logger.info(f" Epoch {epoch}/{self.args.local_epoch} Loss {losses[-1]:.3f} Accuracy {accuracies[-1]:.3f}")
            
            if (epoch+1) % self.args.d_iter == 0:
                self.local_test(generated_loaders, save_path, epoch)
            
    # test generated images by local classifier
    def local_test(self, generated_loaders, save_path, epoch):
        self.local_model.eval()

        gen_loader_sig_values = []

        for i, generated_loader in enumerate(generated_loaders):
            correct_preds = 0
            total_preds = 0
            loss_dict = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label
            sigmoid_dict = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of sigmoid scores for each label
            wrong_count = {j: [] for j in range(self.args.num_classes)}  # Initialize a dictionary to keep track of wrong predictions for each label

            for batch_idx, (images, labels) in enumerate(generated_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass of classifier
                test_preds = self.local_model(images)
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
                
                if not os.path.exists(f"{save_path}/{epoch}/generator{i+1}"):
                    os.makedirs(f"{save_path}/{epoch}/generator{i+1}")

                # Check for misclassified images
                for img_idx, (true_label, pred_label) in enumerate(zip(labels, pred_labels)):
                    if true_label != pred_label:
                        # Log the misclassified image's true and predicted labels
                        # self.logger.info(f"Batch {batch_idx}, Image {img_idx}: True label: {true_label}, Predicted: {pred_label}")
                        
                        # Save the misclassified image
                        img_path = f"{save_path}/{epoch}/generator{i+1}/img{img_idx}_true{true_label}_pred{pred_label}.jpg"
                        torchvision.utils.save_image(images[img_idx].cpu(), img_path, padding=0, normalize=True)
                        
                        # Store the predicted label for this true label
                        if true_label.item() not in wrong_count:
                            wrong_count[true_label.item()] = []
                        wrong_count[true_label.item()].append(pred_label.item())
                
            gen_loader_sig_values.append(sigmoid_dict)

            # After processing all batches for this generator, log the summary of misclassifications
            accuracy = correct_preds / total_preds
            # loss for each label
            loss = {label: sum(loss_dict[label])/len(loss_dict[label]) for label in range(self.args.num_classes)}
            self.logger.info(f"Generator {i+1} Accuracy: {accuracy:.3f}")
            self.logger.info(f"Generator {i+1} Loss: {loss}")
            self.logger.info(f"Summary of misclassifications for Generator {i+1}:")
            for true_label, pred_labels in wrong_count.items():
                self.logger.info(f"True label {true_label} was misclassified as: {pred_labels}")
            
 
