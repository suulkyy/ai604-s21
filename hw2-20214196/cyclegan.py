import os
import time
import datetime
import random
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dataloader import create_dataloader, FolderDataset
from inception_score import Inception_Score
from util import conv, deconv, denorm, save_checkpoint, load_checkpoint


class CycleGAN_Generator(nn.Module):
    def __init__(self, input_nc: int=3, output_nc: int=3, ngf: int=64):
        """
        Parameters
        input_nc: the channels of input images
        output_nc: the channels of ouput images
        ngf: the numer of filters in the last convolutional layer
        """
        super(CycleGAN_Generator, self).__init__()

        # Architecture Hyper-parameters
        n_downsampling = 2
        n_blocks = 6
        model = []

        ### CycleGAN Generator
        # You have to implement CycleGAN Generation with 3 down-sampling layers, 6 residual blocks and 3 up-sampling layers.
        # For more details on the generator architecture, please check the homework description.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use 'nn.ReflectionPad2d' for padding.
        # Note 3: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 15 lines)
        model.append(nn.ReflectionPad2d(3))
        model.append(conv(c_in=input_nc, c_out=64, k_size=7, stride=1, pad=0, bias=True, norm='in', activation='relu'))
        model.append(conv(c_in=64, c_out=128, k_size=3, stride=2, pad=1, bias=True, norm='in', activation='relu'))
        model.append(conv(c_in=128, c_out=256, k_size=3, stride=2, pad=1, bias=True, norm='in', activation='relu'))
        for i in range(n_blocks):
            model.append(ResnetBlock(256))
        model.append(deconv(c_in=256, c_out=128, k_size=3, stride=2, pad=1, output_padding=1, bias=True, norm='in', activation='relu'))
        model.append(deconv(c_in=128, c_out=ngf, k_size=3, stride=2, pad=1, output_padding=1, bias=True, norm='in', activation='relu'))
        model.append(nn.ReflectionPad2d(3))
        model.append(conv(c_in=ngf, c_out=output_nc, k_size=7, stride=1, pad=0, norm=None, activation='tanh'))
        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResnetBlock, self).__init__()

        self.conv_block = None

        ### Residual block
        # You have to implement the residual block with 2 convolutional layers.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use 'nn.ReflectionPad2d' for padding.
        # Note 3: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 4 lines)
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1),
                        conv(c_in=dim, c_out=dim, k_size=3, stride=1, pad=0, bias=True, norm='in', activation='relu'),
                        nn.ReflectionPad2d(1),
                        conv(c_in=dim, c_out=dim, k_size=3, stride=1, pad=0, bias=True, norm='in'))
        ### END YOUR CODE

    def forward(self, x: torch.Tensor):
        out = None

        ### Skip connection
        # Implement the skip connection

        # Add skip connections
        ### YOUR CODE HERE (~ 1 line)
        out = x + self.conv_block(x)

        ### END YOUR CODE
        return out


class CycleGAN_Discriminator(nn.Module):
    def __init__(self, input_nc: int=3, ndf: int=64):
        """
        Parameters
        input_nc: the channels of input images
        ndf: the numer of filters in the first convolutional layer
        """
        super(CycleGAN_Discriminator, self).__init__()

        # Architecture Hyper-parameters
        n_layers = 3
        model = []

        ### CycleGAN Discriminator
        # You have to implement the CycleGAN Discriminator.
        # For more details on the discriminator architecture, please check the homework description.
        # Note 1: Recommend to use 'conv' and 'deconv' function implemented in 'util.py'.
        # Note 2: You have to use instance normalization for normalizing the feature maps.

        ### YOUR CODE HERE (~ 12 lines)
        model.append(conv(c_in=input_nc, c_out=ndf, k_size=4, stride=2, pad=1, norm=None, activation='lrelu'))
        model.append(conv(c_in=ndf, c_out=128, k_size=4, stride=2, pad=1, bias=True, norm='in', activation='lrelu'))
        model.append(conv(c_in=128, c_out=256, k_size=4, stride=2, pad=1, bias=True, norm='in', activation='lrelu'))
        model.append(conv(c_in=256, c_out=512, k_size=4, stride=1, pad=1, bias=True, norm='in', activation='lrelu'))
        model.append(conv(c_in=512, c_out=1, k_size=4, stride=1, pad=1, norm=None))

        ### END YOUR CODE

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        return self.model(input)


class CycleGAN_Solver():
    def __init__(self, lr: float=0.0002, batch_size: int=1, num_workers: int=1):
        """
        Parameters
        lr: learning rate
        batch_size: batch size
        num_workers: the number of workers for train dataloader
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Declare Generator and Discriminator
        self.netG_A2B = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)
        self.netG_B2A = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)
        self.netD_A = CycleGAN_Discriminator(input_nc=3, ndf=64)
        self.netD_B = CycleGAN_Discriminator(input_nc=3, ndf=64)

        # Declare the Criterion for GAN loss
        # Doc for MSE Loss: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        # Doc for L1 Loss: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html

        self.criterion_GAN: nn.Module = None
        self.criterion_cycle: nn.Module = None
        self.criterion_identity: nn.Module = None

        ### YOUR CODE HERE (~ 3 lines)
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        ### END YOUR CODE

        self.optimizerG: optim.Optimizer = None
        self.optimizerD_A: optim.Optimizer = None
        self.optimizerD_B: optim.Optimizer = None

        # Declare the Optimizer for training
        # Doc for Adam optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam

        ### YOUR CODE HERE (~ 3 lines)
        self.optimizerG = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=lr)
        self.optimizerD_A = optim.Adam(self.netD_A.parameters(), lr=lr)
        self.optimizerD_B = optim.Adam(self.netD_B.parameters(), lr=lr) 

        ### END YOUR CODE

        # Declare the DataLoader
        # You have to implement 'Summer2WinterDataset' in 'dataloader.py'
        self.trainloader, self.testloader = create_dataloader(dataset='summer2winter', batch_size=batch_size,
                                                              num_workers=num_workers)
        ### END YOUR CODE

        # Make directory
        os.makedirs('./results/cyclegan/images/', exist_ok=True)
        os.makedirs('./results/cyclegan/checkpoints/', exist_ok=True)

    def train(self, epochs: int=100):

        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        start_time = time.time()

        print("=====Train Start======")

        for epoch in range(epochs):
            for iter, (real_A, real_B) in enumerate(self.trainloader):
                self.netG_A2B.train()
                self.netG_B2A.train()
                self.netD_A.train()
                self.netD_B.train()

                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                ###################################################################################
                # (1) Update Generator
                # Train the generator (self.netG_A2B & self.netG_B2A).
                # You have to implement 3 types of loss functions ('identity loss', 'gan loss', 'cycle consistency loss')
                ###################################################################################

                identity_loss: torch.Tensor() = None
                gan_loss: torch.Tensor() = None
                cycle_loss: torch.Tensor() = None
                lossG: torch.Tensor() = None

                ### YOUR CODE HERE (~ 15 lines)
                target_real = torch.ones(1,1,30,30).to(self.device)
                target_fake = torch.zeros(1,1,30,30).to(self.device)
                # identity loss
                # G_A2B be equal to B if real B is fed
                same_B = self.netG_A2B(real_B)
                identity_loss_B = self.criterion_identity(same_B, real_B)*5.0
                # G_B2A be equal to A if real A is fed
                same_A = self.netG_B2A(real_A)
                identity_loss_A = self.criterion_identity(same_A, real_A)*5.0
                identity_loss = identity_loss_B + identity_loss_A
                
                # gan loss
                # G_A2B loss
                fake_B = self.netG_A2B(real_A)
                pred_fake_B = self.netD_B(fake_B)
                gan_A2B_loss = self.criterion_GAN(pred_fake_B, target_real)
                # G_B2A loss
                fake_A = self.netG_B2A(real_B)
                pred_fake_A = self.netD_A(fake_A)
                gan_B2A_loss = self.criterion_GAN(pred_fake_A, target_real)
                gan_loss = gan_A2B_loss + gan_B2A_loss
                
                # cycle consistency loss
                # Reconstructed loss for ABA
                reconst_A = self.netG_B2A(fake_B)
                cycle_ABA_loss = self.criterion_cycle(reconst_A, real_A)*10.0
                # Reconstructed loss for BAB
                reconst_B = self.netG_A2B(fake_A)
                cycle_BAB_loss = self.criterion_cycle(reconst_B, real_B)*10.0
                cycle_loss = cycle_ABA_loss + cycle_BAB_loss

                # Total loss
                lossG = identity_loss + gan_loss + cycle_loss
                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossG_fuction(identity_loss, gan_loss, cycle_loss)

                self.optimizerG.zero_grad()
                lossG.backward()
                self.optimizerG.step()

                ###################################################################################
                # (2) Update Discriminator
                # Train the discrminator (self.netD_A & self.netD_B).
                ###################################################################################
                # Discriminator A

                lossD_A: torch.Tensor() = None

                ### YOUR CODE HERE (~ 4 lines)
                # Real loss
                pred_real_A = self.netD_A(real_A)
                lossD_real_A = self.criterion_GAN(pred_real_A, target_real)
        
                # Fake loss
                pred_fake_A = self.netD_A(fake_A.detach())
                lossD_fake_A = self.criterion_GAN(pred_fake_A, target_fake)

                # Total loss
                lossD_A = lossD_real_A + lossD_fake_A
                ### END YOUR CODE

                self.optimizerD_A.zero_grad()
                lossD_A.backward()
                self.optimizerD_A.step()

                # Discriminator B

                lossD_B: torch.Tensor() = None

                ### YOUR CODE HERE (~ 4 lines)
                # Real loss
                pred_real_B = self.netD_B(real_B)
                lossD_real_B = self.criterion_GAN(pred_real_B, target_real)
                
                # Fake loss
                pred_fake_B = self.netD_B(fake_B.detach())
                lossD_fake_B = self.criterion_GAN(pred_fake_B, target_fake)
                
                # Total Loss
                lossD_B = lossD_real_B + lossD_fake_B
                ### END YOUR CODE

                # Test code
                if epoch == 0 and iter == 0:
                    test_lossD_fuction(lossD_A, lossD_B)

                self.optimizerD_B.zero_grad()
                lossD_B.backward()
                self.optimizerD_B.step()

                if (iter + 1) % 100 == 0:
                    end_time = time.time() - start_time
                    end_time = str(datetime.timedelta(seconds=end_time))[:-7]
                    print('Time [%s], Epoch [%d/%d], Step[%d/%d], lossD_A: %.4f, lossD_B: %.4f, lossG: %.4f'
                          % (end_time, epoch+1, epochs, iter+1, len(self.trainloader), lossD_A.item(), lossD_A.item(), lossG.item()))

            # Save Images
            fake_A = fake_A.reshape(fake_A.size(0), 3, 256, 256)
            fake_B = fake_B.reshape(fake_B.size(0), 3, 256, 256)

            save_image(denorm(fake_B), os.path.join('./results/cyclegan/images', 'fakeA2B-{:03d}.png'.format(epoch + 1)))
            save_image(denorm(fake_A), os.path.join('./results/cyclegan/images', 'fakeB2A-{:03d}.png'.format(epoch + 1)))

            if (epoch + 1) % 10 == 0:
                save_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_{:02d}.pth'.format(epoch+1), self.device)
                save_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_{:02d}.pth'.format(epoch+1), self.device)

                save_checkpoint(self.netD_A, './results/cyclegan/checkpoints/netD_A_{:02d}.pth'.format(epoch+1), self.device)
                save_checkpoint(self.netD_B, './results/cyclegan/checkpoints/netD_B_{:02d}.pth'.format(epoch+1), self.device)


        # Save Checkpoints
        save_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_final.pth', self.device)
        save_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_final.pth', self.device)

        save_checkpoint(self.netD_A, './results/cyclegan/checkpoints/netD_A_final.pth', self.device)
        save_checkpoint(self.netD_B, './results/cyclegan/checkpoints/netD_B_final.pth', self.device)


    def test(self):
        load_checkpoint(self.netG_A2B, './results/cyclegan/checkpoints/netG_A2B_final.pth', self.device)
        load_checkpoint(self.netG_B2A, './results/cyclegan/checkpoints/netG_B2A_final.pth', self.device)

        os.makedirs('./results/cyclegan/evaluation/', exist_ok=True)

        self.netG_A2B.eval()
        self.netG_B2A.eval()

        print("=====Test Start======")

        with torch.no_grad():
            for iter, (real_A, real_B) in enumerate(self.testloader):
                fake_A = self.netG_B2A(real_B.to(self.device))
                save_image(denorm(fake_A), os.path.join('./results/cyclegan/evaluation', 'fake_image-{:05d}.png'.format(iter + 1)))

        # Compute the Inception score
        dataset = FolderDataset(folder=os.path.join('./results/cyclegan/evaluation'))
        Inception = Inception_Score(dataset)
        score = Inception.compute_score(splits=1)

        print('Inception Score : ', score)


#############################################
# Testing functions below.                  #
#############################################

def test_initializer_and_forward():

    print("=====Model Initializer Test Case======")

    netG = CycleGAN_Generator(input_nc=3, output_nc=3, ngf=64)

    try:
        netG.load_state_dict(torch.load("sanity_check/sanity_check_cyclegan_netG.pth", map_location='cpu'))
    except Exception as e:
        print("Your CycleGAN generator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The first test passed!")

    netD = CycleGAN_Discriminator(input_nc=3, ndf=64)

    try:
        netD.load_state_dict(torch.load("sanity_check/sanity_check_cyclegan_netD.pth", map_location='cpu'))
    except Exception as e:
        print("Your CycleGAN discriminator initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The second test passed!")

    print("All 2 tests passed!")


def test_lossG_fuction(identity_loss, gan_loss, cycle_loss):

    print("=====Generator Loss Test Case======")

    # the first test
    assert identity_loss.detach().allclose(torch.tensor(5.7538), atol=1e-2), \
        f"Identity Loss of the model does not match expected result."
    print("The first test passed!")

    # the second test
    assert gan_loss.detach().allclose(torch.tensor(2.2067), atol=1e-2), \
        f"Adversarial Loss of the model does not match expected result."
    print("The second test passed!")

    # the third test
    assert cycle_loss.detach().allclose(torch.tensor(11.5476), atol=1e-2), \
        f"Cycle Consistency Loss of the model does not match expected result."
    print("The third test passed!")

    print("All test passed!")

def test_lossD_fuction(lossD_A, lossD_B):

    print("=====Discriminator Loss Test Case======")

    assert lossD_A.detach().allclose(torch.tensor(1.1283), atol=1e-2), \
        f"Discriminator A Loss of the model does not match expected result."
    print("The first test passed!")

    assert lossD_B.detach().allclose(torch.tensor(1.2319), atol=1e-2), \
        f"Discriminator B Loss of the model does not match expected result."
    print("The second test passed!")

    print("All test passed!")


if __name__ == "__main__":
    torch.set_printoptions(precision=4)
    random.seed(1234)
    torch.manual_seed(1234)

    test_initializer_and_forward()

    # Hyper-parameters
    # For good performance, you have to train CycleGAN about 200 epochs.
    # However, it takes 1 to 2 days to train the model, so it is okay to train only 30 epochs.
    epochs = 30 # 200
    lr = 0.0002
    batch_size = 1
    num_workers = 1
    train = False  # train : True / test : False (Compute the Inception Score)

    # Train or Test
    solver = CycleGAN_Solver(lr=lr, batch_size=batch_size, num_workers=num_workers)
    if train:
        solver.train(epochs=epochs)
    else:
        solver.test()