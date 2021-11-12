# Created on 2018/12
# Author: Kaituo XU

import os
import time

import torch

from pit_criterion import cal_loss


class Solver(object):
    
    def __init__(self, data, model, optimizer, epochs):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['eval_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = True
        self.epochs = epochs
        self.half_lr = 0.0005
        self._reset()
        self.early_stop = False
        self.save_folder = 'C:/Users/silas/NoisyOverlappingSpeakers/model/'
        self.model_path = 'model.h5'

        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self.checkpoint = False
        self.max_norm = 2

    def _reset(self):
        # Reset

        self.start_epoch = 0
        # Create save folder
        #os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(torch.save(self.model.state_dict(), file_path)
,
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            padded_mixture, padded_source = data
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = padded_mixture.size()[2]
                padded_source = padded_source.cuda()
            estimate_source = self.model(padded_mixture, padded_source)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()


            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                      epoch + 1, i + 1, total_loss / (i + 1),
                      loss.item(), 1000 * (time.time() - start) / (i + 1)),
                  flush=True)

        return total_loss / (i + 1)
