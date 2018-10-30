import sys
sys.path.append('..')
import json
from dataloaders.dataloader_skip import Loader
from models.model_skip import Model
import torch
from torch.utils.data import Dataset, DataLoader
import time
from gensim.models import KeyedVectors
import os
import numpy as np
import tools.criteria as criteria


class Trainer(object):
    def __init__(self, params):

        self.params = params
        self.word2vec = KeyedVectors.load_word2vec_format(params["word2vec"], binary=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = Model(params, self.device).to(self.device)

        train_dataset = Loader(params, params['train_data'], params['train_ques_file'], self.word2vec, flag=True)
        val_dataset = Loader(params, params['val_data'], params['val_ques_file'], self.word2vec)
        test_dataset = Loader(params, params['test_data'], params['test_ques_file'], self.word2vec)


        self.train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            print(name)
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': 0},
                                           {'params': bias_p, 'weight_decay': 0}
                                           ], lr=self.params['learning_rate'])

        self.model_path = os.path.join(self.params['cache_dir']+'_skip')
        if not os.path.exists(self.model_path):
            print('create path: ', self.model_path)
            os.makedirs(self.model_path)

        self.best_model = None
        self.lr_epoch = 0
        # When iteration starts, queue and thread start to load data from files.


    def train(self):
        print('Trainnning begins......')
        best_epoch_acc = 0
        best_epoch_id = 0

        print('=================================')
        print('Model Params:')
        print(self.params)
        print('=================================')

        self.evaluate(self.test_loader)

        for i_epoch in range(self.params['max_epoches']):

            self.model.train()

            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch)
            t_end = time.time()
            print('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % self.params['evaluate_interval'] == 0 and i_epoch != 0:
                print('=================================')
                print('Overall evaluation')
                # print('=================================')
                # print('train set evaluation')
                # train_acc = self.evaluate(self.train_loader)
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.val_loader)
                print('=================================')
                print('test set evaluation')
                test_acc = self.evaluate(self.test_loader)
                print('=================================')
            else:
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.val_loader)
                print('=================================')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.best_model = self.model_path + timestamp + '.ckpt'
                torch.save(self.model.state_dict(), self.best_model)
            else:
                if i_epoch - best_epoch_id >= self.params['early_stopping']:
                    print('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

        print('=================================')
        print('Evaluating best model in file', self.best_model, '...')
        if self.best_model is not None:
            self.model.load_state_dict(torch.load(self.best_model))
            self.evaluate(self.test_loader)
        else:
            print('ERROR: No checkpoint available!')




    def train_one_epoch(self, i_epoch):

        loss_sum = 0
        t1 = time.time()
        for i_batch, (frame_vecs, frame_n, ques_vecs, labels, regs, idxs, windows, gt_windows) in enumerate(self.train_loader):
            frame_vecs = frame_vecs.to(self.device)
            frame_n = frame_n.to(self.device)
            ques_vecs = ques_vecs.to(self.device)
            labels = labels.to(self.device)
            regs = regs.to(self.device)
            idxs = idxs.to(self.device)

            # Forward pass
            batch_loss, predict_score = self.model(frame_vecs, frame_n, ques_vecs, labels, regs, idxs)

            # Backward and optimize
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            self.optimizer.step()

            self.lr_epoch += 1
            loss_sum += batch_loss.item()

            if i_batch % self.params['display_batch_interval'] == 0 and i_batch != 0:
                t2 = time.time()
                print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % ( i_epoch, i_batch, loss_sum / i_batch ,
                    (t2 - t1) / self.params['display_batch_interval']))
                t1 = t2

            if self.lr_epoch > 0 and self.lr_epoch % 3000 == 0:
                self.adjust_learning_rate()

        avg_batch_loss = loss_sum / i_batch

        return avg_batch_loss



    def evaluate(self, data_loader):

        # IoU_thresh = [0.5,0.7]
        # top1,top5

        all_correct_num_topn_IoU = np.zeros(shape=[1,4],dtype=np.float32)
        all_retrievd = 0.0

        self.model.eval()
        for i_batch, (frame_vecs, frame_n, ques_vecs, labels, regs, idxs, windows, gt_windows) in enumerate(data_loader):
            frame_vecs = frame_vecs.to(self.device)
            frame_n = frame_n.to(self.device)
            ques_vecs = ques_vecs.to(self.device)
            labels = labels.to(self.device)
            regs = regs.to(self.device)
            idxs = idxs.to(self.device)
            batch_size = len(frame_vecs)

            # Forward pass
            batch_loss, predict_score = self.model(frame_vecs, frame_n, ques_vecs, labels, regs, idxs)
            predict_score = predict_score.detach().cpu().numpy()

            for i in range(batch_size):
                predict_windows = windows[i]
                # predict_windows = predict_reg[i] + windows[i]
                result = criteria.compute_IoU_recall(predict_score[i], predict_windows, gt_windows[i])
                all_correct_num_topn_IoU += result

            all_retrievd += batch_size

        avg_correct_num_topn_IoU = all_correct_num_topn_IoU / all_retrievd
        print('=================================')
        print(avg_correct_num_topn_IoU)
        print('=================================')

        acc = avg_correct_num_topn_IoU[0,2]

        return acc

    def adjust_learning_rate(self, decay_rate=0.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    trainer = Trainer(config)

    trainer.train()
