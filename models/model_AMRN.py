import sys
sys.path.append('..')
import json
from dataloaders.dataloader_skip import Loader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors


class Model(nn.Module):
    def __init__(self, params, device):
        super(Model, self).__init__()

        self.params = params
        self.device = device
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']

        self.conv_bottom = nn.Sequential(
            nn.Conv1d(self.input_video_dim, self.hidden_size, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU()
        )

        self.conv_encoder_64 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=32, stride=8),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_128 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=64, stride=16),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_256 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=128, stride=32),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_512 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=256, stride=64),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.avg_pool = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)

        self.ques_rnn = nn.GRU(self.input_ques_dim, self.hidden_size, batch_first=True)


        self.fc_base = nn.Linear(self.hidden_size * 3 , self.hidden_size)
        self.fc_score = nn.Linear(self.hidden_size, 1)
        self.fc_reg = nn.Linear(self.hidden_size, 2)
        self.linear = nn.Linear(self.params['skip_thoughts_dim'],self.hidden_size)
        self.tanh = nn.Tanh()
        self.calculate_reg_loss = nn.SmoothL1Loss(reduction='elementwise_mean')

    def forward(self, frame_vecs, frame_n, ques_vecs, labels, regs, idxs):

        frame_vecs = torch.transpose(frame_vecs, 1, 2)
        ques_vecs = ques_vecs.squeeze(1)
        conv_bottom_out = self.conv_bottom(frame_vecs)
        conv_encoder_64_out = self.conv_encoder_64(conv_bottom_out)
        conv_encoder_128_out = self.conv_encoder_128(conv_bottom_out)
        conv_encoder_256_out = self.conv_encoder_256(conv_bottom_out)
        conv_encoder_512_out = self.conv_encoder_512(conv_bottom_out)
        # conv_top_64_out = self.conv_top(conv_encoder_64_out)
        # conv_top_128_out = self.conv_top(conv_encoder_128_out)
        # conv_top_256_out = self.conv_top(conv_encoder_256_out)
        # conv_top_512_out = self.conv_top(conv_encoder_512_out)
        ques_vecs = self.linear(ques_vecs)
        conv_encoder_64_weight = (conv_encoder_64_out * ques_vecs.unsqueeze(2).expand(-1, -1, conv_encoder_64_out.size(2))).sum(1)
        conv_encoder_128_weight = (conv_encoder_128_out * ques_vecs.unsqueeze(2).expand(-1, -1, conv_encoder_128_out.size(2))).sum(1)
        conv_encoder_256_weight = (conv_encoder_256_out * ques_vecs.unsqueeze(2).expand(-1, -1, conv_encoder_256_out.size(2))).sum(1)
        conv_encoder_512_weight = (conv_encoder_512_out * ques_vecs.unsqueeze(2).expand(-1, -1, conv_encoder_512_out.size(2))).sum(1)
        conv_top_64_out = self.avg_pool(conv_encoder_64_out * conv_encoder_64_weight.unsqueeze(1).expand(-1, self.hidden_size, -1))
        conv_top_128_out = self.avg_pool(conv_encoder_128_out * conv_encoder_128_weight.unsqueeze(1).expand(-1, self.hidden_size, -1))
        conv_top_256_out = self.avg_pool(conv_encoder_256_out * conv_encoder_256_weight.unsqueeze(1).expand(-1, self.hidden_size, -1))
        conv_top_512_out = self.avg_pool(conv_encoder_512_out * conv_encoder_512_weight.unsqueeze(1).expand(-1, self.hidden_size, -1))
        conv_encoder_out = torch.cat((conv_top_64_out, conv_top_128_out, conv_top_256_out, conv_top_512_out),dim=2)
        conv_encoder_out = torch.transpose(conv_encoder_out, 1, 2)

        ques_hidden = ques_vecs.unsqueeze(1).expand(-1, conv_encoder_out.size(1), -1)
        fused_mul = ques_hidden * conv_encoder_out
        fused_all = torch.cat((ques_hidden, fused_mul, conv_encoder_out), dim = 2)

        fused_all = self.fc_base(fused_all)
        score = self.fc_score(fused_all)
        score = nn.functional.softmax(score.squeeze(2),1)
        predict_reg = self.fc_reg(fused_all)

        idxs = idxs.squeeze()

        flag = (labels - 0.5) * (-2)
        score_loss = torch.log(1 + torch.exp(flag * score)) # [45, 21, 9, 3]
        raw_index = torch.arange(ques_vecs.size(0)).to(self.device)

        # print(idxs.dtype)
        # print(raw_index.dtype)
        # pos_loss = score_loss[raw_index, idxs]
        # all_score_loss = torch.sum(score_loss) / 78 + torch.sum(pos_loss)

        pos_loss = -torch.log(score[raw_index, idxs])
        all_score_loss = torch.mean(pos_loss, 0)

        pos_reg = predict_reg[raw_index, idxs]
        reg_loss = self.calculate_reg_loss(pos_reg,regs)

        all_loss = all_score_loss + 0.001 * reg_loss

        return  all_loss, score, predict_reg