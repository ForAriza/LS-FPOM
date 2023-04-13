import os

import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from skimage import io


class LsFPOMNet(nn.Module):
    def __init__(self, origin, channel, theta, weight, weight_decay,
                 orient_uniform_coef_sing, orient_uniform_coef_doub,
                 orient_uniform_coef_sing0, orient_uniform_coef_doub0,
                 total, alpha_sing, alpha_doub, background):
        super(LsFPOMNet, self).__init__()
        self.origin = origin
        self.channel = channel
        self.theta = nn.Parameter(data=theta, requires_grad=False)
        self.weight = nn.Parameter(data=weight, requires_grad=False)
        self.weight_decay = weight_decay
        self.orient_uniform_coef_sing0 = orient_uniform_coef_sing0
        self.orient_uniform_coef_doub0 = orient_uniform_coef_doub0

        self.orient_uniform_coef_sing = nn.Parameter(data=orient_uniform_coef_sing, requires_grad=True)
        self.orient_uniform_coef_doub = nn.Parameter(data=orient_uniform_coef_doub, requires_grad=True)
        self.total = nn.Parameter(data=total, requires_grad=True)
        self.alpha_sing = nn.Parameter(data=alpha_sing, requires_grad=True)
        self.alpha_doub = nn.Parameter(data=alpha_doub, requires_grad=True)
        self.background = nn.Parameter(data=background, requires_grad=True)

    def mseloss(self, fluorescence_blur):
        return torch.sum(torch.pow(fluorescence_blur - self.origin, 2))

    def regularloss(self, fluorescence):
        return self.weight_decay[0] * torch.sum(torch.norm(fluorescence, 1)) \
               + self.weight_decay[1] * torch.sum(torch.norm(self.background, 1)) \
               + self.weight_decay[2] * torch.sum(
            torch.pow(
                self.orient_uniform_coef_sing - self.orient_uniform_coef_sing0, 2
            )
            + torch.pow(
                self.orient_uniform_coef_doub - self.orient_uniform_coef_doub0, 2)
        )

    def forward(self):
        # f = t[4oufs*cost*cos(2a-t)+oufd*cos(4a-2t)+(cos2t+2)]
        orient_uniform_coef_sing = torch.minimum(
            self.orient_uniform_coef_sing,
            torch.tensor(1, dtype=torch.float32)
        )
        orient_uniform_coef_doub = torch.minimum(
            self.orient_uniform_coef_doub,
            torch.tensor(1, dtype=torch.float32)
        )
        fluorescence = torch.pow(self.total, 2) * (
                4 * torch.pow(orient_uniform_coef_sing, 2)
                * torch.cos(self.theta * torch.pi / 180)
                * torch.cos((2 * torch.pow(self.alpha_sing, 2) - self.theta) * torch.pi / 180)
                + torch.pow(orient_uniform_coef_doub, 2)
                * torch.cos(2 * (2 * torch.pow(self.alpha_doub, 2) - self.theta) * torch.pi / 180)
                + (torch.cos(2 * self.theta * torch.pi / 180) + 2)
        )
        # fluorescence = torch.maximum(fluorescence, torch.tensor(0, dtype=torch.float32))
        fluorescence_blur = F.conv2d(
            input=fluorescence,
            weight=self.weight,
            bias=None,
            stride=1,
            padding='same',
            groups=self.channel
        ) + torch.pow(self.background, 2)

        risk_empirical = self.mseloss(fluorescence_blur=fluorescence_blur)
        risk_structural = self.regularloss(fluorescence=fluorescence)
        return risk_empirical + risk_structural, [
            torch.pow(orient_uniform_coef_sing, 2),
            torch.pow(orient_uniform_coef_doub, 2),
            torch.pow(self.total, 2),
            torch.pow(self.alpha_sing, 2),
            torch.pow(self.alpha_doub, 2),
            torch.pow(self.background, 2),
            fluorescence,
            fluorescence_blur
        ], risk_empirical, risk_structural


class LsFPOM:
    def __init__(self, sample, weight_decay, learning_rate, epoch, interval):
        self.sample = sample
        self.text = [self.sample]
        self.list_weight_decay_name = [
            'Fluorescence',
            'Background',
            'OUF']
        self.list_result_name = [
            'OUCs',
            'OUCd',
            'Total',
            'Alphas',
            'Alphad',
            'Background',
            'Fluorescence',
            'FluorescenceBlur',
            'Single',
            'Double',
            'RSquare',
            'risk_empirical',
            'risk_structural',
            'risk_total'
        ]

        self.epoch = epoch
        self.interval = interval

        self.weight_decay = torch.tensor(weight_decay, dtype=torch.float32)
        self.learning_rate = np.array(learning_rate, dtype=np.float32)

        self.path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + r'\data'
        self.path_result = self.path + r'\{}\result'.format(self.sample)
        self.path_psf = self.path + r'\rawData\PSF\psf_fish.tif'
        self.path_clear = self.path + r'\{}\block\clear\{}.tif'.format(self.sample, self.sample)
        self.path_blur = self.path + r'\{}\block\blur\{}.tif'.format(self.sample, self.sample)

        self.shape = io.imread(self.path_clear).shape
        self.channel = self.shape[0]

        self.weight = self.load_weight()
        self.origin, self.target, self.intensity = self.load_origin(path=self.path_clear)
        self.ac_component = torch.sqrt(
            torch.from_numpy(np.max(self.target, axis=0) - np.min(self.target, axis=0)) / 2
        )
        self.total = torch.sqrt(torch.from_numpy(np.min(self.target, axis=0)))
        self.sing = self.ac_component * 5 / 10
        self.doub = self.ac_component * 5 / 10
        # self.orient_uniform_coef_sing = self.extend * self.sing
        # self.orient_uniform_coef_doub = self.extend * self.doub
        self.orient_uniform_coef_sing = np.minimum(self.sing / np.maximum(self.total, 1e-6), 1)
        self.orient_uniform_coef_doub = np.minimum(self.doub / np.maximum(self.total, 1e-6), 1)

        self.extend = torch.ones([1, 1, self.shape[1], self.shape[2]], dtype=torch.float32)

        self.total = self.extend * 5 / 10 * self.total
        self.orient_uniform_coef_sing = self.extend * self.orient_uniform_coef_sing
        self.orient_uniform_coef_doub = self.extend * self.orient_uniform_coef_doub
        self.alpha_sing = self.extend * torch.sqrt(torch.tensor(90, dtype=torch.float32))
        self.alpha_doub = self.extend * torch.sqrt(torch.tensor(45, dtype=torch.float32))
        self.background = self.extend * 5 / 10 * self.total

        self.theta = 4 * (np.ones(self.shape).T * np.arange(self.channel)).T.astype(np.float32)
        self.theta = np.mod(self.theta, 180)
        self.theta = torch.from_numpy(self.theta)

        self.orient_uniform_coef_sing0 = self.extend * self.orient_uniform_coef_sing
        self.orient_uniform_coef_doub0 = self.extend * self.orient_uniform_coef_doub

        self.weight_decay = self.weight_decay.cuda()
        self.origin = self.origin.cuda()
        self.sing = self.sing.cuda()
        self.doub = self.doub.cuda()
        self.total = self.total.cuda()
        self.orient_uniform_coef_sing = self.orient_uniform_coef_sing.cuda()
        self.orient_uniform_coef_doub = self.orient_uniform_coef_doub.cuda()
        self.alpha_sing = self.alpha_sing.cuda()
        self.alpha_doub = self.alpha_doub.cuda()
        self.background = self.background.cuda()
        self.theta = self.theta.cuda()

        self.orient_uniform_coef_sing0 = self.orient_uniform_coef_sing0.cuda()
        self.orient_uniform_coef_doub0 = self.orient_uniform_coef_doub0.cuda()

    def load_weight(self):
        psf = io.imread(self.path_psf).astype(np.float32)
        psf = torch.from_numpy(psf)

        weight = torch.zeros([self.channel, psf.shape[0], psf.shape[1]], dtype=torch.float32)
        for k in range(self.channel):
            weight[k] = psf / psf.sum()
        weight = weight.unsqueeze(1).cuda()
        return weight

    @staticmethod
    def load_origin(path):
        origin = io.imread(path).astype(np.float32)
        intensity = np.max(origin)
        origin = origin / intensity
        target = origin
        origin = torch.from_numpy(origin)
        return origin, target, intensity

    def add_text(self, text):
        self.text.append(text)

    def get_round(self):
        weight_decay_round = ['{:.6f}'.format(self.weight_decay[k]) for k in range(len(self.weight_decay))]
        learning_rate_round = ['{:.6f}'.format(self.learning_rate[k]) for k in range(6)]
        parameter = [weight_decay_round, learning_rate_round]
        self.add_text(parameter)
        return parameter

    def log_writer(self, new):
        parameter = self.text[1]
        weight_decay_round = parameter[0]
        learning_rate_round = parameter[1]

        result = self.text[2:]
        text = 'sequence\n{}\n'.format(self.sample) + '\nparameter\n' + 'weight_decay\n'

        for k in range(len(weight_decay_round)):
            text += '{} {}\n'.format(self.list_weight_decay_name[k], weight_decay_round[k])

        text += '\nlearning_rate\n'

        for k in range(len(learning_rate_round)):
            text += '{} {}\n'.format(self.list_result_name[k], learning_rate_round[k])

        text += '\nresult\n'

        for k in range(len(result)):
            text += '\nstep {}\n'.format(result[k][0]) + 'mean\n'
            print(len(result[k]))
            for j in range(len(result[k]) - 1):
                text += '{} {}\n'.format(self.list_result_name[j], result[k][j + 1])

        if not os.path.exists(new):
            os.makedirs(new)

        name = new + r'\log.txt'
        file = open(name, 'w')
        file.write(text)
        file.close()

    def get_optimizer(self, net):
        optimizer = torch.optim.Adagrad([
            {'params': net.orient_uniform_coef_sing,
             'self.learning_rate': self.learning_rate[0],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0},
            {'params': net.orient_uniform_coef_doub,
             'self.learning_rate': self.learning_rate[1],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0},
            {'params': net.total,
             'self.learning_rate': self.learning_rate[2],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0},
            {'params': net.alpha_sing,
             'self.learning_rate': self.learning_rate[3],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0},
            {'params': net.alpha_doub,
             'self.learning_rate': self.learning_rate[4],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0},
            {'params': net.background,
             'self.learning_rate': self.learning_rate[5],
             'betas': (0.9, 0.999),
             'eps': 1e-08,
             'self.weight_decay': 0}
        ])
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epoch)
        return scheduler

    def get_path(self, parameter):
        path_dir = r'\{}_'.format(parameter[0][0]) \
                   + r'{}_'.format(parameter[0][1]) \
                   + r'{}-'.format(parameter[0][2]) \
                   + r'{}_'.format(parameter[1][0]) \
                   + r'{}_'.format(parameter[1][1]) \
                   + r'{}_'.format(parameter[1][2]) \
                   + r'{}_'.format(parameter[1][3]) \
                   + r'{}_'.format(parameter[1][4]) \
                   + r'{}'.format(parameter[1][5])
        new = self.path_result + path_dir
        return path_dir, new

    def get_r_square(self, fluorescence_blur):
        target = self.target
        sum_square_error = np.sum(np.power(target - fluorescence_blur, 2), axis=0)
        mean = np.mean(fluorescence_blur, axis=0)
        sum_square_total = np.sum(np.power(target - mean, 2), axis=0)
        r_square = 1 - sum_square_error / np.maximum(sum_square_total, 1e-6)
        return r_square

    def get_result(self, list_result):
        for k in range(8):
            if k < 6:
                list_result[k] = list_result[k].detach().cpu().numpy()[0, 0]
            else:
                list_result[k] = list_result[k].detach().cpu().numpy()[0]
        list_result[3] = np.mod(list_result[3], 180) / 180
        list_result[4] = np.mod(list_result[4], 90) / 180
        list_result.append(list_result[2] * list_result[0])
        list_result.append(list_result[2] * list_result[1])
        list_result.append(self.get_r_square(fluorescence_blur=list_result[7]))
        return list_result

    def get_mean(self, step, list_result, risk_empirical, risk_structural):
        r_square = list_result[10]
        foreground = np.where(r_square >= r_square.max() / 10, 1, 0).astype(np.float32)
        fore_area = foreground.sum()

        list_mean = [
            '{:.6f}'.format(np.sum(list_result[k] * foreground) / fore_area)
            for k in range(len(list_result))
        ]
        list_mean.append('{:.6f}'.format(risk_empirical.item()))
        list_mean.append('{:.6f}'.format(risk_structural.item()))
        list_mean.append('{:.6f}'.format(risk_empirical.item() + risk_structural.item()))
        list_mean.insert(0, step)
        self.add_text(list_mean)

    def result_writer(self, path_dir, list_result):
        for k in range(len(list_result)):
            io.imsave(self.path_result
                      + path_dir
                      + r'/{}.tif'.format(self.list_result_name[k]),
                      self.intensity * list_result[k])

    def exec(self):
        net = LsFPOMNet(
            origin=self.origin,
            channel=self.channel,
            theta=self.theta,
            weight=self.weight,
            weight_decay=self.weight_decay,
            orient_uniform_coef_sing=self.orient_uniform_coef_sing,
            orient_uniform_coef_doub=self.orient_uniform_coef_doub,
            orient_uniform_coef_sing0=self.orient_uniform_coef_sing,
            orient_uniform_coef_doub0=self.orient_uniform_coef_doub,
            total=self.total,
            alpha_sing=self.alpha_sing,
            alpha_doub=self.alpha_doub,
            background=self.background
        ).cuda()
        parameter_round = self.get_round()
        optimizer = self.get_optimizer(net=net)
        scheduler = self.get_scheduler(optimizer=optimizer)
        path_dir, new = self.get_path(parameter=parameter_round)

        for k in range(self.epoch + 1):
            optimizer.zero_grad()
            loss, list_result, risk_empirical, risk_structural = net()
            loss.backward()
            optimizer.step()
            if k > 0:
                if k % 10 == 0:
                    print(k, '{}%'.format(round(100 * (k + 1) / self.epoch, 2)),
                          'risk_total={}'.format(risk_empirical + risk_structural),
                          'risk_empirical={}'.format(risk_empirical),
                          'risk_structural={}'.format(risk_structural))
                if k % self.interval == 0:
                    list_result = self.get_result(list_result=list_result)
                    self.get_mean(step=k, list_result=list_result,
                                  risk_empirical=risk_empirical,
                                  risk_structural=risk_structural)
                if k == self.epoch:
                    self.log_writer(new=new)
                    self.result_writer(path_dir, list_result)

