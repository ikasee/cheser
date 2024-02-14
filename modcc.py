import torch
import torch.nn as nn
import torch.nn.init as init


class conv_block(nn.Module):
    def __init__(self, ic: int, oc: int, k_size: int, stride: int, pad: int):
        super(conv_block, self).__init__()

        self.oc = oc
        self.conv2d = nn.Conv2d(in_channels = ic, out_channels = oc, kernel_size = k_size, stride = stride, padding = pad)
        init.kaiming_uniform_(self.conv2d.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        self.batchNorm = nn.BatchNorm2d(num_features = oc)
        init.constant_(self.batchNorm.bias, 0)
        init.constant_(self.batchNorm.weight, 1)
        self.ac = nn.LeakyReLU()

    def forward(self, a):
        out = self.ac(self.batchNorm(self.conv2d(a)))

        return out

class convIN(nn.Module):
    def __init__(self, ic: int, oc: int, k_size: int, stride: int, pad: int):
        super(convIN, self).__init__()

        self.oc = oc
        self.conv2d = nn.Conv2d(in_channels = ic, out_channels = oc, kernel_size = k_size, stride = stride, padding = pad)
        init.kaiming_uniform_(self.conv2d.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        self.ac = nn.LeakyReLU()

    def forward(self, a):
        out = self.ac(self.conv2d(a))

        return out

class resIN(nn.Module):
    def __init__(self, ic: int, oc: int, k_size: int, stride: int, pad: int):
        super(resIN, self).__init__()

        self.CI = convIN(ic = ic, oc = oc, k_size = k_size, stride = stride, pad = pad)
        self.CII = convIN(ic = oc, oc = oc, k_size = k_size, stride = stride, pad = pad)

        self.match_dim = ic == self.CII.oc
        if not self.match_dim:
            self.dimer = nn.Conv2d(in_channels = ic, out_channels = self.CII.oc, kernel_size = 1, stride = 1, padding = 0)
            init.kaiming_uniform_(self.dimer.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')


    def forward(self, a: torch.tensor):

        if not self.match_dim:
            aRes = self.dimer(a)
        else:
            aRes = a

        outI = self.CI(a)
        outII = self.CII(outI)

        rCat = torch.add(outII, aRes)

        return rCat


class residual(nn.Module):
    def __init__(self, ic: int, oc: int, k_size: int, stride: int, pad: int, sc: bool):
        super(residual, self).__init__()

        self.CI = conv_block(ic = ic, oc = oc, k_size = k_size, stride = stride, pad = pad)
        self.CII = conv_block(ic = oc, oc = oc, k_size = k_size, stride = stride, pad = pad)

        self.match_dim = ic == self.CII.oc
        if not self.match_dim:
            self.dimer = nn.Conv2d(in_channels = ic, out_channels = self.CII.oc, kernel_size = 1, stride = 1, padding = 0)
            init.kaiming_uniform_(self.dimer.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')


    def forward(self, a: torch.tensor):

        if not self.match_dim:
            aRes = self.dimer(a)
        else:
            aRes = a

        outI = self.CI(a)
        outII = self.CII(outI)

        rCat = torch.add(outII, aRes)

        return rCat


class policy_head(nn.Module):
    def __init__(self):
        super(policy_head, self).__init__()

        self.ac = nn.LeakyReLU()

        self.batchNorm = nn.BatchNorm2d(num_features = 16)
        init.constant_(self.batchNorm.bias, 0)
        init.uniform_(self.batchNorm.weight, 1)

        self.fcI = nn.Linear(1024, 512)
        init.kaiming_uniform_(self.fcI.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        self.fcII = nn.Linear(512, 512)
        init.kaiming_uniform_(self.fcII.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        self.fcIII = nn.Linear(512, 256)
        init.kaiming_uniform_(self.fcIII.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        self.fc_out = nn.Linear(256, 60)
        init.kaiming_uniform_(self.fc_out.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
        # self.softmax = nn.Softmax(dim = -1)

    def forward(self, a):

        ups_out = self.ac(self.batchNorm(a))
        ups_out_flat = ups_out.view(ups_out.size(0), -1)

        out_policy = self.ac(self.fcII(self.ac(self.fcI(ups_out_flat))))
        out_policy = self.ac(self.fc_out(self.ac(self.fcIII(out_policy))))
        # soft_out = self.softmax(out_policy)

        return out_policy
        # return soft_out

class value_head(nn.Module):
    def __init__(self):
        super(value_head, self).__init__()

        self.batchNorm = nn.BatchNorm2d(num_features = 16)
        # self.max = nn.MaxPool2d(kernel_size = (4, 4), stride = 2, padding = 1)
        self.ac = nn.LeakyReLU()
        self.fcI = nn.Linear(1024, 512)
        self.fcII = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, a):

        ups_out = self.ac(self.batchNorm(a))
        ups_out_flat = ups_out.view(ups_out.size(0), -1)

        out_value = self.ac(self.fcII(self.ac(self.fcI(ups_out_flat))))
        out_value = self.ac(self.fc_out(out_value))

        return out_value

def t(input):
    tensor = torch.tensor(input, dtype = torch.float16, device = 'cuda')
    return tensor

def discount(reward, step, gamma):
    tot_reward = reward * (gamma ** step)

    return tot_reward

def p_gradient_loss(prob, reward):

    if not (prob == 0 or reward == 0):
        loss = -torch.sum(torch.log(prob) * reward)

    else:
        loss = torch.tensor(0., dtype = torch.float32, device = 'cuda')

    return loss


def ep_greedy(epsilon, policy):
    rand = torch.rand(1, device ='cuda')
    exploration = False

    if rand >= epsilon:
        choice_ind = torch.argmax(policy)

    else:
        choice_ind = torch.randint(0, len(policy), (1, 1), device = 'cuda')
        exploration = True

    choice_val = policy[choice_ind.item()]

    return choice_ind, choice_val, exploration