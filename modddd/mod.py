import torch
import torch.nn as nn
from modcc import residual, policy_head, value_head, resIN
import chess
import torch.nn.init as init

class cheser_null(nn.Module):
    def __init__(self):
        super().__init__()

        # board state input
        self.L_input = resIN(12, 64, 2, 1, 1)
        self.LLI = residual(64, 256, 2, 1, 1, True)
        # self.LLII = residual(128, 256, 3, 1, 1, True)
        # self.LLIII = residual(256, 256, 3, 1, 1, True)
        self.LLIV = residual(256, 512, 3, 1, 1, True)
        # self.GRU = nn.GRU(input_size = 256, hidden_size = 64)

        # legal move inputs
        self.M_input = resIN(120, 512, 2, 1, 1)
        # self.MLI = residual(256, 512, 3, 1, 1, True)
        self.MLII = residual(512, 1024, 2, 1, 1, True)
        self.MLIII = residual(1024, 512, 3, 1, 1, True)
        # self.MLIV = residual(512, 512, 3, 1, 1, True)

        # concatenated processing
        self.DLII = residual(1024, 512, 3, 1, 1, True)
        self.DLIII = residual(512, 128, 3, 1, 1, True)
        # self.DLIV = residual(256, 128, 3, 1, 1, True)
        self.resP = residual(128, 16, 3, 1, 1, True)
        # self.resV = residual(128, 16, 3, 1, 1, True)

        self.HPolicy = policy_head()
        #self.HValue = value_head()

    def forward(self, a, z, e):

        bbone_out_state = self.LLIV(self.LLI(self.L_input(a)))
        # bbone_out_state = self.GRU(bbone_out_state_flat)
        # bbone_out_state = self.LLIV(self.LLIII(bbone_out_state))

        bbone_out_lm = self.MLIII(self.MLII(self.M_input(z)))

        cat = torch.cat((bbone_out_state, bbone_out_lm), 1)

        bbone_out_decision = self.DLIII(self.DLII(cat))

        out_policy = self.HPolicy(self.resP(bbone_out_decision))
        # out_value = self.HValue(self.resV(bbone_out_decision))

        return out_policy # out_value
