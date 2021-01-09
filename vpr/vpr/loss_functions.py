import torch
from vpr.common.loss_interface import Loss


def kld(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class VPRLoss(Loss):
    def __init__(self):
        super(VPRLoss, self).__init__()

        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, anneal, *model_output):
        score, mask, mu, logvar = model_output

        n_llk = - torch.sum(self.logsigmoid(score) * mask) / mask.sum()
        kld_i = kld(mu, logvar)

        kld_contrib = torch.sigmoid(anneal * kld_i)
        loss = n_llk + kld_contrib

        return loss
