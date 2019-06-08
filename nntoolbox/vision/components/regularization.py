from torch import nn
import torch


class ShakeShakeLayer(nn.Module):
    def __init__(self, keep=False):
        super(ShakeShakeLayer, self).__init__()
        self._keep = keep

    def forward(self, branches, training):
        return ShakeShake.apply(branches, training, self._keep)


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, branches, training, keep):
        print(branches.shape)

        branch_weights = ShakeShake.get_branch_weights(len(branches), branches[0].shape[0], training)
        output = branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * branches

        if training and keep:
            ctx.save_for_backward(torch.ones(1), branch_weights)
        else:
            ctx.save_for_backward(torch.zeros(1), len(branches) * torch.ones(1).int())

        return torch.sum(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.saved_tensors[0] == 1:
            branch_weights = ctx.saved_tensors[1]
        else:
            cardinality = ctx.saved_tensors[1].item()
            branch_weights = ShakeShake.get_branch_weights(cardinality, grad_output.shape[0], True)

        return branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * grad_output, None, None

    @staticmethod
    def get_branch_weights(cardinality, batch_size, training):
        if training:
            branch_weights = torch.rand(size=(cardinality, batch_size))
            branch_weights /= torch.sum(branch_weights, dim=0, keepdim=True)
        else:
            branch_weights = 1.0 / cardinality * torch.ones(size=(cardinality, batch_size)).float()

        return branch_weights
