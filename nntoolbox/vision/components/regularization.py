from torch import nn
import torch


class ShakeShake(nn.Module):
    '''
    Implement shake-shake regularizer:
    y = x + sum_i alpha_i branch_i
    (alpha_i > 0 are random variables such that sum_i alpha_i = 1)
    At test time:
    y = x + 1 / n_branch sum_i branch_i
    Based on https://arxiv.org/abs/1705.07485
    '''
    def __init__(self, keep='shake'):
        super(ShakeShake, self).__init__()
        self._keep = keep

    def forward(self, branches, training):
        return ShakeShakeFunction.apply(branches, training, self._keep)


class ShakeShakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, branches, training, mode):
        '''
        :param ctx: context (to save info for backward pass)
        :param branches: outputs of all branches concatenated (cardinality, batch_size, n_channel, h, w)
        :param training: boolean, true if is training
        :param mode: 'keep': keep the forward weights for backward; 'even': backward with 1/n_branch weight;
                      'shake': randomly choose new weights
        :return: weighted sum of all branches' outputs
        '''
        print(branches.shape)

        branch_weights = ShakeShakeFunction.get_branch_weights(len(branches), branches[0].shape[0], training)
        output = branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * branches

        if training:
            if mode == 'keep':
                ctx.save_for_backward(torch.ones(1), branch_weights)
            elif training and mode == 'even':
                ctx.save_for_backward(-torch.ones(1), len(branches) * torch.ones(1).int())
            else: # shake mode
                ctx.save_for_backward(torch.zeros(1), len(branches) * torch.ones(1).int())

        return torch.sum(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.saved_tensors[0] == 1: # keep mode
            branch_weights = ctx.saved_tensors[1]
        elif ctx.saved_tensors[0] == -1: # even mode:
            cardinality = ctx.saved_tensors[1].item()
            branch_weights = ShakeShakeFunction.get_branch_weights(cardinality, grad_output.shape[0], False)
        else: # shake mode
            cardinality = ctx.saved_tensors[1].item()
            branch_weights = ShakeShakeFunction.get_branch_weights(cardinality, grad_output.shape[0], True)

        return branch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * grad_output, None, None

    @staticmethod
    def get_branch_weights(cardinality, batch_size, training):
        if training:
            branch_weights = torch.rand(size=(cardinality, batch_size))
            branch_weights /= torch.sum(branch_weights, dim=0, keepdim=True)
        else:
            branch_weights = 1.0 / cardinality * torch.ones(size=(cardinality, batch_size)).float()

        return branch_weights