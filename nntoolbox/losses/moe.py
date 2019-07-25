from torch import nn, Tensor


__all__ = ['CompetitiveMOELoss']


class CompetitiveMOELoss(nn.Module):
    """
    Encourage expert specialization:

    loss(expert_op, expert_weight, target) = sum_e weight_e * base_loss(op_e, target)
    """
    def __init__(self, base_loss: nn.Module=nn.MSELoss(reduction='none')):
        super(CompetitiveMOELoss, self).__init__()
        self.base_loss = base_loss
        setattr(self.base_loss, 'reduction', 'none')

    def forward(self, expert_output: Tensor, expert_score: Tensor, targets: Tensor) -> Tensor:
        """
        :param expert_output: (batch_size, *, n_expert)
        :param expert_score: (batch_size, *, n_expert)
        :param targets: (batch_size, *)
        :return:
        """
        targets = targets.unsqueeze(-1)
        loss = self.base_loss(expert_output, targets) # (batch_size, *, n_expert)
        return (loss * expert_score).sum(-1)
