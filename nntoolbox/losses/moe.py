from torch import nn, Tensor
from typing import Tuple


__all__ = ['CompetitiveMOELoss']


class CompetitiveMOELoss(nn.Module):
    """
    Encourage expert specialization:

    loss(expert_op, expert_weight, target) = sum_e weight_e * base_loss(op_e, target)

    Reference:

    https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf
    """
    def __init__(self, base_loss: nn.Module=nn.MSELoss(reduction='none')):
        super(CompetitiveMOELoss, self).__init__()
        self.base_loss = base_loss
        setattr(self.base_loss, 'reduction', 'none')

    def forward(self, experts: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        """
        :param experts: expert_output: (batch_size, *, n_expert), expert_score: (batch_size, *, n_expert)
        :param targets: (batch_size, *)
        :return:
        """
        expert_outputs, expert_scores = experts
        targets = targets.unsqueeze(-1)
        loss = self.base_loss(expert_outputs, targets) # (batch_size, *, n_expert)
        return (loss * expert_scores).sum(-1)
