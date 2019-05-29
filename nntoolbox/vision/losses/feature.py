from torch import nn
from fastai.callbacks import hook_outputs
from nntoolbox.vision.utils import gram_matrix

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts, base_loss = nn.L1Loss):
        '''
        Modified From https://github.com/hiromis/notes/blob/master/Lesson7.md
        Based on https://arxiv.org/pdf/1603.08155.pdf
        :param m_feat: model to extract feature from.
        :param layer_ids:
        :param layer_wgts:
        '''
        super().__init__()
        self.m_feat = m_feat(True).features.cuda().eval()
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

        self._base_loss = base_loss()

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self._base_loss(input, target)]
        self.feat_losses += [self._base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [self._base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()
