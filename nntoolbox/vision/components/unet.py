"""
Modified version of UNet that allows for decoder extraction
"""
from fastai.torch_core import *
from fastai.layers import *
from fastai.vision.models.unet import *
from fastai.callbacks.hooks import *
from nntoolbox.hooks import InputHook


__all__ = ['DynamicUnetV2']


def _get_sfs_idxs(sizes:Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class CustomMergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, input_hook: InputHook, dense: bool=False, remove_store: bool=True):
        super().__init__()
        self.dense = dense
        self.input_hook = input_hook
        self.remove_store = remove_store

    def forward(self, x):
        op = torch.cat([x, self.input_hook.store], dim=1) if self.dense else (x + self.input_hook.store)
        if self.remove_store:
            self.input_hook.store = None
        return op


class DynamicUnetV2(SequentialEx):
    """
    Modified version of UNet that allows for decoder extraction
    """

    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None,
                 last_cross:bool=True, bottle:bool=False, **kwargs):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        input_hook = InputHook(encoder[0], True)
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        self.encoder = [encoder]
        self.decoder = [batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=blur, self_attention=sa,
                                   **kwargs).eval()
            self.decoder.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: self.decoder.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            self.decoder.append(CustomMergeLayer(input_hook=input_hook, dense=True))
            ni += in_channels(encoder)
            self.decoder.append(res_block(ni, bottle=bottle, **kwargs))
            self.decoder += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: self.decoder.append(SigmoidRange(*y_range))
        super().__init__(*(self.encoder + self.decoder))

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

    def get_decoder(self):
        return SequentialEx(*self.decoder)
