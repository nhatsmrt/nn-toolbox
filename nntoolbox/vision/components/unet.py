"""
Modified version of UNet that allows for decoder extraction and a little bit more customization
"""
from fastai.torch_core import *
from fastai.layers import *
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
    """
    Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`.
    Use hook instead of .orig
    """

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


def custom_conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[nn.Module]=nn.BatchNorm2d,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type is not None
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(norm_type(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)



def custom_res_block(nf, dense: bool = False, norm_type: Optional[nn.Module] = nn.BatchNorm2d, bottle: bool = False,
              **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(custom_conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        custom_conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


class CustomPixelShuffle_ICNR(nn.Module):
    """"
    Upsample by `scale` from `ni` filters to `nf` (default `ni`),
    using `nn.PixelShuffle`, `icnr` init, and `weight_norm`.
    """

    def __init__(self, ni: int, nf: int = None, scale: int = 2, blur: bool = False, norm_type=nn.BatchNorm2d,
                 leaky: float = None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class CustomUnetBlock(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, normalization=batchnorm_2d, **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        self.bn = normalization(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, norm_type=normalization, **kwargs)
        self.conv2 = custom_conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, norm_type=normalization, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class DynamicUnetV2(SequentialEx):
    """
    Modified version of UNet that allows for decoder extraction
    """
    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, self_attention:bool=False,
                 normalization=batchnorm_2d, y_range:Optional[Tuple[float,float]]=None,
                 last_cross:bool=True, bottle:bool=False, **kwargs):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        input_hook = InputHook(encoder[0], True)
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(custom_conv_layer(ni, ni * 2, norm_type=normalization, **kwargs),
                                    custom_conv_layer(ni * 2, ni, norm_type=normalization, **kwargs)).eval()
        x = middle_conv(x)
        self.encoder = [encoder]
        self.decoder = [normalization(num_features=ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = CustomUnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur, self_attention=sa,
                                   normalization=normalization, **kwargs).eval()
            self.decoder.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: self.decoder.append(CustomPixelShuffle_ICNR(ni, norm_type=normalization, **kwargs))
        if last_cross:
            self.decoder.append(CustomMergeLayer(input_hook=input_hook, dense=True))
            ni += in_channels(encoder)
            self.decoder.append(custom_res_block(ni, bottle=bottle, norm_type=normalization, **kwargs))
            self.decoder += [custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=normalization, **kwargs)]
        if y_range is not None: self.decoder.append(SigmoidRange(*y_range))
        super().__init__(*(self.encoder + self.decoder))

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

    def get_decoder(self):
        return SequentialEx(*self.decoder)
