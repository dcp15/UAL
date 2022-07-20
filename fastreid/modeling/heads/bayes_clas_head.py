from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY
import copy


@REID_HEADS_REGISTRY.register()
class BayesHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            test_time_ens
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat
        self.feat_dim = feat_dim
        self.test_time_ens = test_time_ens

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self._make_bayes()
        self.reset_parameters()

    def _make_bayes(self):
        in_dim = self.feat_dim
        if in_dim == 1024:
            bayes_dim0, bayes_dim1, fcn_dim = 512, 512, 256
        elif in_dim == 2048:
            bayes_dim0, bayes_dim1, fcn_dim = 1024, 512, 1024
        self.bayes_head = Bayes_head(in_dim=in_dim, mid_dim=[bayes_dim0, bayes_dim1], out_dim=in_dim, p=self.p)
        self.mean_head = FCN_head(in_dim=in_dim, mid_dim=[bayes_dim0, bayes_dim1], out_dim=in_dim)
        self.var_head = FCN_head(in_dim=in_dim, mid_dim=[fcn_dim], out_dim=in_dim)

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        scale = cfg.MODEL.HEADS.SCALE
        margin = cfg.MODEL.HEADS.MARGIN
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        test_time_ens = cfg.MODEL.HEADS.TEST_TIME_ENS

        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type,
            'test_time_ens': test_time_ens
        }

    def _reset_weight(self):
        self.bayes_models = [self.bayes_head]
        self.test_state = []
        for i in range(self.test_time_ens):
            tmp = []
            for m in self.bayes_models:
                m.reset_weight()
                tmp.append(copy.deepcopy(m.state_dict()))
            self.test_state.append(tmp)

    def load_bayes_model(self, index):
        tmp = self.test_state[index]
        count = 0
        for m in self.bayes_models:
            m.load_state_dict(tmp[count])
            count += 1

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        # bayes

        if self.training:
            features = self.bayes_head(features)
            feat_mean = self.mean_head(features)
            feat_var = torch.log1p(torch.exp(self.var_head(features)))
            features = feat_mean / feat_var.clamp(min=1e-10)
            g_var = feat_var.mean(dim=3).mean(dim=2)

            pool_feat = self.pool_layer(features)
            neck_feat = self.bottleneck(pool_feat)
            neck_feat = neck_feat[..., 0, 0]
        else:
            pool_feat = []
            neck_feat = []
            for i in range(self.test_time_ens):
                self.load_bayes_model(i)
                tmp_features = self.bayes_head(features)
                feat_mean = self.mean_head(tmp_features)
                feat_var = torch.log1p(torch.exp(self.var_head(tmp_features)))
                tmp_features = feat_mean / feat_var

                tmp_pool_feat = self.pool_layer(tmp_features)
                tmp_neck_feat = self.bottleneck(tmp_pool_feat)
                tmp_neck_feat = tmp_neck_feat[..., 0, 0]
                pool_feat.append(tmp_pool_feat.unsqueeze(dim=0))
                neck_feat.append(tmp_neck_feat.unsqueeze(dim=0))
            pool_feat = torch.cat(pool_feat, dim=0).mean(dim=0)
            neck_feat = torch.cat(neck_feat, dim=0).mean(dim=0)

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':
            feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after':
            feat = neck_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
            "g_var": g_var
        }


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl


class Bayes_Gaussion_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(Bayes_Gaussion_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = 1.0

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.01,
                'posterior_mu_initial': (0, 0.01),
                'posterior_rho_initial': (-4, 0.01),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_test = Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
            self.bias_test = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_rho', None)

        self.train_sample_nums = 5
        self.test_sample_nums = 5

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def reset_weight(self):
        weight_test, bias_test = self.sample_weight(self.test_sample_nums)
        self.weight_test.zero_()
        self.weight_test += weight_test
        if self.bias is not None:
            self.bias_test.zero_()
            self.bias_test += bias_test
        else:
            self.bias_test = None

    def sample_weight(self, nums):
        weight, bias = 0, None

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        normal = torch.distributions.Normal(loc=self.weight, scale=self.scale * weight_sigma)
        for i in range(nums):
            weight += normal.rsample()
        weight /= nums

        if self.bias is not None:
            bias = 0
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            normal = torch.distributions.Normal(loc=self.bias, scale=self.scale * bias_sigma)
            for i in range(nums):
                bias += normal.rsample()
            bias /= nums
        return weight, bias

    def forward(self, input):
        if self.training:
            weight, bias = self.sample_weight(self.train_sample_nums)
        else:
            # weight, bias = self.sample_weight(self.test_sample_nums)
            weight, bias = self.weight_test, self.bias_test
            # weight, bias = self.weight, self.bias
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.weight, weight_sigma)
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias, bias_sigma)
        return kl


class Bayes_Dropout_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, prob=0.5):

        super(Bayes_Dropout_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prob = prob

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_test = Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_test = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.train_sample_nums = 1
        self.test_sample_nums = 1

        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_weight(self):
        weight_test, bias_test = self.sample_weight(self.test_sample_nums)
        self.weight_test.zero_()
        self.weight_test += weight_test
        if self.bias is not None:
            self.bias_test.zero_()
            self.bias_test += bias_test
        else:
            self.bias_test = None

    def sample_weight(self, nums):
        weight = 0
        if self.bias is not None:
            bias = 0
        for i in range(nums):
            bernolli = torch.distributions.Bernoulli(probs=self.prob)
            weight += self.weight * bernolli.sample(self.weight.shape).to(self.weight.device)
            weight *= 0.7 / self.prob
            if self.bias is not None:
                bias += self.bias * bernolli.sample(self.bias.shape).to(self.bias.device)
                bias *= 0.7 / self.prob
            else:
                bias = None
        weight /= nums
        if self.bias is not None:
            bias /= nums
        return weight, bias

    def forward(self, input):
        if self.training:
            weight, bias = self.sample_weight(self.train_sample_nums)
        else:
            weight, bias = self.weight_test, self.bias_test
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        return torch.zeros(1).cuda()


class Bayes_head(nn.Module):
    def __init__(self, in_dim=1920, mid_dim=[1024, 512], out_dim=1024, p=0.7):
        super(Bayes_head, self).__init__()
        BN_MOMENTUM = 0.1
        dim_list = [in_dim] + mid_dim + [out_dim]
        self.layer = nn.Sequential()
        self.bayes_count = len(dim_list) - 1
        self.bayes_index = []
        self.p = p
        # self.p = 0.95
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]
            self.bayes_index.append(i * 3)
            # self.layer.add_module('Bayes_{}'.format(i),
            #                       Bayes_Gaussion_Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
            #                                             stride=1, padding=0))
            self.layer.add_module('Bayes_{}'.format(i),
                                  Bayes_Dropout_Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                                       stride=1, padding=0, prob=self.p))
            self.layer.add_module('BN_{}'.format(i), nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM))
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

    def reset_weight(self):
        self.my_modules(self)

    def my_modules(self, models):
        for name, module in models._modules.items():
            if module is None:
                continue
            if type(module) is Bayes_Gaussion_Conv2d or type(module) is Bayes_Dropout_Conv2d:
                module.reset_weight()
            if len(module._modules.items()) == 0:
                continue
            else:
                self.my_modules(module)

    def uncertainty_state_dict(self):
        return {'layer': self.layer.state_dict()}

    def load_uncertainty_modules(self, pth):
        self.layer.load_state_dict(pth['layer'])

    def kl_loss(self):
        kl_loss = torch.zeros(1).cuda()
        for i in self.bayes_index:
            kl_loss += self.layer[i].kl_loss()
        return kl_loss


class FCN_head(nn.Module):
    def __init__(self, in_dim=1920, mid_dim=[256], out_dim=1024):
        super(FCN_head, self).__init__()
        BN_MOMENTUM = 0.1
        dim_list = [in_dim] + mid_dim + [out_dim]
        self.layer = nn.Sequential()
        self.bayes_count = len(dim_list) - 1
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]
            self.layer.add_module('Conv_{}'.format(i),
                                  nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                            stride=1, padding=0))
            self.layer.add_module('BN_{}'.format(i), nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM))
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)
