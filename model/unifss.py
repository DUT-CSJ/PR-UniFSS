r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res#, extract_feat_chossed, extract_feat_vgg_dense
from .base.correlation import Correlation

from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from .eiu import EIU
import math
from mmseg.ops import resize
from einops import rearrange
import numpy as np
from .base.internimage import HSCU, InternImageLayer

import open_clip
import segmentation_models_pytorch as smp 
from collections import OrderedDict

from .base.conv4d import CenterPivotConv4d as Conv4d

class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        self.feature_affinity = [False, True, True]

        self.proj_query_feat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 64, 1),
                nn.ReLU(),
            ) if self.feature_affinity[0] else nn.Identity(),
            nn.Sequential(
                nn.Conv2d(512, 32, 1),
                nn.ReLU(),
            ) if self.feature_affinity[1] else nn.Identity(),
            nn.Sequential(
                nn.Conv2d(256, 16, 1),
                nn.ReLU(),
            ) if self.feature_affinity[2] else nn.Identity()
        ])

        self.dropout2d = nn.Dropout2d(p=0.5)
        decoder_dim = [
            (128 + 32) if True else 128,
            (96 + 32) if self.feature_affinity[1] else 96,
            (48 + 16) if self.feature_affinity[2] else 48
        ]

        # self.pwam = EIU(dim=1024, v_in_channels=1024, l_in_channels=1024, key_channels=256, value_channels=256,
                #  num_heads=8, dropout=0.5) # resnet50
        self.pwam = EIU(dim=512, v_in_channels=512, l_in_channels=512, key_channels=256, value_channels=256,
                 num_heads=8, dropout=0.5) # resnet101

        self.visual_proj = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.visual_proj2 = nn.Sequential(
                InternImageLayer(channels=128+64, groups=8),
                InternImageLayer(channels=128+64, groups=8),
                nn.Conv2d(128+64, 64, 1)
            )
        self.visual_proj3 = nn.Sequential(
            InternImageLayer(channels=128 + 64, groups=8),
            InternImageLayer(channels=128 + 64, groups=8),
            nn.Conv2d(128 + 64, 32, 1)
        )

        self.conv_decoder = nn.ModuleList([
            nn.Sequential(
                InternImageLayer(channels=decoder_dim[0], groups=8),
                InternImageLayer(channels=decoder_dim[0], groups=8),
                nn.Conv2d(decoder_dim[0], 96, 1)
            ),
            nn.Sequential(
                InternImageLayer(channels=decoder_dim[1], groups=8),
                InternImageLayer(channels=decoder_dim[1], groups=8),
                nn.Conv2d(decoder_dim[1], 48, 1)
            ),
            nn.Sequential(
                InternImageLayer(channels=decoder_dim[2], groups=4),
                InternImageLayer(channels=decoder_dim[2], groups=4),
                nn.Conv2d(decoder_dim[2], 32, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(True),
                nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True)
            )
        ])

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode='nearest') for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [F.interpolate(x, size=size, mode='nearest') for x, size in zip(feats, sizes)]
        return recoverd_feats

    def forward(self, hypercorr_pyramid, query_feats, visual_feat, text_feat):
        _, query_feat4, query_feat3, query_feat2 = query_feats
        query_feat4, query_feat3, query_feat2 = [
            self.proj_query_feat[i](x) for i, x in enumerate((query_feat4, query_feat3, query_feat2))
        ]
        query_feat4, query_feat3 = self.apply_dropout(self.dropout2d, query_feat4, query_feat3)

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # --------------------------------------------------------------------------------------------------------
        feat_4d4 = hypercorr_sqz4.mean(dim=(-2, -1))
        hw_shape = (12, 12)
        visual_feat = self.visual_proj(nlc_to_nchw(self.pwam(nchw_to_nlc(visual_feat), text_feat.unsqueeze(-1)), hw_shape))
        feat_4d4 = self.visual_proj2(torch.cat((feat_4d4, visual_feat), 1))
        feat_4d4 = F.interpolate(feat_4d4, size=(25, 25), mode='bilinear', align_corners=True)
        # --------------------------------------------------------------------------------------------------------

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        feat_4d3 = hypercorr_mix43.mean(dim=(-2, -1))
        visual_feat = self.visual_proj3(torch.cat((feat_4d4, feat_4d3), 1))
        visual_feat = F.interpolate(visual_feat, size=(50, 50), mode='bilinear', align_corners=True)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # Decode the encoded 4D-tensor
        x = hypercorr_mix432.mean(dim=(-2, -1))  # 50, 50
        x = self.conv_decoder[0](torch.cat((x, visual_feat), dim=1) if True else x)
        x = F.interpolate(x, size=(50, 50), mode='bilinear', align_corners=True)
        x = self.conv_decoder[1](torch.cat((x, query_feat3), dim=1) if self.feature_affinity[1] else x)
        x = F.interpolate(x, size=(100, 100), mode='bilinear', align_corners=True)
        logit_mask = self.conv_decoder[2](torch.cat((x, query_feat2), dim=1) if self.feature_affinity[2] else x)

        return logit_mask


def extract_feat_chossed_clip(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    # feat = backbone.relu.forward(feat)
    feat = backbone.act1.forward(feat)
    # feat = backbone.relu1.forward(feat)
    feat = backbone.conv2.forward(feat)
    feat = backbone.bn2.forward(feat)
    # feat = backbone.relu2.forward(feat)
    feat = backbone.act2.forward(feat)
    feat = backbone.conv3.forward(feat)
    feat = backbone.bn3.forward(feat)
    # feat = backbone.relu3.forward(feat)
    feat = backbone.act3.forward(feat)

    # feat = backbone.maxpool.forward(feat)
    feat = backbone.avgpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        # feat = backbone.__getattr__('layer%d' % lid)[bid].relu1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].act1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        # feat = backbone.__getattr__('layer%d' % lid)[bid].relu2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].act2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].avgpool.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        # feat = backbone.__getattr__('layer%d' % lid)[bid].relu3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].act3.forward(feat)

    return feats

def extract_feat_chossed(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    # feat = backbone.relu.forward(feat)
    # feat = backbone.act1.forward(feat)
    feat = backbone.relu1.forward(feat)
    feat = backbone.conv2.forward(feat)
    feat = backbone.bn2.forward(feat)
    feat = backbone.relu2.forward(feat)
    feat = backbone.conv3.forward(feat)
    feat = backbone.bn3.forward(feat)
    feat = backbone.relu3.forward(feat)

    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        # feat = backbone.__getattr__('layer%d' % lid)[bid].act1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        # feat = backbone.__getattr__('layer%d' % lid)[bid].act2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        # feat = backbone.__getattr__('layer%d' % lid)[bid].act3.forward(feat)

    return feats


def convert_model2float32(model):
    for param in model.parameters():
        param.data = param.data.float()

from skimage import measure
class UniFSS(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(UniFSS, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        if backbone == 'resnet50':
            self.clip = open_clip.create_model('RN50', pretrained='openai', device='cuda')
            self.backbone = self.clip.visual
            self.backbone.eval()
            self._init_conv()
            self.clip.visual = None
            self.tokenizer = open_clip.get_tokenizer('RN50')
            self.feat_ids = list(range(3, 17))  # (4, 17)
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.clip = open_clip.create_model('RN101', pretrained='openai', device='cuda')
            self.backbone = self.clip.visual
            self.backbone.eval()
            self._init_conv()
            self.clip.visual = None
            self.tokenizer = open_clip.get_tokenizer('RN101')
            self.feat_ids = list(range(3, 34))  # (4, 17)
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        if self.backbone_type == 'resnet50':
            self.hpn_learner = HPNLearner([6+2, 12, 8])  # resnet dense
        if self.backbone_type == 'resnet101':
            self.hpn_learner = HPNLearner([6+2, 46, 8])  # resnet dense
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.scale = 0

        self.affinty1 = HSCU(channels=12 * 12, groups=12 * 12)
        self.affinty2 = HSCU(channels=25 * 25, groups=25 * 25, mlp_ratio=2)
    def _init_conv(self):
        self.conv1 = nn.Conv2d(self.backbone.attnpool.v_proj.in_features,
                               self.backbone.attnpool.v_proj.out_features,
                               kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(self.backbone.attnpool.c_proj.in_features,
                               self.backbone.attnpool.c_proj.out_features,
                               kernel_size=(1, 1))
        conv1_weight_shape = (*self.backbone.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.backbone.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.backbone.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.backbone.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.backbone.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.backbone.attnpool.c_proj.bias))
        self.conv1.eval()
        self.conv2.eval()
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

    def forward(self, query_img, support_img, support_mask, class_name):
        def find_bbox(mask):
            mask = mask.float()
            nonzero_corrds = torch.where(mask == 1)
            min_x = torch.min(nonzero_corrds[1])
            min_y = torch.min(nonzero_corrds[0])
            max_x = torch.max(nonzero_corrds[1])
            max_y = torch.max(nonzero_corrds[0])
            return min_x, min_y, max_x, max_y
        def draw_rectangle(mask, bbox):
            b, h, w = mask.shape
            batch_images = torch.zeros_like(mask.float())
            for i in range(b):
                # min_x, min_y, max_x, max_y = bbox[i]
                batch_box = bbox[i]
                for j in range(len(batch_box)):
                    min_x, min_y, max_x, max_y = batch_box[j]
                    batch_images[i, min_y:max_y+1, min_x:max_x+1] = 1
            return batch_images
        def find_connect_area(mask):
            batch_bbox = []
            for i in range(mask.shape[0]):
                label_mask, num_components = measure.label(mask[i].cpu(), connectivity=1, return_num=True)
                bbox = []
                for connected_label in range(1, num_components + 1):
                    component_corrds = torch.where(torch.from_numpy(label_mask) == connected_label)
                    min_x = torch.min(component_corrds[1])
                    min_y = torch.min(component_corrds[0])
                    max_x = torch.max(component_corrds[1])
                    max_y = torch.max(component_corrds[0])
                    bbox.append((min_x, min_y, max_x, max_y))
                batch_bbox.append(bbox)
            return batch_bbox
        # -----------------------------------------box FSS----------------------------------------------
        # bbox_batch = find_connect_area(support_mask)
        # # bbox_batch = [find_bbox(mask) for mask in support_mask]
        # support_box = draw_rectangle(support_mask, bbox_batch)
        # support_mask = support_box
        # -----------------------------------------box FSS----------------------------------------------

        # -----------------------------------------mask FSS----------------------------------------------
        support_img2 = torch.zeros_like(support_img)
        support_img2[:, 0, :, :] = support_img[:, 0, :, :] * support_mask
        support_img2[:, 1, :, :] = support_img[:, 1, :, :] * support_mask
        support_img2[:, 2, :, :] = support_img[:, 2, :, :] * support_mask
        # -----------------------------------------mask FSS----------------------------------------------

        # -----------------------------------------image FSS----------------------------------------------
        # support_img = query_img.clone()
        # support_mask = torch.ones_like(support_mask)
        # -----------------------------------------image FSS----------------------------------------------

        # -----------------------------------------text FSS----------------------------------------------
        # support_img2 = support_img.clone()
        # -----------------------------------------text FSS----------------------------------------------
        with torch.no_grad():
            # intermediate feature extraction for resnet
            if self.backbone_type == 'resnet50' or 'resnet101':
                support_feats = extract_feat_chossed_clip(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                query_feats_dense = extract_feat_chossed_clip(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_dense = extract_feat_chossed_clip(support_img2, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                corr_dense = Correlation.multilayer_correlation_dense(query_feats_dense[-self.stack_ids[-1]:],
                                                                      support_feats_dense[-self.stack_ids[-1]:],
                                                                      self.stack_ids)
                corr_self_simi = Correlation.multilayer_correlation_dense(query_feats_dense[-self.stack_ids[-1]:],
                                                                          support_feats[-self.stack_ids[-1]:],
                                                                          self.stack_ids)

            query_feat_last = self.backbone.layer4[2].act3.forward(query_feats_dense[-1].clone())
            support_feat_last = self.backbone.layer4[2].act3.forward(support_feats_dense[-1].clone())
            query_feat_last = self.conv2(self.conv1(query_feat_last))
            support_feat_last = self.conv2(self.conv1(support_feat_last))

            visual_feat = query_feat_last.clone()

            # -----------------------------------------Class-aware FSS textual feat----------------------------------------------
            # text = self.tokenizer(class_name).cuda()
            # # text = class_name.squeeze(1)
            # text_feat = self.clip.encode_text(text)
            # text_decoder = text_feat.clone()
            # text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 12, 12)
            # -----------------------------------------Class-aware FSS textual feat----------------------------------------------

            # -----------------------------------------Support image as textual feat----------------------------------------------
            text_feat = self.backbone.layer4[2].act3.forward(support_feats[-1].clone())
            text_feat = self.conv2(self.conv1(text_feat))
            text_feat = Weighted_GAP(text_feat, F.interpolate(support_mask.unsqueeze(1), size=(text_feat.size(2), text_feat.size(3)), mode='bilinear', align_corners=True))
            text_feat = text_feat.squeeze(-1).squeeze(-1)
            text_decoder = text_feat.clone()
            text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 12, 12)
            # -----------------------------------------Support image as textual feat----------------------------------------------

            bsz, ch, hb, wb = text_feat.size()
            text_feat = text_feat.view(bsz, ch, -1)
            text_feat = text_feat / (text_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            bsz, ch, hb, wb = support_feat_last.size()
            support_feat_last = support_feat_last.view(bsz, ch, -1)
            support_feat_last = support_feat_last / (support_feat_last.norm(dim=1, p=2, keepdim=True) + 1e-5)

            bsz, ch, ha, wa = query_feat_last.size()
            query_feat_last = query_feat_last.view(bsz, ch, -1)
            query_feat_last = query_feat_last / (query_feat_last.norm(dim=1, p=2, keepdim=True) + 1e-5)

            corr = torch.bmm(query_feat_last.transpose(1, 2), support_feat_last).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0).unsqueeze(1)
            corrt = torch.bmm(query_feat_last.transpose(1, 2), text_feat).view(bsz, ha, wa, hb, wb)
            corrt = corrt.clamp(min=0).unsqueeze(1)


        for i in range(len(corr_dense)):
            corr_dense[i] = torch.cat([corr_dense[i], corr_self_simi[i]], dim=1)
        corr_self_simi = None
        corr_dense[0] = torch.cat((corr_dense[0], corr), dim=1)
        b, c, h1, w1, h2, w2 = corr_dense[0].size()
        corr_dense[0] = self.affinty1(corr_dense[0].reshape(b * c, h1, w1, h2 * w2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, c, h1, w1, h2, w2)
        b, c, h1, w1, h2, w2 = corr_dense[1].size()
        corr_dense[1] = self.affinty2(corr_dense[1].reshape(b * c, h1, w1, h2 * w2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, c, h1, w1, h2, w2)
        corr_dense[0] = torch.cat((corr_dense[0], corrt), dim=1)
        del corr_self_simi, support_feats, corr, support_feats_dense, corrt
        with torch.no_grad():
            query_feats_dense = self.extract_last(self.stack_feats(query_feats_dense))

        logit_mask = self.hpn_learner(corr_dense, query_feats_dense, visual_feat, text_decoder)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def stack_feats(self, feats):

        feats_l4 = torch.stack(feats[-self.stack_ids[0]:]).transpose(0, 1)
        feats_l3 = torch.stack(feats[-self.stack_ids[1]:-self.stack_ids[0]]).transpose(0, 1)
        feats_l2 = torch.stack(feats[-self.stack_ids[2]:-self.stack_ids[1]]).transpose(0, 1)
        feats_l1 = torch.stack(feats[:-self.stack_ids[2]]).transpose(0, 1)

        return [feats_l4, feats_l3, feats_l2, feats_l1]

    def predict_mask_nshot(self, batch, nshot, dataset=None):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx],
                              batch['class_name'])
            # logit_mask = self(batch['query_img'], batch['query_img'], batch['support_masks'][:, s_idx],
            #                   batch['class_name'])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        logit_mask = F.interpolate(logit_mask, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
        dice = self.dice_loss(logit_mask, gt_mask.long())
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask) + dice

    def train_mode(self):
        self.train()
        self.clip.eval()
        self.conv1.eval()
        self.conv2.eval()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging


class MaxPool4d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=True)

    def forward(self, x):
        """
        x: Hyper correlation.
            shape: B L H_q W_q H_s W_s
        """
        B, L, H_q, W_q, H_s, W_s = x.size()
        x = rearrange(x, 'B L H_q W_q H_s W_s -> (B H_q W_q) L H_s W_s')
        x = self.pool(x)
        x = rearrange(x, '(B H_q W_q) L H_s W_s -> B L H_q W_q H_s W_s', H_q=H_q, W_q=W_q)
        return x

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
