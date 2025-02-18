import torch
from torch import nn
import torch.nn.functional as F


class EIU(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(EIU, self).__init__()
        self.vis_project = nn.Sequential(nn.Conv1d(dim, key_channels, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
        self.image_lang_att = Attention(v_in_channels,
                                        l_in_channels, 
                                        key_channels,
                                        value_channels,
                                        out_channels=value_channels,
                                        num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask=None):
        vis = self.vis_project(x.permute(0, 2, 1))
        lang = self.image_lang_att(x, l, l_mask)
        lang = lang.permute(0, 2, 1)
        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)
        mm = mm.permute(0, 2, 1)
        return mm


class Attention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(Attention, self).__init__()
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask=None):
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)

        query = self.f_query(x)
        query = query.permute(0, 2, 1)
        key = self.f_key(l)
        value = self.f_value(l)

        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels) 
        out = out.permute(0, 2, 1)
        out = self.W(out)
        out = out.permute(0, 2, 1)

        return out

