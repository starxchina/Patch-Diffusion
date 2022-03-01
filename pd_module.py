import torch
import torch.nn as nn

class Patch_Diffusion(nn.Module):
    def __init__(self, input_feature_channel, patch_channel, patch_num, l=1):
        """Patch Diffusion Moudle

        Args:
            input_feature_channel (int): the channel number of input feature
            patch_channel (int): length of patch vector
            patch_num (int): number of patches
            l (int): number of diffusions. Defaults to 1.
        """
        super(Patch_Diffusion, self).__init__()
        self.input_feature_channel = input_feature_channel
        self.patch_channel = patch_channel
        self.patch_num = patch_num
        self.l = l
        self.psi = nn.Conv2d(input_feature_channel, patch_channel, kernel_size=1)
        self.rho = nn.Conv2d(input_feature_channel, input_feature_channel, kernel_size=1)

        modules = []
        for i in range(l):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv1d(input_feature_channel, input_feature_channel, kernel_size=1, bias=False))  # W
        self.diffusion = nn.ModuleList(modules)

        self.bn = nn.BatchNorm2d(input_feature_channel, eps=1e-04)

    def forward(self, x):
        b, c, h, w = x.size()
        patch = self.psi(x).view(b, self.patch_channel, -1).permute(0, 2, 1)
        gram_mat = torch.matmul(patch, patch.permute(0,2,1))
        denominator_mid = torch.sqrt(torch.sum(patch.pow(2), dim=2).view(b, -1, 1))
        denominator = torch.matmul(denominator_mid, denominator_mid.permute(0,2,1)) + 1e-08
        attention_mat = gram_mat / denominator
        x_graph = self.rho(x).view(b, self.input_feature_channel, -1)

        for i in range(self.l):
            x_graph = torch.matmul(attention_mat, x_graph.permute(0,2,1).contiguous()).permute(0,2,1)
            x_graph = self.diffusion[i*2](x_graph)
            x_graph = self.diffusion[i*2+1](x_graph)

        x_out = x_graph.view(b, c, h, w)
        out = x + self.bn(x_out)
        return out, patch