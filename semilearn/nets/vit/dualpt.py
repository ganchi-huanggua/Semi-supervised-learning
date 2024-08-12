import torch
from torch import nn
from torch.nn import Module

from .vit import vit_tiny_patch2_32, vit_small_patch2_32, vit_small_patch16_224, vit_base_patch16_224, \
    vit_base_patch16_96, VisionTransformer
from ..utils import load_checkpoint


class DualPrompt(VisionTransformer):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=10,
                 **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         drop_path_rate=drop_path_rate, num_classes=num_classes)
        self.prompt_length = kwargs['prompt_length']
        self.insert_layers = kwargs['insert_layers']

        prompt_shape = (len(self.insert_layers), self.prompt_length, self.embed_dim)
        if kwargs['prompt_init'] == 'uniform':
            self.simclr_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.simclr_prompt, -1, 1)
            # self.ce_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # new cls token: 1, 1, 768
            self.ce_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.ce_prompt, -1, 1)
        self.projector = nn.Linear(768, 128)
        self.simclr_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.ce_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.noise_prompt = None

    def extract(self, x: torch.Tensor, simclr=False, ce=False, noise=False):

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if noise:
            if self.noise_prompt is None:
                noise_prompt = nn.Parameter(torch.randn(1, 12, 768), requires_grad=False).to(x.device)
                nn.init.uniform_(noise_prompt, -1, 1)
                # noise_prompt = nn.Parameter(torch.zeros(1, 12, 768), requires_grad=False).to(x.device)
                self.noise_prompt = noise_prompt.detach()

        if simclr:
            x = torch.cat([x[:, 0, :].unsqueeze(1),
                           self.simclr_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
                           x[:, 1:, :]], dim=1)
        if ce:
            # x = torch.cat((x, self.ce_prompt.expand(x.shape[0], -1, -1)), dim=1)
            x = torch.cat([x, self.ce_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1)], dim=1)
        # [1 (original cls token), 12 (simclr prompt), 196 (patched embedding), 1 (ce prompt / new cls token)] so far
        if noise:
            x = torch.cat([x[:, 0, :].unsqueeze(1),
                           self.noise_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
                           x[:, 1:, :]], dim=1)
        # print(x.shape)
        for i, block in enumerate(self.blocks):
            # if i == 7 and noise:
            #     x = torch.cat([x[:, 0, :].unsqueeze(1),
            #                    self.noise_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
            #                    x[:, 1:, :]], dim=1)
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

    def forward(self, x, only_feat=False, simclr=False, ce=False, projected=False, noise=False, **kwargs):

        x = self.extract(x, simclr=simclr, ce=ce, noise=noise)
        # if simclr and not ce:
        #     self.global_pool = 'token'
        # if simclr and ce:
        #     self.global_pool = 'new_token_mapped'
        if ce:
            self.global_pool = 'new_token_mapped'
        else:
            self.global_pool = 'token'

        if self.global_pool == 'avg':
            x = x[:, 1:].mean(dim=1)
        elif self.global_pool == 'token':
            x = x[:, 0]
        elif self.global_pool == 'new_token_mapped':
            x = x[:, -self.prompt_length:].mean(dim=1)

        x = self.fc_norm(x)

        if only_feat:
            if projected:
                x = self.projector(x)
            return x

        output = self.simclr_head(x)
        result_dict = {'logits': output, 'feat': x}
        return result_dict

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.ce_prompt.requires_grad = True
        self.simclr_prompt.requires_grad = True
        for param in self.ce_head.parameters():
            param.requires_grad = True
        for param in self.simclr_head.parameters():
            param.requires_grad = True
        # for param in self.head.parameters():
        #     param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True

        # fixme: ######################################################################
        # self.simclr_prompt.requires_grad = False

def dualpt_on_vit_base_patch_16_224(pretrained=True, pretrained_path='/home/lhz/code/semi-pt/vit_base_patch16_224'
                                                                     '.augreg2_in21k_ft_in1k/pytorch_model.bin',
                                    num_classes=1000):
    # insert_layers: 0 means shallow pt, [] means finetune a classifier
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = DualPrompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2,
                       num_classes=num_classes, **model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    model.freeze()
    return model


if __name__ == "__main__":
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = DualPrompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2,
                       num_classes=10, **model_kwargs)

