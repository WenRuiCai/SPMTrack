import torch
import copy


_default_config = {
    'block_chunks': 0,
    'init_values': 1.0e-05,
    'drop_path_uniform': True,
    'img_size': 518
}


def build_dino_v2_backbone(name: str, load_pretrained: bool, **kwargs):
    config = copy.deepcopy(_default_config)
    config.update(kwargs)

    if name == 'ViT-S/14':
        from . import vit_small
        model = vit_small(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'))
    elif name == 'ViT-B/14':
        from . import vit_base
        model = vit_base(patch_size=14, **config)
        if load_pretrained:
            state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth')
            #new_state_dict = {}
            #for k, v in state_dict.items():
            #    if k.startwith('blocks.') and '.mlp.fc' in k:
            #        new_k = k.replace('.mlp.fc', '.mlp.shared_experts.fc')
            #        new_state_dict[new_k] = v
            #    else:
            #        new_state_dict[k] = v
            model.load_state_dict(state_dict)
            #for block in model.blocks:
            #    for expert in block.mlp.experts:
            #        expert.initialize()
    elif name == 'ViT-L/14':
        from . import vit_large
        model = vit_large(patch_size=14, **config)
        if load_pretrained:
            state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth')
            #new_state_dict = {}
            #for k, v in state_dict.items():
            #    if k.startwith('blocks.') and '.mlp.fc' in k:
            #        new_k = k.replace('.mlp.fc', '.mlp.shared_experts.fc')
            #        new_state_dict[new_k] = v
            #    else:
            #        new_state_dict[k] = v
            model.load_state_dict(state_dict)
    elif name == 'ViT-g/14':
        from . import vit_giant2
        model = vit_giant2(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'))
    else:
        raise NotImplementedError(f'Unknown DINO v2 model name: {name}')
    return model
