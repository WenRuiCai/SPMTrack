import math
from typing import Mapping, Any, Optional
from collections import OrderedDict
import torch
import torch.nn as nn


def tmoe_merge_state_dict(module: nn.Module, state_dict: Mapping[str, Any]) -> OrderedDict:
    state_dict = OrderedDict(**state_dict)
    expert_alpha = None
    use_rsexpert = False
    if 'expert_alpha' in state_dict:
        expert_alpha = state_dict['expert_alpha'].item()
        use_rsexpert = state_dict['use_rsexpert'].item()
        del state_dict['expert_alpha']
        del state_dict['use_rsexpert']
    for name in list(state_dict.keys()):
        if 'q.tmoe.A' in name:
            device = state_dict[name].device
            prefix = name[:-len('.q.tmoe.A')]
            qkv_module: nn.Linear = module.get_submodule(prefix)
            state_dict_has_linear_weight = prefix + '.q.tmoe.linear.weight' in state_dict
            state_dict_has_linear_bias = prefix + '.q.tmoe.linear.bias' in state_dict
            dim = qkv_module.in_features

            q_A = state_dict[prefix + '.q.tmoe.A']
            q_B = state_dict[prefix + '.q.tmoe.B']
            if state_dict_has_linear_weight:
                q_linear_weight = state_dict[prefix + '.q.tmoe.linear.weight']
            else:
                q_linear_weight = qkv_module.weight.data[:dim].to(device)
            q_merged_weight = _tmoe_merge(q_linear_weight, q_A, q_B, expert_alpha, use_rsexpert)
            has_tmoe_k = (prefix + '.k.tmoe.A') in state_dict
            if has_tmoe_k:
                k_A = state_dict[prefix + '.k.tmoe.A']
                k_B = state_dict[prefix + '.k.tmoe.B']
                if state_dict_has_linear_weight:
                    k_linear_weight = state_dict[prefix + '.k.tmoe.linear.weight']
                else:
                    k_linear_weight = qkv_module.weight.data[dim:2 * dim].to(device)
                k_merged_weight = _tmoe_merge(k_linear_weight, k_A, k_B, expert_alpha, use_rsexpert)
            else:
                k_linear_weight = qkv_module.weight.data[dim:2 * dim].to(device)
                k_merged_weight = k_linear_weight
            v_A = state_dict[prefix + '.v.tmoe.A']
            v_B = state_dict[prefix + '.v.tmoe.B']
            if state_dict_has_linear_weight:
                v_linear_weight = state_dict[prefix + '.v.tmoe.linear.weight']
            else:
                v_linear_weight = qkv_module.weight.data[2 * dim:].to(device)
            v_merged_weight = _tmoe_merge(v_linear_weight, v_A, v_B, expert_alpha, use_rsexpert)
            qkv_merged_weight = torch.cat((q_merged_weight, k_merged_weight, v_merged_weight), dim=0)
            state_dict[prefix + '.weight'] = qkv_merged_weight

            if state_dict_has_linear_bias:
                q_bias = state_dict[prefix + '.q.tmoe.linear.bias']
                k_bias = state_dict[prefix + '.k.tmoe.linear.bias']
                v_bias = state_dict[prefix + '.v.tmoe.linear.bias']
                qkv_merged_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                state_dict[prefix + '.bias'] = qkv_merged_bias

            if state_dict_has_linear_weight:
                del state_dict[prefix + '.q.tmoe.linear.weight']
                del state_dict[prefix + '.k.tmoe.linear.weight']
                del state_dict[prefix + '.v.tmoe.linear.weight']
            if state_dict_has_linear_bias:
                del state_dict[prefix + '.q.tmoe.linear.bias']
                del state_dict[prefix + '.k.tmoe.linear.bias']
                del state_dict[prefix + '.v.tmoe.linear.bias']

            del state_dict[prefix + '.q.tmoe.A']
            del state_dict[prefix + '.q.tmoe.B']
            if has_tmoe_k:
                del state_dict[prefix + '.k.tmoe.A']
                del state_dict[prefix + '.k.tmoe.B']
            del state_dict[prefix + '.v.tmoe.A']
            del state_dict[prefix + '.v.tmoe.B']
    for name in list(state_dict.keys()):
        if 'tmoe.A' in name:
            device = state_dict[name].device
            prefix = name[:-len('.tmoe.A')]
            state_dict_has_linear_weight = prefix + '.linear.weight' in state_dict
            state_dict_has_linear_bias = prefix + '.linear.bias' in state_dict
            if state_dict_has_linear_weight:
                linear_weight = state_dict[prefix + '.linear.weight']
            else:
                linear_weight = module.get_submodule(prefix).weight.data.to(device)
            A = state_dict[prefix + '.tmoe.A']
            B = state_dict[prefix + '.tmoe.B']
            merged_weight = _tmoe_merge(linear_weight, A, B, expert_alpha, use_rsexpert)
            state_dict[prefix + '.weight'] = merged_weight
            if state_dict_has_linear_bias:
                state_dict[prefix + '.bias'] = state_dict[prefix + '.linear.bias']
            if state_dict_has_linear_weight:
                del state_dict[prefix + '.linear.weight']
            if state_dict_has_linear_bias:
                del state_dict[prefix + '.linear.bias']
            del state_dict[prefix + '.tmoe.A']
            del state_dict[prefix + '.tmoe.B']
    return state_dict


def _tmoe_delta(expert_A: torch.Tensor, expert_B: torch.Tensor, alpha: Optional[float], use_rsexpert: bool) -> torch.Tensor:
    r = expert_A.size(0)
    if alpha is not None:
        if use_rsexpert:
            scaling = alpha / math.sqrt(r)
        else:
            scaling = alpha / r
    else:
        scaling = 1.
    return (expert_B @ expert_A) * scaling


def _tmoe_merge(weight: torch.Tensor, expert_A: torch.Tensor, expert_B: torch.Tensor, alpha: Optional[float], use_rsexpert: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _tmoe_delta(expert_A.to(torch.float32), expert_B.to(torch.float32), alpha, use_rsexpert)

    return (weight.to(torch.float32) + delta).to(original_dtype)


def _tmoe_unmerge(weight: torch.Tensor, expert_A: torch.Tensor, expert_B: torch.Tensor, alpha: Optional[float], use_rsexpert: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _tmoe_delta(expert_A.to(torch.float32), expert_B.to(torch.float32), alpha, use_rsexpert)

    return (weight.to(torch.float32) - delta).to(original_dtype)
