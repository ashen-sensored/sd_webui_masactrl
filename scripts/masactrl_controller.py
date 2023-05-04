
from __future__ import annotations

import enum
from inspect import isfunction

from diffusers.utils import deprecate
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import torch
from ldm.util import default
from modules.hypernetworks import hypernetwork
from modules import shared
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import os
import math
import numpy as np

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class ProxyReconMasaSattn(object):
    def __init__(self, controller: MasaController, module_key: str, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None

        self.attached = False
        self.controller = controller
        self.module_key = module_key



    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'attached', 'controller', 'module_key'] and self.attached:
            return getattr(self.org_module, attr)




    def attach(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    # implementation from diffusers
    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=3):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                (
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def forward(self, x, context=None, mask=None):

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
            masa_active = self.controller.query_masa_active(self.module_key)
            if masa_active:
                batch_size, sequence_length, inner_dim = x.shape
                masa_mask, masa_kv, masa_mask_threshold = self.controller.retrieve_masa_info_suite(self.module_key)
                # interpolate and convert to binary mask
                scale_factor = np.sqrt(sequence_length / masa_mask.shape[-1] / masa_mask.shape[-2])
                scaled_mask_shape = (int(masa_mask.shape[-2] * scale_factor), int(masa_mask.shape[-1] * scale_factor))
                scaled_mask = F.interpolate(masa_mask.unsqueeze(0).unsqueeze(0),
                                            (scaled_mask_shape[0], scaled_mask_shape[1])).flatten()

                # # this is original implementation for reference, behavior for fg_mask is not ideal
                # scaled_mask[scaled_mask >= masa_mask_threshold] = 1
                # scaled_mask[scaled_mask < masa_mask_threshold] = 0
                # fg_mask = scaled_mask.masked_fill(scaled_mask == 0, -float('inf'))
                # bg_mask = scaled_mask.masked_fill(scaled_mask == 1, -float('inf'))

                fg_attn_mask = torch.zeros_like(scaled_mask)
                fg_attn_mask[scaled_mask < masa_mask_threshold] = torch.finfo(masa_kv['k_in'].dtype).min

                bg_attn_mask = torch.zeros_like(scaled_mask)
                bg_attn_mask[scaled_mask >= masa_mask_threshold] = torch.finfo(masa_kv['k_in'].dtype).min

                fg_sattn_out = self.masa_scaled_dot_product_attention_forward(x, context, fg_attn_mask, masa_kv['k_in'], masa_kv['v_in'])
                bg_sattn_out = self.masa_scaled_dot_product_attention_forward(x, context, bg_attn_mask, masa_kv['k_in'], masa_kv['v_in'])

                fg_binary_mask = torch.ones_like(scaled_mask)
                fg_binary_mask[scaled_mask < masa_mask_threshold] = 0

                masa_sattn_out = fg_sattn_out * fg_binary_mask.unsqueeze(-1) + bg_sattn_out * (1 - fg_binary_mask.unsqueeze(-1))
                return masa_sattn_out
            else:
                return self.masa_scaled_dot_product_attention_forward(x, context, mask)

    def masa_scaled_dot_product_attention_forward(self, x, context=None, mask=None, external_k_in=None, external_v_in=None):
        batch_size, sequence_length, inner_dim = x.shape
        h = self.heads
        head_dim = inner_dim // h

        if mask is not None:
            mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
            if len(mask.shape) == 1 and mask.shape[0] == sequence_length:
                # we are getting a slice of the mask covering sequence_length, need to repeat in all other dimensions
                mask = mask.unsqueeze(-1).repeat(batch_size, h, 1, sequence_length)
            else:
                mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])


        q_in = self.to_q(x)

        if mask is not None:
            mask = mask.to(q_in.dtype)

        if external_k_in is None or external_v_in is None:
            context = default(context, x)
            context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
            k_in = self.to_k(context_k)
            v_in = self.to_v(context_v)
        else:
            # be aware that hypernetworks will have no effect
            k_in = external_k_in
            v_in = external_v_in


        q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
        k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
        v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

        del q_in, k_in, v_in

        dtype = q.dtype
        if shared.opts.upcast_attn:
            q, k, v = q.float(), k.float(), v.float()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class ProxyLoggedCrossAttn(object):
    def __init__(self, controller: MasaController, module_key: str, org_module: torch.nn.Module = None, is_xattn=False):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None

        self.attached = False
        self.controller = controller
        self.module_key = module_key
        self.is_xattn = is_xattn


    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'attached', 'controller', 'module_key'] and self.attached:
            return getattr(self.org_module, attr)




    def attach(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if not self.is_xattn:
            sattn_data_suite = {'k_in': k, 'v_in': v}
            self.controller.report_sattn(self.module_key, sattn_data_suite)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # if self.is_xattn:
        #     del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        if self.is_xattn:
            xattn_data_suite = {'sim': sim}
            self.controller.report_xattn(self.module_key, xattn_data_suite)




        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    # def scaled_dot_product_attention_forward(self, x, context=None, mask=None):
    #     batch_size, sequence_length, inner_dim = x.shape
    #
    #     if mask is not None:
    #         mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
    #         mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])
    #
    #     h = self.heads
    #     q_in = self.to_q(x)
    #     context = default(context, x)
    #
    #     context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    #     k_in = self.to_k(context_k)
    #     v_in = self.to_v(context_v)
    #
    #     q_t, k_t, v_t = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    #     with torch.autocast(enabled=False, device_type='cuda'):
    #         q_t, k_t = q_t.float(), k_t.float()
    #         sim = einsum('b i d, b j d -> b i j', q_t, k_t) * self.scale
    #
    #     head_dim = inner_dim // h
    #     q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    #     k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    #     v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    #
    #
    #
    #     del q_in, k_in, v_in
    #
    #     dtype = q.dtype
    #     if shared.opts.upcast_attn:
    #         q, k, v = q.float(), k.float(), v.float()
    #
    #
    #
    #     # the output of sdp = (batch, num_heads, seq_len, head_dim)
    #     hidden_states = torch.nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    #     )
    #
    #
    #
    #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    #
    #
    #
    #     if self.is_xattn:
    #         xattn_report_data_dict = {'v': v, 'hidden_states': hidden_states}
    #         self.controller.report_xattn(self.module_key, xattn_report_data_dict)
    #     else:
    #         sattn_report_data_dict = {'k': k, 'v': v}
    #         self.controller.report_sattn(self.module_key, sattn_report_data_dict)
    #
    #     # Compute the transpose of 'v'
    #     v_transpose = torch.transpose(v_t, 1, 2)
    #
    #     # Calculate the product of 'v' and its transpose
    #     vvT = torch.matmul(v_t, v_transpose)
    #
    #     # Compute the pseudo-inverse of the 'vvT'
    #     inv_vvT = torch.inverse(vvT.cpu().to(torch.float32))
    #
    #     hidden_states_re = rearrange(hidden_states, 'b n (h d) -> (b h) n d', h=h)
    #     # Calculate the product of 'out' and the pseudo-inverse
    #     # sim_recovered = torch.matmul(hidden_states_re.cpu().to(torch.float32), inv_vvT)
    #     sim_recovered = torch.einsum('ikj,ilk->ikl', hidden_states_re.cpu().to(torch.float32), inv_vvT)
    #
    #     # calculate loss between recovered sim and original sim
    #     loss = torch.nn.functional.mse_loss(sim_recovered, sim.cpu())
    #
    #
    #     hidden_states = hidden_states.to(dtype)
    #
    #     # linear proj
    #     hidden_states = self.to_out[0](hidden_states)
    #
    #
    #
    #
    #
    #
    #
    #     # dropout
    #     hidden_states = self.to_out[1](hidden_states)
    #     return hidden_states


class ProxyMasaUNetModel(object):
    def __init__(self, controller:MasaController, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.attached = False
        self.controller = controller




    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'attached', 'controller'] and self.attached:
            return getattr(self.org_module, attr)

    def attach(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.controller.masa_unet_signal(x, timesteps)
        return self.org_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)

aggregate_xattn_map_selected_module_keys = ['input_blocks.7.1.transformer_blocks.0.attn2', 'input_blocks.8.1.transformer_blocks.0.attn2', 'output_blocks.3.1.transformer_blocks.0.attn2', 'output_blocks.4.1.transformer_blocks.0.attn2', 'output_blocks.5.1.transformer_blocks.0.attn2']

class MasaControllerMode(enum.IntEnum):
    LOGGING = 0
    RECON = 1
    LOGRECON = 2
    IDLE = 3


class MasaController:
    def __init__(self, ori_unet: UNetModel):
        self.monitoring_xattn_modules = {}
        self.monitoring_sattn_modules = {}
        self.logged_xattn_map_data_suite = {}
        self.logged_sattn_data_suite = {}
        self.proxy_xattn_modules = {}
        self.proxy_sattn_modules = {}
        self.proxy_recon_sattn_mmodules = {}

        self.recording_mode = True
        self.current_timestep: float = -1.0
        self.current_latent_size = (0,0)
        self.unet_proxy = ProxyMasaUNetModel(self, ori_unet)
        self.recon_sattn_map_data_suite = {}
        self.mode = MasaControllerMode.LOGGING
        self.start_timestep = 900.0
        self.start_layer = 10
        self.recon_mask_threshold = 0.1
        for name, module in ori_unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                if 'attn2' in name:
                    self.proxy_xattn_modules[name] = ProxyLoggedCrossAttn(self, name, module, True)

                elif 'attn1' in name:
                    self.proxy_sattn_modules[name] = ProxyLoggedCrossAttn(self, name, module)
                    self.proxy_recon_sattn_mmodules[name] = ProxyReconMasaSattn(self, name, module)

        self.log_recon = False




    def logging_attach_all(self):
        for name, module in self.proxy_xattn_modules.items():
            module.attach()
        for name, module in self.proxy_sattn_modules.items():
            module.attach()
        self.unet_proxy.attach()

    def logging_detach_all(self):
        for name, module in self.proxy_xattn_modules.items():
            module.detach()
        for name, module in self.proxy_sattn_modules.items():
            module.detach()
        self.unet_proxy.detach()

    def logging_attach_xattn(self):
        for name, module in self.proxy_xattn_modules.items():
            module.attach()

    def logging_detach_xattn(self):
        for name, module in self.proxy_xattn_modules.items():
            module.detach()






    def report_xattn(self, name, xattn_map_data_dict):
        timestep_str_key = str(self.current_timestep)
        self.logged_xattn_map_data_suite[timestep_str_key][name] = xattn_map_data_dict

    def report_sattn(self, name, sattn_map_data_dict):
        timestep_str_key = str(self.current_timestep)
        self.logged_sattn_data_suite[timestep_str_key][name] = sattn_map_data_dict


    def recon_attach_all(self):
        layer_idx = 0
        for name, module in self.proxy_recon_sattn_mmodules.items():
            layer_idx += 1
            if layer_idx < self.start_layer:
                continue
            module.attach()

        self.unet_proxy.attach()

    def recon_detach_all(self):

        for name, module in self.proxy_recon_sattn_mmodules.items():
            module.detach()
        self.unet_proxy.detach()

    def retrieve_sattn_mask(self, name):
        return self.recon_sattn_map_data_suite[self.current_timestep]

    def query_masa_active(self, name):
        return self.current_timestep <= self.start_timestep

    def retrieve_masa_info_suite(self, key):
        current_mask = self.recon_sattn_map_data_suite[str(self.current_timestep)]
        current_kv = self.logged_sattn_data_suite[str(self.current_timestep)][key]
        return current_mask, current_kv, self.recon_mask_threshold


    def masa_unet_signal(self, x, timesteps):
        self.current_timestep = timesteps[0].item()
        timestep_str_key = str(self.current_timestep)
        self.current_latent_size = x.shape[-2:]
        if self.mode == MasaControllerMode.LOGGING:
            if timestep_str_key not in self.logged_xattn_map_data_suite:
                self.logged_xattn_map_data_suite[timestep_str_key] = {}
                self.logged_sattn_data_suite[timestep_str_key] = {}
            else:
                print('warning: overwriting existing logged data, this should not happen')

    def calculate_reconstruction_maps(self, foreground_token_idx):
        if self.logged_xattn_map_data_suite:
            print('Calculating mask from logged xattn maps...')
            reconstruction_xattn_timestep_map_dict = {}
            for timestep_str_key in self.logged_xattn_map_data_suite.keys():

                xattn_maps_of_interest = [
                    self.logged_xattn_map_data_suite[timestep_str_key][module_key]['sim'][:, ...,
                    foreground_token_idx] for module_key in aggregate_xattn_map_selected_module_keys]
                for i in range(len(xattn_maps_of_interest)):
                    attn_map = xattn_maps_of_interest[i]
                    # only interested in cond map
                    attn_map, _ = attn_map.chunk(2, dim=0)  # (head_count,N)
                    # mean along head dim
                    attn_map = attn_map.mean(0)
                    # xattn_maps_of_interest[i] = attn_map
                    res_h, res_w = self.current_latent_size
                    xattn_maps_of_interest[i] = attn_map.reshape(math.ceil(res_h/4), math.ceil(res_w/4))

                attn_maps_aggregate = torch.stack(xattn_maps_of_interest, dim=0).mean(0)

                maps_min = attn_maps_aggregate.min()
                maps_max = attn_maps_aggregate.max()
                final_map = (attn_maps_aggregate - maps_min) / (maps_max - maps_min)
                reconstruction_xattn_timestep_map_dict[timestep_str_key] = final_map

                print(f'Processed timestep {timestep_str_key}...')

            self.recon_sattn_map_data_suite = reconstruction_xattn_timestep_map_dict
            del self.logged_xattn_map_data_suite
            self.logged_xattn_map_data_suite = {}
    def mode_init(self, mode:MasaControllerMode, masa_start_step=5, masa_start_layer=10, mask_threshold=0.1):
        self.mode = mode
        match mode:
            case MasaControllerMode.LOGGING:
                self.logging_attach_all()
            case MasaControllerMode.RECON | MasaControllerMode.LOGRECON:

                # order matters because of start_layer

                self.recon_params_init(masa_start_step, masa_start_layer, mask_threshold)
                self.recon_attach_all()
                if mode == MasaControllerMode.LOGRECON:
                    self.log_recon = True

    def recon_params_init(self, masa_start_step, masa_start_layer,mask_threshold):
        self.start_timestep = float(list(self.recon_sattn_map_data_suite.keys())[masa_start_step])
        self.start_layer = masa_start_layer
        self.recon_mask_threshold = mask_threshold


    def mode_end(self, mode:MasaControllerMode, foreground_indexes=None):
        match mode:
            case MasaControllerMode.LOGGING:
                self.logging_detach_all()
                self.calculate_reconstruction_maps(int(foreground_indexes))
            case MasaControllerMode.RECON:
                self.recon_detach_all()





