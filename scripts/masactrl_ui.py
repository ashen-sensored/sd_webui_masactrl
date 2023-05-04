import base64
import enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from inspect import isfunction
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from modules.processing import StableDiffusionProcessing

import numpy as np
import cv2
import math


from modules import devices, script_callbacks

import modules.scripts as scripts
from modules import shared
import gradio as gr

import os

from scripts.masactrl_controller import MasaController, MasaControllerMode

masa_controller: Optional[MasaController] = None


def update_script_args(p, value, arg_idx, script_class):

    for s in scripts.scripts_txt2img.alwayson_scripts:
        if isinstance(s, script_class):
            args = list(p.script_args)
            # print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx - 1]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break





class Script(scripts.Script):

    def __init__(self):
        global masa_controller
        if masa_controller is None:
            masa_controller = MasaController(shared.sd_model.model.diffusion_model)
        pass
        self.xyzgraph_hooked_axis_modes = []
        self.xyzgraph_apply_tracking_index = -1
        self.xyzgraph_format_tracking_index = -1
    def title(self):
        return "Masa Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible





    def ui(self, is_img2img):
        with gr.Accordion('Masa Control', open=False):
            # arm_logging = gr.Checkbox(label="Arm Masa Logging", default=False)
            masactrl_mode = gr.Radio(label="Masa Control Mode", choices=[mode.name for mode in MasaControllerMode], value=str(MasaControllerMode.IDLE.name), interactive=True, type="index")
            with gr.Row():
                masa_start_step = gr.Number(label="Masa Start Step", value=5)
                masa_start_layer = gr.Number(label="Masa Start Layer", value=10)
                mask_threshold = gr.Slider(label="Mask Threshold", minimum=0.0, maximum=1.0, value=0.1, step=0.01)


            foreground_indexes_textbox = gr.Textbox(label="Foreground Indexes", value="2")
            # calculate_masks_button = gr.Button(value="Calculate Masks")

            # def calculate_masks_clicked(foreground_indexes):
            #     foreground_indexes = int(foreground_indexes)
            #     masa_controller.calculate_reconstruction_maps(foreground_indexes)
            #
            # calculate_masks_button.click(fn=calculate_masks_clicked, inputs=[foreground_indexes_textbox],outputs=None)
            hook_xyzgraph_button = gr.Button(value="Hook XYZ Graph")
            xyzgraph_hooked_mode_textbox = gr.Textbox(label="XYZ Graph Hooked Mode List", value="2,1,2")
            def apply_prompt_with_masa(p, x, xs):
                self.xyzgraph_apply_tracking_index += 1

                # current_axis_index = xs.index(x)
                current_axis_index = self.xyzgraph_apply_tracking_index
                current_masa_mode = self.xyzgraph_hooked_axis_modes[current_axis_index]
                update_script_args(p,current_masa_mode,0, self.__class__)
                if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
                    raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

                p.prompt = p.prompt.replace(xs[0], x)
                p.negative_prompt = p.negative_prompt.replace(xs[0], x)
                if self.xyzgraph_apply_tracking_index == len(self.xyzgraph_hooked_axis_modes) - 1:
                    self.xyzgraph_apply_tracking_index = -1

            def format_value_add_label(p, opt, x):
                self.xyzgraph_format_tracking_index += 1

                if type(x) == float:
                    x = round(x, 8)
                output = f"{x},\r\nMasaCtrl Mode: {MasaControllerMode(self.xyzgraph_hooked_axis_modes[self.xyzgraph_format_tracking_index]).name}"
                if self.xyzgraph_format_tracking_index == len(self.xyzgraph_hooked_axis_modes) - 1:
                    self.xyzgraph_format_tracking_index = -1
                return output


            def hook_xyzgraph_clicked(xyzgraph_hooked_mode_list):
                self.xyzgraph_hooked_axis_modes = [int(v) for v in xyzgraph_hooked_mode_list.split(",")]
                target_xyz_script_object = next(( v for v in scripts.scripts_txt2img.scripts if str(v).startswith('<xyz_grid.py.Script')), None)
                if target_xyz_script_object:

                    target_axis = next((v for v in target_xyz_script_object.current_axis_options if
                          v.label=="Prompt S/R"), None)
                    if target_axis:
                        target_axis.apply = apply_prompt_with_masa
                        target_axis.format_value = format_value_add_label



            hook_xyzgraph_button.click(fn=hook_xyzgraph_clicked, inputs=[xyzgraph_hooked_mode_textbox],outputs=None)


        return [masactrl_mode, masa_start_step, masa_start_layer, mask_threshold, foreground_indexes_textbox]



    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        masactrl_mode, masa_start_step, masa_start_layer, mask_threshold, foreground_indexes_textbox = args
        match masactrl_mode:
            case MasaControllerMode.LOGGING | MasaControllerMode.RECON:
                masa_controller.mode_init(masactrl_mode, int(masa_start_step), int(masa_start_layer), mask_threshold)
            case MasaControllerMode.IDLE:
                pass







        return

    def postprocess(self, p, processed, *args):
        masactrl_mode, masa_start_step, masa_start_layer, mask_threshold, foreground_indexes_textbox = args
        match masactrl_mode:
            case MasaControllerMode.LOGGING | MasaControllerMode.RECON:
                masa_controller.mode_end(masactrl_mode, foreground_indexes_textbox)
            case MasaControllerMode.IDLE:
                pass
