# A Simple Guide of How to Use This Plugin

## Disclaimer
There are no explainations of the inner workings of Masa Control here. This is only a guide for anyone who has been breaking the plugin very often and want to know how to use it properly.

## Basic usage
There are 4 modes in this plugin, described below: 
- `Idle`: Do nothing (Plugin turned off)
- `Log`: Log the current generation detail
- `Recon`: Generate with the guide of the log from the previous logging session
- `LogRecon`: Doing Log and Recon at the same time

Basic steps to use the plugin are as belows:
1. Set whatever generation parameter you want (models, prompts, seeds, etc.)
2. Go to Masa Ctrl UI and switch mode to `Log`
3. Enter the foreground index into the UI (method described by @ashen-sensored)
4. Generate as usual and tweak untill you get the base image you want
5. (Important) Press `Calculate Mask` to calculate mask for guidance
6. Add other generation constraints (prompts, guide images, etc.) 
7. (Optional) Set `Masa Start Step` and `Masa Start Layer`
8. Generate and observer the magic unleashes

If you'd like the subsequent image to be guided by the last generated one, set mode to `LogRecon` instead of `Recon` when generating. However, don't forget to press `Calculate Mask` after each successive run to calculate the mask for it.

## Important notes
Some things to know before using the UI:
- Masa Ctrl is used to preserve the forground object, and is not designed to use for background swapping
- Max foreground index is 75 (The foreground prompt should not have more than 75 tokens) and the foreground prompt should always be in the very front of the prompt text
- The image size cannot be changed once the guidance image is logged


## FAQ
1. Getting an `Index Error: Index xxx is out of bounds for dimension o with size xxx` when generating.    
Check if your foreground prompt is longer than 75 tokens. Try reducing them down by stripping out unecessary white spaces and keywords until only less than 75 tokens are left. Therere are currently no workaround for this.
Because of the constraints of Stable Diffusion, the model can only take up to 75 tokens for text guidance. The UI splits the prompt into many chunks (each with maxium size of 75) and then merges them back together later on. If there are too many prompt tokens, the ui will be unable to retrieve the tokens as they are splitted.
2. `RuntimeError: The size of tensor a (xxx) must match the size of tensor b (ooo) at non-singleton dimension 3`.    
Make sure you did not change the image dimension after logging. Masa Control can only work with same-dimension images.
3. `KeyError: '-1'`.    
Press `Reset` button to reset the plugin's state. The data logged by the plugin won't be lost and you can directly regenerate without logging again after resolving the issue.
4. Can I use T2A and ControlNet with Masa Control?    
Absoulutely yes. This methodology is designed to be compatible with pretty much any other constraints out there. Feel free to experiment with them as long as you don't get an OOM (Out of Memory).
