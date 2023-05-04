# Implementation of Self Attention Guidance in webui
https://arxiv.org/abs/2210.00939

For AUTOMATIC1111 webui:
at commit 22bcc7be

run the following command in root directory of webui:
```
git apply --ignore-whitespace extensions/sd_webui_SAG/0001-CFGDenoiser-and-script_callbacks-mod-for-SAG.patch
```

Demos with stealth pnginfo:
![xyz_grid-0014-232592377.png](resources%2Fimg%2Fxyz_grid-0014-232592377.png)
![xyz_grid-0001-232592377.png](resources%2Fimg%2Fxyz_grid-0001-232592377.png)