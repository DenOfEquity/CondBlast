# Cond Blast #
### extension for Forge webui for Stable Diffusion ###
---
## Basic usage ##
It's a simple UI. Select the sampling step when to start shuffling the text conditionings.
Also, you have the option to scale the text conditionings towards/away from an empty conditioning. This happens *after* any shuffling.
Settings used are saved with metadata, and restored from loading through the **PNG Info** tab.

---
## Advanced / Details ##
Steps to start shuffling are selected as a proportion of the way through the diffusion process: 0.0 is the first step, 1.0 is last, 1.01 is never. 
Each step from that point will receive a new shuffle.
Scaling the text conditionings is based on the idea on the idea of the [Negative Prompt Weight extension by muerrilla](https://github.com/muerrilla/stable-diffusion-NPW).

---
#### update 25/06/2024 ####
corrected location to write extra generation params, now they get added to *params.txt* too

added option to zero text conds after number of steps


---
## License ##
Public domain. Unlicense. Free to a good home.
All terrible code is my own. Use at your own risk, read the code.

> Written with [StackEdit](https://stackedit.io/).
