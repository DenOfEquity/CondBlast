# Cond Blast #
### extension for Forge webui for Stable Diffusion ###
---
## Basic usage ##
It's a simple UI. Experiment. Changes to Negative are most useful.

Can improve details and contrast.

Settings used are saved with metadata, and restored from loading through the **PNG Info** tab.

---
## Details ##
Scaling the text conditionings is based on the idea of the [Negative Prompt Weight extension by muerrilla](https://github.com/muerrilla/stable-diffusion-NPW).

Idea to add noise from [this person with many names](https://github.com/thderoo). Adding noise to the negative (text unconds) is essentially the idea behind Independant Condition Guidance.

Order of actions is: empty (overrides others), blend (neg only), noise, shuffle, weight.

---
#### 11/01/2025 ####
various updates

#### 03/07/2024 ####
* options to add noise

#### 25/06/2024 ####
* corrected location to write extra generation params, now they get added to *params.txt* too
* added option to zero text conds after number of steps


---
## License ##
Public domain. Unlicense. Free to a good home.
All terrible code is my own. Use at your own risk, read the code.


