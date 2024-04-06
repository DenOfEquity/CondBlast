import gradio as gr

from modules import scripts
import modules.shared as shared
import modules as modules
from modules.prompt_parser import SdConditioning
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
import torch, math, random, time


#   shuffling and scaling text conds

class CondBlastForge(scripts.Script):
    def __init__(self):
        self.empty_uncond = None    #   specifically, a negative text conditioning
        self.empty_cond = None      #   specifically, a positive text conditioning

    def title(self):
        return "Cond Blastr"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label='Enabled')
            with gr.Row():
                shufflePos = gr.Slider(minimum=0.0, maximum=1.01, step=0.01, value=1.01, label='start step to shuffle positive text conds')
                shuffleNeg = gr.Slider(minimum=0.0, maximum=1.01, step=0.01, value=1.01, label='start step to shuffle negative text conds')
                
            with gr.Row():
                scalePos = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='positive prompt weight')
                scaleNeg = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='negative prompt weight')

        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("cb_enabled" in d))),
            (shufflePos, "cb_shufflePos"),
            (shuffleNeg, "cb_shuffleNeg"),
            (scalePos, "cb_scalePos"),
            (scaleNeg, "cb_scaleNeg"),
        ]

        return enabled, shufflePos, shuffleNeg, scalePos, scaleNeg


    def denoiser_callback(self, params):
#        torch.manual_seed(int(time.time()))     #   this is probably wrong, makes results non-deterministic as this seed will be lost
                                                #   but good for testing
        pos = params.text_cond
        neg = params.text_uncond
        is_SDXL = isinstance (pos, dict)
        #   positive conds
        if params.sampling_step >= self.shufflePos * params.total_sampling_steps:
            if is_SDXL:
                vector, cross = pos['vector'], pos['crossattn']
                indexes = torch.randperm(vector.size(1))
                params.text_cond['vector'] = vector[:, indexes]
                indexes = torch.randperm(cross.size(1))
                params.text_cond['crossattn'] = cross[:, indexes]
                del vector, cross, indexes
            else:
                indexes = torch.randperm(pos.size(1))
                params.text_cond = pos[:, indexes]
                del indexes

                if self.scalePos == 1.0:   #   filthy hack
                    self.scalePos = 0.999

        #negative conds
        if params.sampling_step >= self.shuffleNeg * params.total_sampling_steps:
            if is_SDXL:
                vector, cross = neg['vector'], neg['crossattn']
                indexes = torch.randperm(vector.size(1))
                params.text_uncond['vector'] = vector[:, indexes]
                indexes = torch.randperm(cross.size(1))
                params.text_uncond['crossattn'] = cross[:, indexes]
                del vector, cross, indexes
            else:
                indexes = torch.randperm(neg.size(1))
                params.text_uncond = neg[:, indexes]
                del indexes

                if self.scaleNeg == 1.0:
                    self.scaleNeg = 0.999


#for sdxl: works
#for sd: the shuffle works, but results don't change: seems like cached values?
#the lerp later makes it work (not if weight is 1.0, probably lerp function catches that case to avoid work?)

        pos = params.text_cond
        if self.scalePos != 1.0:
#            is_SDXL = isinstance (pos, dict)
            if is_SDXL:
                cond, cross = pos['vector'], pos['crossattn']
                empty_cond, empty_cross = self.empty_cond['vector'], self.empty_cond['crossattn']
                empty_cond.resize_as_(cond)
                empty_cross.resize_as_(cross)
                empty_cond = torch.reshape(empty_cond, cond.shape)
                empty_cross = torch.reshape(empty_cross, cross.shape)
                torch.lerp(empty_cond, cond, self.scalePos, out=cond)
                torch.lerp(empty_cross, cross, self.scalePos, out=cross)
                params.text_cond['vector'] = cond
                params.text_cond['crossattn'] = cross
                del cond, cross, empty_cond, empty_cross
            else:
                self.empty_cond.resize_as_(pos)
                self.empty_cond = torch.reshape(self.empty_cond, pos.shape)
                torch.lerp(self.empty_cond, pos, self.scalePos, out=params.text_cond)
        del pos

        neg = params.text_uncond
        if self.scaleNeg != 1.0:
#            is_SDXL = isinstance (neg, dict)
            if is_SDXL:
                uncond, cross = neg['vector'], neg['crossattn']
                empty_uncond, empty_cross = self.empty_uncond['vector'], self.empty_uncond['crossattn']
                empty_uncond.resize_as_(uncond)
                empty_cross.resize_as_(cross)
                empty_uncond = torch.reshape(empty_uncond, uncond.shape)
                empty_cross = torch.reshape(empty_cross, cross.shape)
                torch.lerp(empty_uncond, uncond, self.scaleNeg, out=uncond)
                torch.lerp(empty_cross, cross, self.scaleNeg, out=cross)
                params.text_uncond['vector'] = uncond
                params.text_uncond['crossattn'] = cross
                del uncond, cross, empty_uncond, empty_cross
            else:
                #   first, reduce the size to match uncond
                self.empty_uncond.resize_as_(neg)
                #   second, reshape to match uncond
                self.empty_uncond = torch.reshape(self.empty_uncond, neg.shape)
               #   third lerp; .text_uncond is fresh each step so no worries about repeat processing
                torch.lerp(self.empty_uncond, neg, self.scaleNeg, out=params.text_uncond)

        del neg



    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.


        enabled, shufflePos, shuffleNeg, scalePos, scaleNeg = script_args


        if not enabled:
            return


        self.shufflePos = shufflePos
        self.shuffleNeg = shuffleNeg
        self.scalePos = scalePos
        self.scaleNeg = scaleNeg

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(
            cb_enabled = enabled,
            cb_shufflePos = shufflePos,
            cb_shuffleNeg = shufflePos,
            cb_scalePos = scalePos,
            cb_scaleNeg = scaleNeg,
        ))

        prompt = SdConditioning([""], is_negative_prompt=True, width=params.width, height=params.height)
        self.empty_uncond = shared.sd_model.get_learned_conditioning(prompt)
        prompt = SdConditioning([""], is_negative_prompt=False, width=params.width, height=params.height)
        self.empty_cond = shared.sd_model.get_learned_conditioning(prompt)

        on_cfg_denoiser(self.denoiser_callback)

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return

