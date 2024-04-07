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

        is_SDXL = isinstance (params.text_cond, dict)

        if is_SDXL:
            pos = params.text_cond
            neg = params.text_uncond

            #   shuffle conds
            if params.sampling_step >= self.shufflePos * params.total_sampling_steps:
                vector, cross = pos['vector'][0], pos['crossattn'][0]
                indexes = torch.randperm(vector.size(0))
                params.text_cond['vector'][0] = vector[indexes]
                indexes = torch.randperm(cross.size(0))
                params.text_cond['crossattn'][0] = cross[indexes]
                del vector, cross, indexes

            if params.sampling_step >= self.shuffleNeg * params.total_sampling_steps:
                vector, cross = neg['vector'][0], neg['crossattn'][0]
                indexes = torch.randperm(vector.size(0))
                params.text_uncond['vector'][0] = vector[indexes]
                indexes = torch.randperm(cross.size(0))
                params.text_uncond['crossattn'][0] = cross[indexes]
                del vector, cross, indexes

            pos = params.text_cond
            neg = params.text_uncond

            #   weight
            if self.scalePos != 1.0:
                cond, cross = pos['vector'][0], pos['crossattn'][0]
                empty_cond, empty_cross = self.empty_cond['vector'], self.empty_cond['crossattn']
                empty_cond.resize_as_(cond)
                empty_cross.resize_as_(cross)
                empty_cond = torch.reshape(empty_cond, cond.shape)
                empty_cross = torch.reshape(empty_cross, cross.shape)
                torch.lerp(empty_cond, cond, self.scalePos, out=cond)
                torch.lerp(empty_cross, cross, self.scalePos, out=cross)
                params.text_cond['vector'][0] = cond
                params.text_cond['crossattn'][0] = cross
                del cond, cross, empty_cond, empty_cross
#            del pos

            if self.scaleNeg != 1.0:
                uncond, cross = neg['vector'][0], neg['crossattn'][0]
                empty_uncond, empty_cross = self.empty_uncond['vector'], self.empty_uncond['crossattn']
                empty_uncond.resize_as_(uncond)
                empty_cross.resize_as_(cross)
                empty_uncond = torch.reshape(empty_uncond, uncond.shape)
                empty_cross = torch.reshape(empty_cross, cross.shape)
                torch.lerp(empty_uncond, uncond, self.scaleNeg, out=uncond)
                torch.lerp(empty_cross, cross, self.scaleNeg, out=cross)
                params.text_uncond['vector'][0] = uncond
                params.text_uncond['crossattn'][0] = cross
                del uncond, cross, empty_uncond, empty_cross
#            del neg

            #   batch: copy shuffled/scaled results
            i = 1
            while i < len(params.text_cond['vector']):
                params.text_cond['vector'][i] = params.text_cond['vector'][0]
                params.text_cond['crossattn'][i] = params.text_cond['crossattn'][0]
                params.text_uncond['vector'][i] = params.text_uncond['vector'][0]
                params.text_uncond['crossattn'][i] = params.text_uncond['crossattn'][0]
                i += 1

        else:   #   not sdXL
            pos = params.text_cond[0]
            neg = params.text_uncond[0]

            #   shuffle
            if params.sampling_step >= self.shufflePos * params.total_sampling_steps:
                indexes = torch.randperm(pos.size(0))
                params.text_cond[0] = pos[indexes]
                del indexes

                if self.scalePos == 1.0:   #   filthy hack
                    self.scalePos = 0.999

            if params.sampling_step >= self.shuffleNeg * params.total_sampling_steps:
                indexes = torch.randperm(neg.size(0))
                params.text_uncond[0] = neg[indexes]
                del indexes

                if self.scaleNeg == 1.0:
                    self.scaleNeg = 0.999

            pos = params.text_cond[0]
            neg = params.text_uncond[0]

            #   weight
            if self.scalePos != 1.0:
                empty = self.empty_cond
                empty.resize_as_(pos)
                empty = torch.reshape(empty, pos.shape)
                torch.lerp(empty, pos, self.scalePos, out=params.text_cond[0])
                del empty
#            del pos

            if self.scaleNeg != 1.0:
                self.empty_uncond.resize_as_(neg)
                self.empty_uncond = torch.reshape(self.empty_uncond, neg.shape)
                torch.lerp(self.empty_uncond, neg, self.scaleNeg, out=params.text_uncond[0])
#            del neg

            #   batch: copy shuffled/scaled results
            i = 1
            while i < len(params.text_cond):
                params.text_cond[i] = params.text_cond[0]
                params.text_uncond[i] = params.text_uncond[0]
                i += 1



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
            cb_shuffleNeg = shuffleNeg,
            cb_scalePos = scalePos,
            cb_scaleNeg = scaleNeg,
        ))

        prompt = SdConditioning([""], is_negative_prompt=True, width=params.width, height=params.height)
        self.empty_uncond = shared.sd_model.get_learned_conditioning(prompt)
        prompt = SdConditioning([""], is_negative_prompt=False, width=params.width, height=params.height)
        self.empty_cond = shared.sd_model.get_learned_conditioning(prompt)

#   can't weight the conditionings here because of prompt editing?

        

        on_cfg_denoiser(self.denoiser_callback)

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return

