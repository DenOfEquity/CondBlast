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
                shufflePos = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='shuffle positive text conds after step')
                shuffleNeg = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='shuffle negative text conds after step')
            with gr.Row():
                noisePos = gr.Slider(minimum=0.0, maximum=0.2, step=0.001, value=0.0, label='add noise to positive text conds')
                noiseNeg = gr.Slider(minimum=0.0, maximum=0.2, step=0.001, value=0.0, label='add noise to negative text conds')
            with gr.Row():
                scalePos = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='positive prompt weight')
                scaleNeg = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='negative prompt weight')
            with gr.Row():
                zeroPos  = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='empty positive after step')
                zeroNegS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='empty negative before step')
                zeroNegE = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='empty negative after step')

        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("cb_enabled" in d))),
            (shufflePos, "cb_shufflePos"),
            (shuffleNeg, "cb_shuffleNeg"),
            (noisePos, "cb_noisePos"),
            (noiseNeg, "cb_noiseNeg"),
            (scalePos, "cb_scalePos"),
            (scaleNeg, "cb_scaleNeg"),
            (zeroPos, "cb_zeroPos"),
            (zeroNegS, "cb_zeroNegS"),
            (zeroNegE, "cb_zeroNegE"),
        ]

        return enabled, shufflePos, shuffleNeg, noisePos, noiseNeg, scalePos, scaleNeg, zeroPos, zeroNegS, zeroNegE
#add empty neg before step
#shuffle pos continues to be ineffective
    @torch.no_grad()
    def denoiser_callback(self, params):
        is_SDXL = isinstance (params.text_cond, dict)
        lastStep = params.total_sampling_steps - 1
        batchSize = len(params.text_cond['vector']) if is_SDXL else len(params.text_cond)
        
#   is it correct to shuffle/weight the cross-attention?

#shuffle messes up sdxl - maybe clips are cat'd, shuffle blends across?
#mat crossattn need to match

        if is_SDXL:
            # positive conds
            if self.zeroPos < 1.0 or self.shufflePos < 1.0 or self.noisePos > 0.0 or self.scalePos != 1.0:
                cross = params.text_cond['crossattn'][0]
                empty_cross = self.empty_cond['crossattn'].resize_as_(cross).reshape(cross.shape)
                if self.zeroPos * lastStep < params.sampling_step:
                    cross = empty_cross
                else:
                    if self.shufflePos * lastStep < params.sampling_step:
                        indexes = torch.randperm(cross.size(0))
                        cross = cross[indexes]
                        del indexes
                    if self.noisePos > 0.0:
                        noise = torch.randn_like(cross)
                        torch.lerp(cross, noise, self.noisePos, out=cross)
                        del noise
                    if self.scalePos != 1.0:
                        torch.lerp(empty_cross, cross, self.scalePos, out=cross)
                del empty_cross
                if batchSize == 1:
                    params.text_cond['crossattn'][0] = cross
                else:
                    params.text_cond['crossattn'] = cross.repeat(batchSize)
                del cross
            
            # negative conds
            if self.zeroNegS > 0.0 or self.zeroNegE < 1.0 or self.shuffleNeg < 1.0 or self.noiseNeg > 0.0 or self.scaleNeg != 1.0:
                cross = params.text_uncond['crossattn'][0]
                empty_cross = self.empty_uncond['crossattn'].resize_as_(cross).reshape(cross.shape)
                if self.zeroNegS * lastStep > params.sampling_step or self.zeroNegE * lastStep < params.sampling_step:
                    cross = empty_cross
                else:
                    #   shuffle
                    if self.shuffleNeg * lastStep < params.sampling_step:
                        indexes = torch.randperm(cross.size(0))
                        cross = cross[indexes]
                        del indexes
                    #noise
                    if self.noiseNeg > 0.0:
                        noise = torch.randn_like(cross)
                        torch.lerp(cross, noise, self.noiseNeg, out=cross)
                        del noise
                    #   weight
                    if self.scaleNeg != 1.0:
                        torch.lerp(empty_cross, cross, self.scaleNeg, out=cross)
                del empty_cross
                
                if batchSize == 1:
                    params.text_uncond['crossattn'][0] = cross
                else:
                    params.text_uncond['crossattn'] = cross.repeat(batchSize)
                del cross

        else:   #   not sdXL
            #   positive
            if self.zeroPos < 1.0 or self.shufflePos < 1.0 or self.noisePos > 0.0 or self.scalePos != 1.0:
                cond = params.text_cond[0]
                empty = self.empty_cond.resize_as_(cond).reshape(cond.shape)

                if self.zeroPos * lastStep < params.sampling_step:
                    cond = empty
                else:
                    #   shuffle
                    if self.shufflePos * lastStep < params.sampling_step:
                        indexes = torch.randperm(cond.size(0))
                        cond = cond[indexes]
                        del indexes

                        if self.scalePos == 1.0:   #   filthy hack
                            self.scalePos = 0.999

                    #noise - should noise be same for all steps? maybe can't be, due to prompt editing
                    if self.noisePos > 0.0:
                        noise = torch.randn_like(cond)
                        torch.lerp(cond, noise, self.noisePos, out=cond)
                        del noise

                    #   weight
                    if self.scalePos != 1.0:
                        torch.lerp(empty, cond, self.scalePos, out=cond)
                        del empty
                if batchSize == 1:
                    params.text_cond[0] = cond
                else:
                    params.text_cond = cond.repeat(batchSize)
                del cond

            #   negative
            if self.zeroNegS > 0.0 or self.zeroNegE < 1.0 or self.shuffleNeg < 1.0 or self.noiseNeg > 0.0 or self.scaleNeg != 1.0:
                cond = params.text_uncond[0]
                empty = self.empty_uncond.resize_as_(cond).reshape(cond.shape)

                if self.zeroNegS * lastStep > params.sampling_step or self.zeroNegE * lastStep < params.sampling_step:
                    cond = empty
                else:
                    #   shuffle
                    if self.shuffleNeg * lastStep < params.sampling_step:
                        indexes = torch.randperm(cond.size(0))
                        cond = cond[indexes]
                        del indexes

                        if self.scaleNeg == 1.0:   #   filthy hack
                            self.scaleNeg = 0.999

                    #noise - should noise be same for all steps? maybe can't be, due to prompt editing
                    if self.noiseNeg > 0.0:
                        noise = torch.randn_like(cond)
                        torch.lerp(cond, noise, self.noiseNeg, out=cond)
                        del noise

                    #   weight
                    if self.scaleNeg != 1.0:
                        torch.lerp(empty, cond, self.scaleNeg, out=cond)
                        del empty

                if batchSize == 1:
                    params.text_uncond[0] = cond
                else:
                    params.text_uncond = cond.repeat(batchSize)
                del cond


    def process(self, params, *script_args, **kwargs):
        enabled, shufflePos, shuffleNeg, noisePos, noiseNeg, scalePos, scaleNeg, zeroPos, zeroNegS, zeroNegE = script_args
        if enabled:
            self.shufflePos = shufflePos
            self.shuffleNeg = shuffleNeg
            self.noisePos = noisePos
            self.noiseNeg = noiseNeg
            self.scalePos = scalePos
            self.scaleNeg = scaleNeg
            self.zeroPos = zeroPos
            self.zeroNegS = zeroNegS
            self.zeroNegE = zeroNegE

            # Below codes will add some logs to the texts below the image outputs on UI.
            params.extra_generation_params.update(dict(
                cb_enabled = enabled,
                cb_shufflePos = shufflePos,
                cb_shuffleNeg = shuffleNeg,
                cb_noisePos = noisePos,
                cb_noiseNeg = noiseNeg,
                cb_scalePos = scalePos,
                cb_scaleNeg = scaleNeg,
                cb_zeroPos = zeroPos,
                cb_zeroNegS = zeroNegS,
                cb_zeroNegE = zeroNegE,
            ))

            on_cfg_denoiser(self.denoiser_callback)

        return

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled = script_args[0]

        if enabled:
            prompt = SdConditioning([""], is_negative_prompt=True, width=params.width, height=params.height)
            self.empty_uncond = shared.sd_model.get_learned_conditioning(prompt)
            prompt = SdConditioning([""], is_negative_prompt=False, width=params.width, height=params.height)
            self.empty_cond = shared.sd_model.get_learned_conditioning(prompt)

        return


    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return

