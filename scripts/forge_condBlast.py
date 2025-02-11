import gradio as gr

from modules import scripts
import modules.shared as shared
import modules as modules
from modules.prompt_parser import SdConditioning
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
import torch, math, random, time
from modules.ui_components import InputAccordion


class CondBlastForge(scripts.Script):
    def __init__(self):
        self.empty_uncond = None
        self.empty_cond = None

    def title(self):
        return "Cond Blastr"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Accordion(open=False, label="controls for Positive"):
                with gr.Row():
                    noisePos = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.0, label='add noise to text conds')
                    noisePosS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='noise after step')
                shufflePos = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='shuffle text conds after step')
                scalePos = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='prompt weight')
                zeroPos  = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='empty prompt after step')
            with gr.Accordion(open=True, label="controls for Negative"):
                with gr.Row():
                    posNeg   = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='blend positive into negative')
                    posNegS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='blend after step')
                with gr.Row():
                    noiseNeg = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.0, label='add noise to text conds (ICG)')
                    noiseNegS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='noise after step')
                shuffleNeg = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='shuffle text conds after step')
                scaleNeg = gr.Slider(minimum=0.1, maximum=2.0, step=0.005, value=1.0, label='prompt weight')
                with gr.Row():
                    zeroNegS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='empty negative before step')
                    zeroNegE = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='empty negative after step')


        enabled.do_not_save_to_config = True
        shufflePos.do_not_save_to_config = True
        shuffleNeg.do_not_save_to_config = True
        noisePos.do_not_save_to_config = True
        noiseNeg.do_not_save_to_config = True
        noisePosS.do_not_save_to_config = True
        noiseNegS.do_not_save_to_config = True
        scalePos.do_not_save_to_config = True
        scaleNeg.do_not_save_to_config = True
        zeroPos.do_not_save_to_config = True
        posNeg.do_not_save_to_config = True
        posNegS.do_not_save_to_config = True
        zeroNegS.do_not_save_to_config = True
        zeroNegE.do_not_save_to_config = True

        self.infotext_fields = [
            (enabled, lambda d: d.get("cb_enabled", False)),
            (shufflePos, "cb_shufflePos"),
            (shuffleNeg, "cb_shuffleNeg"),
            (noisePos, "cb_noisePos"),
            (noiseNeg, "cb_noiseNeg"),
            (noisePosS, "cb_noisePosS"),
            (noiseNegS, "cb_noiseNegS"),
            (scalePos, "cb_scalePos"),
            (scaleNeg, "cb_scaleNeg"),
            (zeroPos,  "cb_zeroPos"),
            (posNeg,   "cb_posNeg"),
            (posNegS,  "cb_posNegS"),
            (zeroNegS, "cb_zeroNegS"),
            (zeroNegE, "cb_zeroNegE"),
        ]

        return enabled, shufflePos, shuffleNeg, noisePos, noiseNeg, noisePosS, noiseNegS, scalePos, scaleNeg, zeroPos, posNeg, posNegS, zeroNegS, zeroNegE

    @torch.no_grad()
    def denoiser_callback(self, params):
        if getattr (CondBlastForge, 'empty_cond', None) is None:
            # partial workaround for BMAB blocking process_before_every_sampling
            prompt = SdConditioning([""], is_negative_prompt=True, width=1, height=1)
            CondBlastForge.empty_uncond = shared.sd_model.get_learned_conditioning(prompt)
            prompt = SdConditioning([""], is_negative_prompt=False, width=1, height=1)
            CondBlastForge.empty_cond = shared.sd_model.get_learned_conditioning(prompt)
            return

        is_SDXL = isinstance (params.text_cond, dict)
        lastStep = params.total_sampling_steps - 1

        batchSize = len(params.text_cond['vector']) if is_SDXL else len(params.text_cond)

        ##  POSITIVE
        if self.zeroPos < 1.0 or self.shufflePos < 1.0 or (self.noisePos > 0.0 and self.noisePosS < 1.0) or self.scalePos != 1.0:
            for i in range(batchSize):
                if is_SDXL:
                    cond = params.text_cond['crossattn'][i]
                    empty = CondBlastForge.empty_cond['crossattn'][0].clone()
                else:
                    cond = params.text_cond[i]
                    empty = CondBlastForge.empty_cond[0].clone()

                resize = cond.shape[0] // empty.shape[0]
                if resize > 1:
                    empty = empty.repeat(resize, 1)
                    
                if self.zeroPos * lastStep < params.sampling_step:
                    cond = empty
                else:
                    if self.noisePos > 0.0 and self.noisePosS * lastStep < params.sampling_step:
                        noise = torch.randn_like(cond) * cond.std()
                        torch.lerp(cond, noise, self.noisePos, out=cond)
                        del noise
                    if self.shufflePos * lastStep < params.sampling_step:
                        indexes = torch.randperm(cond.size(0))
                        cond = cond[indexes]
                        del indexes
                    if self.scalePos != 1.0:
                        torch.lerp(empty, cond, self.scalePos, out=cond)

                del empty
            
                if is_SDXL:
                    params.text_cond['crossattn'][i] = cond
                else:
                    params.text_cond[i] = cond

                del cond

        ##  NEGATIVE
        if getattr (params, 'text_uncond', None) is None:
            return
        if getattr (CondBlastForge, 'empty_uncond', None) is None:
            return


        batchSize = len(params.text_uncond['vector']) if is_SDXL else len(params.text_uncond)

        if self.zeroNegS > 0.0 or self.zeroNegE < 1.0 or self.shuffleNeg < 1.0 or (self.noiseNeg > 0.0 and self.noiseNegS < 1.0)  or self.scaleNeg != 1.0 or (self.posNeg > 0.0 and self.posNegS < 1.0):
            for i in range(batchSize):
                if is_SDXL:
                    cond = params.text_uncond['crossattn'][i]
                    empty = CondBlastForge.empty_uncond['crossattn'][0].clone()
                else:
                    cond = params.text_uncond[i]
                    empty = CondBlastForge.empty_uncond[0].clone()

                resize = cond.shape[0] // empty.shape[0]
                if resize > 1:
                    empty = empty.repeat(resize, 1)
                    
                if self.zeroNegS * lastStep > params.sampling_step or self.zeroNegE * lastStep < params.sampling_step:
                    cond = empty
                else:
                    #   blend positive
                    if self.posNeg > 0.0 and self.posNegS * lastStep < params.sampling_step:
                        if is_SDXL:
                            pos_cond = params.text_cond['crossattn'][0]
                        else:
                            pos_cond = params.text_cond[0]

                        if pos_cond.shape[0] > cond.shape[0]:
                            # positive longer than negative, just truncate
                            pos_cond = pos_cond[:cond.shape[0]]
                        elif pos_cond.shape[0] < cond.shape[0]:
                            # negative longer than positive, expand positive with zero
                            new_cond = torch.zeros_like(cond)
                            new_cond[:pos_cond.shape[0], :] = pos_cond
                            pos_cond = new_cond
                            del new_cond

                        torch.lerp(cond, pos_cond, self.posNeg, out=cond)

                        del pos_cond
                
                    #   noise
                    if self.noiseNeg > 0.0 and self.noiseNegS * lastStep < params.sampling_step:
                        noise = torch.randn_like(cond) * cond.std()
                        torch.lerp(cond, noise, self.noiseNeg, out=cond)
                        del noise
                
                    #   shuffle
                    if self.shuffleNeg * lastStep < params.sampling_step:
                        indexes = torch.randperm(cond.size(0))
                        cond = cond[indexes]
                        del indexes
                
                    #   weight
                    if self.scaleNeg != 1.0:
                        torch.lerp(empty, cond, self.scaleNeg, out=cond)

                del empty
                        
                if is_SDXL:
                    params.text_uncond['crossattn'][i] = cond
                else:
                    params.text_uncond[i] = cond

                del cond


    def process(self, params, *script_args, **kwargs):
        enabled, shufflePos, shuffleNeg, noisePos, noiseNeg, noisePosS, noiseNegS, scalePos, scaleNeg, zeroPos, posNeg, posNegS, zeroNegS, zeroNegE = script_args
        if enabled:
            self.shufflePos = shufflePos
            self.shuffleNeg = shuffleNeg
            self.noisePos = noisePos
            self.noiseNeg = noiseNeg
            self.noisePosS = noisePosS
            self.noiseNegS = noiseNegS
            self.scalePos = scalePos
            self.scaleNeg = scaleNeg
            self.zeroPos  = zeroPos
            self.posNeg   = posNeg
            self.posNegS  = posNegS
            self.zeroNegS = zeroNegS
            self.zeroNegE = zeroNegE

            # Below codes will add some logs to the texts below the image outputs on UI.
            params.extra_generation_params.update(dict(
                cb_enabled = enabled,
                cb_shufflePos = shufflePos,
                cb_shuffleNeg = shuffleNeg,
                cb_noisePos = noisePos,
                cb_noiseNeg = noiseNeg,
                cb_noisePosS = noisePosS,
                cb_noiseNegS = noiseNegS,
                cb_scalePos = scalePos,
                cb_scaleNeg = scaleNeg,
                cb_zeroPos  = zeroPos,
                cb_posNeg   = posNeg,
                cb_posNegS  = posNegS,
                cb_zeroNegS = zeroNegS,
                cb_zeroNegE = zeroNegE,
            ))

            on_cfg_denoiser(self.denoiser_callback)

        return

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled = script_args[0]

        # must create empties here, in case model architecture is different for second pass
        if enabled:
            prompt = SdConditioning([""], is_negative_prompt=True, width=params.width, height=params.height)
            CondBlastForge.empty_uncond = shared.sd_model.get_learned_conditioning(prompt)
            prompt = SdConditioning([""], is_negative_prompt=False, width=params.width, height=params.height)
            CondBlastForge.empty_cond = shared.sd_model.get_learned_conditioning(prompt)
        return


    def postprocess(self, params, processed, *args):
        CondBlastForge.empty_uncond = None
        CondBlastForge.empty_cond = None
        remove_current_script_callbacks()
        return

