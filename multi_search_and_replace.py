from copy import copy
from PIL import Image

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.hypernetworks import hypernetwork
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models

# Multi-scaler allows you to append keyword groups that scale independently of each other.
# This script works exactly the same way as Prompt S/R in the XY grid, except there will only be one dimension.
# For example, you might want to scale (cat:1.6) downwards and (dog:0.4) upwards, to avoid confusing the model.
# To do this, simply enter (cat:1.6) in Group 1 prompts, and specify 1.6, 1.4, 1.2, 1.0 in Group 1 scale,
# and then enter (dog:0.4) in Group 2 prompts, and 0.4, 0.8, 1.2, 1.4 in Group 2 scale.
# Note that the number of scale interations for each group must be the same. 

def draw_series(p:StableDiffusionProcessingTxt2Img, g1kw:str, g1sr:list[str], g1_isneg:bool, g2kw:str, g2sr:list[str], g2_isneg:bool, g3kw:str, g3sr:list[str], g3_isneg:bool, g4kw:str, g4sr:list[str], g4_isneg:bool):
    longest = max(len(g1sr), len(g2sr), len(g3sr), len(g4sr))
    state.job_count = longest * p.n_iter
    image_cache = []
    processed_result = None
    cell_mode = "P"
    cell_size = (1,1)
    for i in range(longest):
        pc = copy(p)
        if g1kw and len(g1sr) > 0:
            if g1_isneg:
                pc.negative_prompt = pc.negative_prompt + ' ' + g1kw.replace(g1sr[0], g1sr[i])
            else:
                pc.prompt = pc.prompt + ' ' + g1kw.replace(g1sr[0], g1sr[i])

        if g2kw and len(g2sr) > 0:
            if g2_isneg:
                pc.negative_prompt = pc.negative_prompt + ' ' + g2kw.replace(g2sr[0], g2sr[i])
            else:
                pc.prompt = pc.prompt + ' ' + g2kw.replace(g2sr[0], g2sr[i])

        if g3kw and len(g3sr) > 0:
            if g3_isneg:
                pc.negative_prompt = pc.negative_prompt + ' ' + g3kw.replace(g3sr[0], g3sr[i])
            else:
                pc.prompt = pc.prompt + ' ' + g3kw.replace(g3sr[0], g3sr[i])

        if g4kw and len(g4sr) > 0:
            if g4_isneg:
                pc.negative_prompt = pc.negative_prompt + ' ' + g4kw.replace(g4sr[0], g4sr[i])
            else:
                pc.prompt = pc.prompt + ' ' + g4kw.replace(g4sr[0], g4sr[i])
        processed:Processed = process_images(pc) # Generation occurs here
        try:
            processed_image = processed.images[0]
            if processed_result is None:
                processed_result = copy(processed)
                cell_mode = processed_image.mode
                cell_size = processed_image.size
                processed_result.images = [Image.new(cell_mode, cell_size)]
            image_cache.append(processed_image)
        except:
            image_cache.append(Image.new(cell_mode, cell_size))

    if not processed_result:
        print("Unexpected error: Multi S/R failed to return even a single processed image.")
        return Processed()
    processed_result.images[0] = images.image_grid(image_cache, rows=1)
    return processed_result

class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.hypernetwork = opts.sd_hypernetwork
        self.model = shared.sd_model

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights(self.model)

        hypernetwork.load_hypernetwork(self.hypernetwork)
        hypernetwork.apply_strength()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers

class Script(scripts.Script):
    def title(self):
        return "Multi Search and Replace"

    def ui(self, is_img2img):
        group1_isneg = gr.Checkbox(label="Group 1: is negative prompt", value=False)
        group1kw = gr.Textbox(label="Group 1: prompts", value="")
        group1sr = gr.Textbox(label="Group 1: prompt search and replace", value="", placeholder="Enter comma-separated values here")

        group2_isneg = gr.Checkbox(label="Group 2: is negative prompt", value=False)
        group2kw = gr.Textbox(label="Group 2: prompts", value="")
        group2sr = gr.Textbox(label="Group 2: prompt search and replace", value="", placeholder="Enter comma-separated values here")

        group3_isneg = gr.Checkbox(label="Group 3: is negative prompt", value=False)
        group3kw = gr.Textbox(label="Group 3: prompts", value="")
        group3sr = gr.Textbox(label="Group 3: prompt search and replace", value="", placeholder="Enter comma-separated values here")

        group4_isneg = gr.Checkbox(label="Group 4: is negative prompt", value=False)
        group4kw = gr.Textbox(label="Group 4: prompts", value="")
        group4sr = gr.Textbox(label="Group 4: prompt search and replace", value="", placeholder="Enter comma-separated values here")

        return [group1kw, group1sr, group1_isneg,
                group2kw, group2sr, group2_isneg,
                group3kw, group3sr, group3_isneg,
                group4kw, group4sr, group4_isneg
                ]

    def run(self, p:StableDiffusionProcessingTxt2Img, g1kw:str, group1scale:str, g1_isneg:bool, g2kw:str, group2scale:str, g2_isneg:bool, g3kw:str, group3scale:str, g3_isneg:bool, g4kw:str, group4scale:str, g4_isneg:bool):
        if not opts.return_grid:
            p.batch_size = 1

        g1sr = [i.strip() for i in group1scale.split(',')]
        g2sr = [i.strip() for i in group2scale.split(',')]
        g3sr = [i.strip() for i in group3scale.split(',')]
        g4sr = [i.strip() for i in group4scale.split(',')]

        #print(f"X/Y plot will create {len(xs) * len(ys) * p.n_iter} images on a {len(xs)}x{len(ys)} grid. (Total steps to process: {total_steps * p.n_iter})")
        #shared.total_tqdm.updateTotal(total_steps * p.n_iter)

        with SharedSettingsStackHelper():
            processed = draw_series(
                p,
                g1kw, g1sr, g1_isneg,
                g2kw, g2sr, g2_isneg,
                g3kw, g3sr, g3_isneg,
                g4kw, g4sr, g4_isneg
            )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "transform_seq", extension=opts.grid_format, prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed
