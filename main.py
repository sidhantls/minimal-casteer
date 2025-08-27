# !pip install diffusers transformers accelerate torch torchvision safetensors
from diffusers.utils import logging
import steering
import prompt_catalog
from diffusers import StableDiffusionXLPipeline
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"
torch_dtype = "float16"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16,
   generator=torch.Generator(device=device).manual_seed(0)
)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

## Hyper Paramters
STEER_TYPE = "default"
INF_STEPS = 20
GUIDE_SCALE = 5.0
SCALE_STEER = 1.0 # not applied during cache creation
PROMPT_THEME = "metal"

# Loads "calibration dataset": dataset from which steering vectors are derived from
if PROMPT_THEME == "anime":
    prompts = prompt_catalog.ANIME_PROMPT[:20]
elif PROMPT_THEME == "metal":
    prompts = prompt_catalog.METALLIC_SCULPTURE_SET[:20]

else:
  raise NotImplementedError(f"Prompt theme {PROMPT_THEME} not implemented")

# add hooks to collect activations and later applies steering vector to intermiate activations 
steer_hooks = steering.add_steer_hooks(pipe, steer_type=STEER_TYPE, save_every=1, initial_scale=SCALE_STEER)

final_vecs = steering.build_final_steering_vectors(
    pipe,
    steer_hooks,
    prompts,
    num_inference_steps=INF_STEPS,
    guidance_scale=GUIDE_SCALE
)

print(final_vecs[0].shape) # (20, 2, 640)

# adds calculated steering vectors to hooks so it can be applied during forward pass
steering.add_final_steer_vectors(steer_hooks, final_vecs)


# Prompts to apply steering vectors to
TEST_PROMPTS = [
    "Studio-lit Batman logo on a plain background, 4K detail.",
    "Cinematic close-up of Batman's face, dramatic shadows across the cowl, ultra-detailed, 4K.",
    "Close-up of a labrador sitting.",
    "Portrait of an Apple.",
    "Close-up of an elderly woman, natural lighting, detailed wrinkles.",
    "Macro shot of a fox face, fur glowing in sunlight.",
    "Close-up of a robot face, metallic surface with glowing eyes, with a neural expression, futuristic design."
    "Portrait of a carved pumpkin jack-o’-lantern with a neural expression, dramatic lighting."
]


# generates images using steering vectors
STEER_SCALE_LIST = [0.0, 1.0, 2.0, 10.0]
steering.run_grid_experiment(
    pipe, steer_hooks, TEST_PROMPTS,
    num_inference_steps=INF_STEPS,
    steer_type=STEER_TYPE,
    gscale_list=[GUIDE_SCALE],
    steer_scale_list=STEER_SCALE_LIST, # (0.0: no steering (baseline), 10.0: strong steering)
    out_root=f"{PROMPT_THEME}_experiments",
)
