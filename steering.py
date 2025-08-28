from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence, Optional, Union
import os
import math
from tqdm.auto import tqdm
import torch
from diffusers import StableDiffusionXLPipeline
import torch.nn.functional as F


class SteeringHooks:
    """
    Registers on cross-attn blocks. Either:
      (A) Records activations into .cache every `save_every` steps
      (B) Applies provided steering vectors when steer_vectors is a tensor
    """
    def __init__(self, save_every: int = 1,
                 steer_vectors: Optional[torch.Tensor] = None,
                 scale_steer: float = 1.0,
                 steer_type: str = "default",
                 last_half_timesteps: bool = False):
        
        """
        Initializes the steering object with optional steering vectors and configuration.
        Args:
            save_every (int, optional): Interval for saving state or results. Defaults to 1.
            steer_vectors (Optional[torch.Tensor], optional): Tensor containing steering vectors, shaped (timesteps, 1, dim). Defaults to None (during activation collection)
            scale_steer (float, optional): Scaling factor applied to steering vectors. Defaults to 1.0.
            steer_type (str, optional): Type of steering to use. Must be 'default'. Defaults to "default".
        """

        self.save_every = save_every
        self.steer_vectors = steer_vectors  # (timesteps, 1, dim)
        self.step = 0
        self.cache: List[torch.Tensor] = []
        self.scale_steer = scale_steer
        assert steer_type == "default", "steer_type must be 'default'"
        self.steer_type = steer_type
        self.last_half_timesteps = last_half_timesteps

    def hook_fn(self, module, inputs, x_seq):
        """
        This function is run immediately after each forward pass of the layer it's hooked to

        Args:
            module: The module to which the hook is attached. (Unused)
            inputs: The inputs to the module. (Unused)
            x_seq (torch.Tensor): The output sequence from the module, shape (2, L, dim) with CFG (unconditional/conditional).

        Returns:
            torch.Tensor: The modified output sequence after steering or caching.
        """
        device = x_seq.device
        
        # use steering vectors if present
        if isinstance(self.steer_vectors, torch.Tensor):
            total_steps = self.steer_vectors.shape[0]
            if self.last_half_timesteps and self.step < total_steps // 2:
                self.step += 1
                return x_seq

            # get steering vector of current step. v: (n, dim), n=1
            v = self.steer_vectors[self.step // self.save_every]
            norm1 = torch.norm(x_seq, dim=2, keepdim=True)

            v_padded = v[:, None, :].to(device)  # broadcast along sequence length. (n, 1, dim)
            x_seq = x_seq + self.scale_steer * v_padded
            x_seq = F.normalize(x_seq, dim=2)
            x_seq = x_seq * norm1

        # cache activation to calculate steering vector
        else:
            if self.step % self.save_every == 0:  # can choose to cache only k-timesteps
                # use only conditional portion (from CASteer impl)
                # average across sequence length
                self.cache.append(x_seq[1, :, :].mean(0).detach().cpu()[None, None, :])  # (1, 1, dim)

        self.step += 1
        return x_seq

def clear_steer_cache(steer_hooks: Sequence[SteeringHooks]) -> None:
    """
    Clears the cache and resets the step for all steering hooks.

    This is typically executed after generating a positive image, so that the cache is clear to collect activations from the negative image.

    Args:
        steer_hooks: List of steering hook objects whose cache and step should be reset.
    """
    for sh in steer_hooks:
        sh.cache.clear()
        sh.step = 0


def find_cross_attention_modules(unet, select_blocks=None):
    """
    Finds all cross-attention modules in the UNet.

    Args:
        unet: The UNet model (pipe.unet) to search for cross-attention modules.
        select_block: If None, finds all CA blocks. If string, returns block with select_block 
            in the module name (for eg. "up_blocks", "mid_blocks", "down_blocks")
    Returns:
        List of tuples containing (module_name, module) for each cross-attention module found.
    """
    targets = []
    for name, m in unet.named_modules():
        if getattr(m, "is_cross_attention", False):
            if select_blocks is None or any(item in name for item in select_blocks):
                assert isinstance(select_blocks, list), "expected select_blocks to be of list type"

                targets.append((name, m))
    return targets

def add_steer_hooks(
    pipe: StableDiffusionXLPipeline,
    steer_type: str = "default",
    save_every: int = 1,
    last_half_timesteps: bool = False,
    select_blocks: Optional[List] = None
) -> List[SteeringHooks]:
    """Adds steering hooks to cross-attention layers in the Stable Diffusion pipeline.

    Args:
        pipe: The Hugging Face Stable Diffusion pipeline model.
        steer_type (str): Type of steering to use.
        save_every (int): Frequency at which to collect or apply steering vectors. Defaults to 1.
        last_half_timesteps (bool): If ``True``, steering vectors are applied only during the last
            half of diffusion timesteps.
        select_blocks: Add steering hooks to only selected blocks (None, "up_blocks", "down_blocks", "mid_blocks":)
    Returns:
        List[SteeringHooks]: List of ``SteeringHooks`` objects registered to cross-attention layers.
    """

    targets = find_cross_attention_modules(pipe.unet, select_blocks=select_blocks)


    steer_hooks = []
    for _, m in targets:
        steerer = SteeringHooks(
            save_every=save_every,
            steer_vectors=None,
            scale_steer=1.0,  # unused during collection
            steer_type=steer_type,
            last_half_timesteps=last_half_timesteps,
        )
        m.register_forward_hook(steerer.hook_fn)  # register the hook so that activations are collected
        steer_hooks.append(steerer)
    print(f"[hooks] Added {len(steer_hooks)} cross-attn hooks")
    return steer_hooks

def add_final_steer_vectors(steer_hooks,final_steering_vectors):
    """
    Attaches the final steering vectors to each SteeringHooks object for use during image generation.

    Args:
        steer_hooks: List of steering hook objects to update.
        final_steering_vectors: List of steering vectors to assign to each hook.
    """
    assert len(steer_hooks) == len(final_steering_vectors), "Number of hooks should be the same as number of steering vectors"
    
    for i, hook in enumerate(steer_hooks):
        hook.steer_vectors = final_steering_vectors[i].detach().cpu()
    
    print(f"[hooks] Attached {len(final_steering_vectors)} steering vectors")

def set_steer_scale(steer_hooks, scale):
    """
    Sets the steering scale for all SteeringHooks objects.

    Args:
        steer_hooks: List of steering hook objects to update.
        scale (float): The scaling factor to apply to each hook.
    """

    for hook in steer_hooks:
        hook.scale_steer = float(scale)

def reset_step(steer_hooks):
    """
    Resets the running timestep counter to 0 for all SteeringHooks.

    This should be called before generating each image to ensure the steering hooks start from the first timestep.

    Args:
        steer_hooks: List of steering hook objects whose step counters should be reset.
    """
    for hook in steer_hooks:
        hook.step = 0


def build_final_steering_vectors(
    pipe: StableDiffusionXLPipeline,
    steer_hooks: Sequence[SteeringHooks],
    prompts: List[str],
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    save_posneg_images_to: Optional[str] = "posneg_images",
    show_progress: bool = True,
    seed=0
) -> List[torch.Tensor]:
    """
    Calculates the final steering vectors from the dataset (prompt pairs).

    Args:
        pipe: The Stable Diffusion pipeline used for image generation and activation collection.
        steer_hooks: List of steering hook objects registered to cross-attention layers.
        prompts: List of prompt pairs [(positive, negative)] for steering vector calculation.
        num_inference_steps: Number of inference steps for the diffusion process. Defaults to 20.
        guidance_scale: Scale for classifier-free guidance. Defaults to 5.0.
        save_posneg_images_to: Directory to save preview images for each positive/negative pair. Defaults to "posneg_images".
        show_progress: Whether to show a progress bar during processing. Defaults to True.
        seed: Random seed for reproducibility. Defaults to 0.

    Returns:
        List of tensors [(timesteps, 1, dim), ...] representing the final steering vectors.
    """
    os.makedirs(save_posneg_images_to, exist_ok=True)

    num_layers = len(steer_hooks)
    pos_sums: List[Optional[torch.Tensor]] = [None] * num_layers
    neg_sums: List[Optional[torch.Tensor]] = [None] * num_layers

    iterator = tqdm(enumerate(prompts), total=len(prompts)) if show_progress else enumerate(prompts)
    # Helper function to create safe filenames from prompts
    def safe_filename(prompt):
        # Replace problematic characters
        return "".join([c if c.isalnum() else "_" for c in prompt])[:50]
        
    for idx, (prompt_pos, prompt_neg) in iterator:
        # POS - Generate image, save it, and collect activations
        pos_img = pipe(prompt=prompt_pos,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
        
        # Save positive image
        pos_filename = f"{idx:02d}_pos_{safe_filename(prompt_pos)}.png"
        pos_img.save(os.path.join(save_posneg_images_to, pos_filename))
        
        # Collect activations for positive image - need to run again with same seed
        _ = pipe(prompt=prompt_pos,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
                 
        # Cache activations
        steer_matrix_pos = [] 
        for h in steer_hooks:
            activations = torch.cat(h.cache, dim=0) # (timestep, n, dim=640)
            steer_matrix_pos.append(activations)
        clear_steer_cache(steer_hooks) # clear activation cache    

        # NEG - Generate image, save it, and collect activations
        neg_img = pipe(prompt=prompt_neg,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
                 
        # Save negative image
        neg_filename = f"{idx:02d}_neg_{safe_filename(prompt_neg)}.png"
        neg_img.save(os.path.join(save_posneg_images_to, neg_filename))
        
        # Collect activations for negative image - need to run again with same seed
        _ = pipe(prompt=prompt_neg,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
        
        # cache activations
        steer_matrix_neg = [] 
        for h in steer_hooks:
            activations = torch.cat(h.cache, dim=0)
            steer_matrix_neg.append(activations)
        clear_steer_cache(steer_hooks) # clear activation cache

        # running sum of positive and negative activations across dataset
        for layer_idx in range(num_layers):
            P = steer_matrix_pos[layer_idx] # (timesteps, 1, 640)
            N = steer_matrix_neg[layer_idx] # (timesteps, 1, 640)
            pos_sums[layer_idx] = P.clone() if pos_sums[layer_idx] is None else pos_sums[layer_idx] + P
            neg_sums[layer_idx] = N.clone() if neg_sums[layer_idx] is None else neg_sums[layer_idx] + N

    # calculate final steering vectors
    final_vectors = [] 
    for layer_idx in range(num_layers):
        pos_avg = pos_sums[layer_idx] / len(prompts)
        neg_avg = neg_sums[layer_idx] / len(prompts)

        steer_vector = pos_avg - neg_avg

        # normalize so only direction matters and scale_steer has a consistent effect
        steer_vector = F.normalize(steer_vector, dim=-1)
        final_vectors.append(steer_vector)

    print(f"[vectors] Built {len(final_vectors)} steering vectors; each like {tuple(final_vectors[0].shape)}")
    return final_vectors # [(timesteps, n=1, dim=640), ...]

def generate_images(
    pipe: StableDiffusionXLPipeline,
    prompts: List,
    num_inference_steps: int,
    guidance_scale: float,
    steer_hooks,
    out_dir: str,
    steering_scale: Optional[float] = None,
    seed=0,
):
    """
    Runs inference using a Stable Diffusion pipeline to generate and save images for a list of prompts.
    
    Args:
        pipe (StableDiffusionXLPipeline): The Stable Diffusion pipeline used for image generation.
        prompts (List): List of text prompts for image generation.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        guidance_scale (float): Scale for classifier-free guidance.
        steer_hooks: Hooks for steering the generation process.
        out_dir (str): Directory to save generated images.
        steering_scale (Optional[float], optional): Scale for steering hooks. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    Returns:
        List[str]: List of filepaths to the saved images.
    """
    os.makedirs(out_dir, exist_ok=True)
    if steering_scale is not None:
        set_steer_scale(steer_hooks, steering_scale)

    saved = []
    for i, prompt in enumerate(prompts, 1):
        reset_step(steer_hooks) # reset running step before each generation
        img = pipe(prompt=prompt,
                   num_inference_steps=num_inference_steps,
                   guidance_scale=guidance_scale,
                   generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
        fp = os.path.join(out_dir, f"gen_{i:02d}.png")
        img.save(fp)
        saved.append(fp)

    return saved


def run_grid_experiment(
    pipe: StableDiffusionXLPipeline,
    steer_hooks: List[SteeringHooks],
    test_prompts,
    num_inference_steps,
    steer_type,
    gscale_list,
    steer_scale_list,
    out_root: str = "experiments",
    seed=0
) -> None:
    """
    Runs a grid search experiment over combinations of guidance scale and steering scale for image generation.
    
    Args:
        pipe (StableDiffusionXLPipeline): The diffusion pipeline used for image generation.
        steer_hooks (List[SteeringHooks]): List of steering hook objects to control image generation.
        test_prompts: Prompts to use for image generation.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        steer_type: Type of steering applied during generation.
        gscale_list (list): List of guidance scale values to sweep over.
        steer_scale_list (list): List of steering scale values to sweep over.
        out_root (str, optional): Root directory to save experiment results. Defaults to "experiments".
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    Saves:
        Generated images for each combination of guidance and steering scale under the specified output directory.
    
    """
    os.makedirs(out_root, exist_ok=True)
    clear_steer_cache(steer_hooks)

    for gscale in gscale_list:
        for k in steer_scale_list:
            # set steer_type and scale live
            for sh in steer_hooks:
                sh.scale_steer = float(k)

            sub = os.path.join(
                out_root, f"steps={num_inference_steps}_guide={gscale}_steer={k}_type={steer_type}"
            )
            paths = generate_images(
                pipe,
                test_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=gscale,
                steer_hooks=steer_hooks,
                out_dir=sub,
                seed=seed
            )
            print(
                f"[grid] Saved {len(paths)} images "
                f"-> steps={num_inference_steps}, guide={gscale}, steer={k}, type={steer_type}"
            )
