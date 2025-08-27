# Minimal CASteer Implementation

A minimal, customizable implementation of [CASteer](https://arxiv.org/abs/2503.09630) for concept steering in Stable Diffusion models.

![Figure 1: Castle with metallic steering](results/metal_experiments/plots_output/gen_08_comparison.png)
*Figure: Castle with metallic steering vector, prompt: “Epic fantasy castle on top of a mountain, clouds swirling around, dramatic lighting.”*

## Overview
This repository provides a minimal implementation of concept steering using activation-based guidance vectors. It works by:

1. Collecting activations from cross-attention layers when processing prompt pairs 
    - This implementation takes a different approach than the official [implementation](https://github.com/Atmyre/CASteer), using steering hooks to collect activations rather than re-implementing the cross-attention layer.
2. Applying these vectors during inference to guide image generation

![Figure: Architecture of integrating vector steering into the diffusion pipeline (SD 1.5 pipeline from Demir's [blog](https://towardsdatascience.com/the-arrival-of-sdxl-1-0-4e739d5cc6c7/)).](results/architecture.png) 
  
  
*Figure: Architecture of integrating vector steering into the diffusion pipeline (SD 1.5 pipeline from Demir's [blog](https://towardsdatascience.com/the-arrival-of-sdxl-1-0-4e739d5cc6c7/)).*


## Features
- **Hook-based Implementation**: Uses forward hooks for both activation collection and steering application.
- **Experimentally Customizable**: Easy modification of how activations are collected and steering vectors are applied (editing the hook).
- **Composable Implementation**: Example of how to perform composable vector steering of multiple attributes.
- **Steering only after N diffusion steps**: Optionally steer only the last 50% of diffusion steps (see `tutorial_efficient.ipynb`). Or edit one line to apply steering after any N<sup>th</sup> step [here](https://github.com/sidhantls/minimal-casteer/blob/4f54844859f0993139b7a3907ae9e1a3825e7c1f/steering.py#L59).


## Notebooks
- **tutorial.ipynb**: Example of CASteer
- **tutorial_composition.ipynb**: Example of composing vectors using CASteer
- **tutorial_efficient.ipynb**: Example of applying CASteer based on diffusion steps

Notebooks should readily run in Google Colab. Can confirm it works on the L4 GPU. 


## High Level Usage
The implementation requires only a few steps:

```python
# Add hooks to collect activations
steer_hooks = steering.add_steer_hooks(pipe)

# Build steering vectors from prompt pairs
final_vectors = steering.build_final_steering_vectors(pipe, steer_hooks, prompts)

# Add calculated vectors to hooks for inference
steering.add_final_steer_vectors(steer_hooks, final_vectors)

# Generate images with steering
steering.run_grid_experiment(pipe, steer_hooks, test_prompts, steer_scale_list=[0.0, 5.0, 10.0])
```
