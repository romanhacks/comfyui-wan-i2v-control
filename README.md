# Wan I2V Channel Controller

Custom ComfyUI nodes for controlling image conditioning in Wan 2.2 I2V models.

## What it does

The Wan I2V model has 36 input channels in its `patch_embedding` layer:
- **Channels 0-15**: Latent input (standard diffusion)
- **Channels 16-35**: Image conditioning (reference image)

These nodes let you control how much influence the reference image has on generation, turning strict I2V into something more like an IP-Adapter.

## Nodes

### Wan I2V Channel Controller
Basic control over image influence.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | MODEL | - | The Wan I2V model |
| image_influence | float | 1.0 | 0.0 = ignore image, 1.0 = normal, 2.0 = amplified |
| channel_group_a | float | 1.0 | Scale for channels 16-25 |
| channel_group_b | float | 1.0 | Scale for channels 26-35 |
| invert_influence | bool | False | Negate image channels |

### Wan I2V Channel Controller (Advanced)
Per-channel-group control with noise injection.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | MODEL | - | The Wan I2V model |
| image_influence | float | 1.0 | Overall scaling (can be negative) |
| channels_0_4 | float | 1.0 | Scale for image channels 0-4 |
| channels_5_9 | float | 1.0 | Scale for image channels 5-9 |
| channels_10_14 | float | 1.0 | Scale for image channels 10-14 |
| channels_15_19 | float | 1.0 | Scale for image channels 15-19 |
| noise_injection | float | 0.0 | Add random noise for variation |
| noise_seed | int | 0 | Seed for noise |

### Wan I2V Channel Analyzer
Analyzes the model's patch_embedding and outputs statistics.

## Usage

```
[Load Diffusion Model] → [Wan I2V Channel Controller] → [Rest of Workflow]
```

## Examples

**Soft I2V (IP-Adapter-like):**
- `image_influence = 0.3`
- Image becomes a loose style/composition guide

**Anti-Reference:**
- `image_influence = 1.0`
- `invert_influence = True`
- Output actively diverges from input

**Selective Influence:**
- `channel_group_a = 0.8` (structure?)
- `channel_group_b = 0.2` (style?)
- Experiment to find what each group controls

## Compatibility

- Works with Wan 2.2 I2V models (high and low noise)
- Passes through T2V models unchanged
- Works with fp16 and fp8 models

## Notes

- The node clones the model, so the original is not modified
- Changes only affect the current workflow execution
- Stacks with LoRAs and other model modifications
