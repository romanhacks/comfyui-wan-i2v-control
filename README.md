# ComfyUI WAN I2V Control

Custom ComfyUI nodes for advanced control over WAN 2.1/2.2 Image-to-Video generation, featuring built-in person and face mask generation using MediaPipe.

## Features

- **Built-in person/face masking** - No external mask nodes needed
- **MediaPipe-powered detection** - Face, hair, body, clothes, background segmentation
- **Fine-grained face landmarks** - Eyes, lips, nose, forehead, cheeks, and more
- **Flexible mask modes** - Depth-based, preset patterns, or custom masks
- **Channel manipulation** - Control how much influence the reference image has

## Installation

1. Clone into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/shootthesound/comfyui-wan-i2v-control.git
   ```

2. Install dependencies:
   ```bash
   pip install mediapipe
   ```

3. Restart ComfyUI

## Nodes

### WAN I2V Conditioning Mask Pro

The main node for controlling what parts of the reference image influence the generated video.

**Mask Priority:** Person mask generation > External mask > Depth map > Preset mode

#### Person Segmentation (MediaPipe)
| Option | Description |
|--------|-------------|
| `mask_face` | Face skin area |
| `mask_hair` | Hair |
| `mask_body` | Body/skin |
| `mask_clothes` | Clothing |
| `mask_background` | Background |

#### Face Landmarks (works best on close-ups or 720p+)
| Option | Description |
|--------|-------------|
| `mask_face_oval` | Full face oval outline |
| `mask_eyes` | Both eyes |
| `mask_eyebrows` | Both eyebrows |
| `mask_lips` | Lips/mouth |
| `mask_pupils` | Pupils only |
| `mask_nose` | Nose area |
| `mask_cheeks` | Cheek areas |
| `mask_forehead` | Forehead area |
| `mask_jaw_chin` | Jaw and chin |
| `mask_ears` | Both ears |

#### Mask Controls
| Option | Description |
|--------|-------------|
| `mask_mode` | Preset patterns: full, face_focus, soft_vignette, etc. |
| `mask_strength` | Blend between masked (0.0) and unmasked (1.0) |
| `grow_mask` | Expand/contract mask (positive = grow, negative = shrink) |
| `feather` | Soft edge blur radius |
| `invert_mask` | Flip fill/keep regions |
| `ignore_area` | Exclude left/right/top/bottom portion from detection |

#### Fill Options
| Option | Description |
|--------|-------------|
| `fill_brightness` | Brightness of filled areas (0.0-1.0) |
| `tint_fill` | Apply color tint to filled areas |
| `tint_color` | Hex color for tint (e.g., #FF0000) |

### WAN I2V Channel Controller

Basic control over image influence in the model's conditioning channels.

| Input | Description |
|-------|-------------|
| `image_influence` | 0.0 = ignore image, 1.0 = normal, 2.0 = amplified |
| `channel_group_a` | Scale for channels 16-25 |
| `channel_group_b` | Scale for channels 26-35 |
| `invert_influence` | Negate image channels |

### WAN I2V Channel Controller (Advanced)

Per-channel-group control with noise injection for more variation.

## Usage Examples

**Regenerate face while keeping background:**
- Enable `generate_person_mask`
- Set `mask_face = True`
- Detected face will be regenerated, background preserved

**Soft eye enhancement:**
- Enable `generate_person_mask`
- Set `mask_eyes = True`
- Set `mask_strength = 0.5` for subtle effect

**Isolate one person in frame:**
- Set `ignore_area = "left"` or `"right"`
- Set `ignore_percent = 0.5`
- Only the person on one side will be detected

**Use alpha from transparent PNG:**
- Connect LoadImage's MASK output to the `mask` input
- The alpha channel controls which areas are filled

## Compatibility

- WAN 2.1 and 2.2 I2V models
- Works with fp16 and fp8 models
- Stacks with LoRAs and other model modifications

## Credits

Person and face mask generation powered by [MediaPipe](https://github.com/google-ai-edge/mediapipe).

## License

MIT
