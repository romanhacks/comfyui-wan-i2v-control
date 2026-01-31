"""
Wan I2V Conditioning Manipulation Nodes
=======================================
"""

import torch
import torch.nn.functional as F
import copy


class WanI2VConditioningMaskPro:
    """
    Full-featured pixel-space masking for Wan I2V.

    Combines the proper VAE re-encoding approach with all mask options:
    - Preset mask modes (top_half, gradient, vignette, etc.)
    - Custom mask input
    - Depth map input (use depth as mask)
    - Feathering
    - Fill color control

    Protected regions keep original latent (no quality loss).
    Masked regions get VAE-encoded fill (no flicker).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Custom mask. White=keep reference, Black=fill. Overrides mask_mode."
                }),
                "depth_map": ("IMAGE", {
                    "tooltip": "Depth map image. Converted to mask. White (close)=keep, Black (far)=fill."
                }),
                "mask_mode": (["full", "top_half", "bottom_half", "left_half", "right_half",
                              "center", "edges", "gradient_top", "gradient_bottom",
                              "gradient_left", "gradient_right", "vignette"], {
                    "default": "full",
                    "tooltip": "Preset mask shape. Ignored if mask or depth_map connected."
                }),
                "depth_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Depth cutoff. 0=use depth map as gradient mask. 0.05=most of depth masked out except furthest away. 0.95=minimal foreground area masked out."
                }),
                "fill_brightness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 0.75,
                    "step": 0.05,
                    "tooltip": "Fill brightness for masked regions. Below 0.5=darker generation, above 0.5=brighter. Model doesn't process well outside 0.25-0.75 range."
                }),
                "tint_fill": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable color tint on fill to influence the vibe of generated areas"
                }),
                "tint_color": ("STRING", {
                    "default": "",
                    "tooltip": "Hex color for tint (e.g. FF8800 for warm orange, 4488FF for cool blue). Leave empty for no tint."
                }),
                "mask_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask intensity. Values below 1 let some original detail bleed through in filled areas, useful for subtle changes."
                }),
                "feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Soften mask edges (0=hard, 1=maximum blur)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask (swap protected and fill regions)"
                }),
                "text_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Scale text conditioning strength. >1.0 boosts text influence for sharper results."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "MASK",)
    RETURN_NAMES = ("positive", "negative", "mask_out",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/Wan"
    DESCRIPTION = "Full-featured pixel-space masking with VAE re-encoding"

    def apply_mask(self, positive, negative, vae, image, mask=None, depth_map=None,
                   mask_mode="full", depth_threshold=0.5, fill_brightness=0.5,
                   tint_fill=False, tint_color="",
                   mask_strength=1.0, feather=0.0, invert_mask=False,
                   text_strength=1.0):
        import comfy.utils

        # Get dimensions from existing conditioning
        cond_dict = positive[0][1] if positive else None
        if cond_dict is None or "concat_latent_image" not in cond_dict:
            raise ValueError("Input conditioning must have concat_latent_image (from WanImageToVideo)")

        original_latent = cond_dict["concat_latent_image"]
        B_lat, C, T, H, W = original_latent.shape

        # Get image dimensions
        img_h, img_w = image.shape[1], image.shape[2]

        # Create or process mask
        # Priority: mask > depth_map > mask_mode preset
        if mask is not None:
            # Use custom mask
            work_mask = mask.clone()
            if len(work_mask.shape) == 2:
                work_mask = work_mask.unsqueeze(0)
            if len(work_mask.shape) == 3:
                work_mask = work_mask.unsqueeze(1)
            # Resize to image dimensions
            work_mask = F.interpolate(work_mask, size=(img_h, img_w), mode='bilinear', align_corners=False)
        elif depth_map is not None:
            # Use depth map as mask
            # depth_map is IMAGE type: [B, H, W, C]
            # Convert to greyscale and use as mask
            depth = depth_map.clone()

            # Convert to greyscale if RGB (average channels)
            if len(depth.shape) == 4 and depth.shape[-1] >= 3:
                depth = depth[..., :3].mean(dim=-1)  # [B, H, W]
            elif len(depth.shape) == 4:
                depth = depth[..., 0]  # Single channel

            # Resize to match main image
            depth = depth.unsqueeze(1)  # [B, 1, H, W]
            depth = F.interpolate(depth, size=(img_h, img_w), mode='bilinear', align_corners=False)

            # Apply threshold if > 0 (otherwise use raw depth as gradient mask)
            if depth_threshold > 0:
                # Hard threshold: above = 1 (keep), below = 0 (fill)
                work_mask = (depth > depth_threshold).float()
            else:
                # Use raw depth values as gradient mask
                work_mask = depth
        else:
            # Create preset mask at image resolution
            work_mask = torch.ones((1, 1, img_h, img_w), device=image.device, dtype=image.dtype)

            if mask_mode == "top_half":
                work_mask[:, :, img_h//2:, :] = 0
            elif mask_mode == "bottom_half":
                work_mask[:, :, :img_h//2, :] = 0
            elif mask_mode == "left_half":
                work_mask[:, :, :, img_w//2:] = 0
            elif mask_mode == "right_half":
                work_mask[:, :, :, :img_w//2] = 0
            elif mask_mode == "center":
                work_mask[:, :, :img_h//4, :] = 0
                work_mask[:, :, 3*img_h//4:, :] = 0
                work_mask[:, :, :, :img_w//4] = 0
                work_mask[:, :, :, 3*img_w//4:] = 0
            elif mask_mode == "edges":
                work_mask[:, :, img_h//4:3*img_h//4, img_w//4:3*img_w//4] = 0
            elif mask_mode == "gradient_top":
                for h in range(img_h):
                    work_mask[:, :, h, :] = 1.0 - (h / img_h)
            elif mask_mode == "gradient_bottom":
                for h in range(img_h):
                    work_mask[:, :, h, :] = h / img_h
            elif mask_mode == "gradient_left":
                for w in range(img_w):
                    work_mask[:, :, :, w] = 1.0 - (w / img_w)
            elif mask_mode == "gradient_right":
                for w in range(img_w):
                    work_mask[:, :, :, w] = w / img_w
            elif mask_mode == "vignette":
                cy, cx = img_h // 2, img_w // 2
                max_dist = ((img_h/2)**2 + (img_w/2)**2) ** 0.5
                for h in range(img_h):
                    for w in range(img_w):
                        dist = ((h - cy)**2 + (w - cx)**2) ** 0.5
                        work_mask[:, :, h, w] = 1.0 - min(dist / max_dist, 1.0)

        # Apply feathering
        # Use replicate padding to avoid feathering canvas edges
        if feather > 0:
            max_kernel = max(3, min(img_h, img_w) // 4)
            kernel_size = max(3, int(feather * max_kernel))
            if kernel_size % 2 == 0:
                kernel_size += 1
            pad_size = kernel_size // 2
            for _ in range(3):
                # Replicate padding keeps edge values solid
                work_mask = F.pad(work_mask, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
                work_mask = F.avg_pool2d(work_mask, kernel_size, stride=1, padding=0)
            work_mask = torch.clamp(work_mask, 0.0, 1.0)

        # Apply mask strength
        if mask_strength < 1.0:
            work_mask = work_mask * mask_strength + (1.0 - mask_strength)

        # Invert if requested
        if invert_mask:
            work_mask = 1.0 - work_mask

        # Early exit if mask is all 1s (full protection, no changes needed)
        if work_mask.min() > 0.999:
            mask_out = work_mask.squeeze(1)  # [B, H, W] for output
            return (positive, negative, mask_out)

        # Create fill image (grey with optional tint)
        fill_image = torch.ones_like(image) * fill_brightness

        # Apply color tint if enabled
        if tint_fill and tint_color:
            # Parse hex color (strip # if present)
            hex_str = tint_color.strip().lstrip('#')
            if len(hex_str) == 6:
                try:
                    r = int(hex_str[0:2], 16) / 255.0
                    g = int(hex_str[2:4], 16) / 255.0
                    b = int(hex_str[4:6], 16) / 255.0
                    tint_rgb = torch.tensor([r, g, b], device=image.device, dtype=image.dtype)

                    # Subtle blend towards tint (7% tint influence)
                    tint_strength = 0.07
                    # fill_image is [B, H, W, C], blend each channel towards tint
                    for c in range(min(3, fill_image.shape[-1])):
                        fill_image[..., c] = fill_image[..., c] * (1 - tint_strength) + tint_rgb[c] * tint_strength
                except ValueError:
                    pass  # Invalid hex, skip tint

        # Resize to match latent dimensions
        target_h = H * 8
        target_w = W * 8
        fill_image_resized = comfy.utils.common_upscale(
            fill_image.movedim(-1, 1),
            target_w, target_h,
            "bilinear", "center"
        ).movedim(1, -1)

        # Expand to video length
        length = T * 4
        video_frames = fill_image_resized.unsqueeze(1).expand(-1, length, -1, -1, -1)
        video_frames = video_frames.reshape(-1, target_h, target_w, video_frames.shape[-1])

        # VAE encode the fill
        fill_latent = vae.encode(video_frames[:, :, :, :3])
        if fill_latent.shape[2] != T:
            fill_latent = fill_latent[:, :, :T]

        # Resize mask to latent space
        mask_latent = F.interpolate(work_mask, size=(H, W), mode='bilinear', align_corners=False)
        # Expand mask to full latent shape - affects all temporal frames
        mask_latent = mask_latent.unsqueeze(2).expand(-1, C, T, -1, -1)

        # Process conditioning
        def update_conditioning(cond_list, scale_text=False):
            result = []
            for text_cond, cond_dict in cond_list:
                new_dict = copy.copy(cond_dict)

                # Scale text embeddings if requested (positive only)
                if scale_text and text_strength != 1.0:
                    text_cond = text_cond * text_strength

                # Blend: original * mask + fill * (1-mask)
                this_orig = cond_dict.get("concat_latent_image", original_latent)
                this_blended = this_orig * mask_latent + fill_latent * (1.0 - mask_latent)
                new_dict["concat_latent_image"] = this_blended

                # Update concat_mask to tell model to generate freely in filled areas
                if "concat_mask" in new_dict:
                    concat_mask = new_dict["concat_mask"].clone()
                    cm_h, cm_w = concat_mask.shape[-2], concat_mask.shape[-1]
                    cm_t = concat_mask.shape[2]  # Temporal dimension

                    mask_resized = F.interpolate(
                        work_mask,
                        size=(cm_h, cm_w),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Where mask is 0 (fill), set concat_mask to 1 (generate freely) - all frames
                    inv_mask = 1.0 - mask_resized
                    inv_mask = inv_mask.unsqueeze(2).expand(-1, -1, cm_t, -1, -1)
                    concat_mask = torch.max(concat_mask, inv_mask)

                    new_dict["concat_mask"] = concat_mask

                result.append([text_cond, new_dict])
            return result

        new_positive = update_conditioning(positive, scale_text=True)
        new_negative = update_conditioning(negative, scale_text=False)

        # Output mask for visualization/chaining
        mask_out = work_mask.squeeze(1)  # [B, H, W]

        return (new_positive, new_negative, mask_out)


class DropFirstFrames:
    """
    Drops the first N frames from an image batch.
    Useful for removing initial flash/glitch frames from I2V generations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frames_to_drop": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of frames to remove from the start. 0 = passthrough (disabled)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "drop_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Removes first N frames from image batch"

    def drop_frames(self, images, frames_to_drop=4):
        # images shape: [B, H, W, C] where B is frame count
        if frames_to_drop == 0:
            return (images,)

        if images.shape[0] <= frames_to_drop:
            # Don't drop all frames, keep at least 1
            return (images[-1:],)

        return (images[frames_to_drop:],)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanI2VConditioningMaskPro": WanI2VConditioningMaskPro,
    "DropFirstFrames": DropFirstFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanI2VConditioningMaskPro": "Wan I2V Conditioning Mask Pro",
    "DropFirstFrames": "Drop First Frames",
}
