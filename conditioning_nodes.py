"""
Wan I2V Conditioning Manipulation Nodes
=======================================

Person mask generation uses Google MediaPipe for segmentation and face landmarks.
https://ai.google.dev/edge/mediapipe/solutions/guide
"""

import torch
import torch.nn.functional as F
import copy
import os
import numpy as np
from PIL import Image

# Conditional import of MediaPipe for person/face segmentation
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def get_mediapipe_model_path(model_name: str) -> str:
    """
    Get path to MediaPipe model, downloading if needed.
    Models are stored in ComfyUI's models/mediapipe/ directory.
    """
    import folder_paths

    # Create mediapipe model directory
    models_dir = os.path.join(folder_paths.models_dir, "mediapipe")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        # Download model from MediaPipe
        model_urls = {
            "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        }

        if model_name not in model_urls:
            raise ValueError(f"Unknown MediaPipe model: {model_name}")

        url = model_urls[model_name]
        print(f"[WanI2V] Downloading MediaPipe model: {model_name}")

        import urllib.request
        urllib.request.urlretrieve(url, model_path)
        print(f"[WanI2V] Downloaded to: {model_path}")

    return model_path


# MediaPipe segmentation class indices (selfie_multiclass_256x256)
SEGMENT_BACKGROUND = 0
SEGMENT_HAIR = 1
SEGMENT_BODY_SKIN = 2
SEGMENT_FACE_SKIN = 3
SEGMENT_CLOTHES = 4

# Face mesh landmark indices for different regions
# Based on MediaPipe face mesh topology
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

LEFT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

RIGHT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]

LEFT_EYEBROW_INDICES = [
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336
]

RIGHT_EYEBROW_INDICES = [
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107
]

LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78
]

# Pupils - approximate center points
LEFT_PUPIL_INDICES = [468, 469, 470, 471, 472]  # Left iris landmarks
RIGHT_PUPIL_INDICES = [473, 474, 475, 476, 477]  # Right iris landmarks


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
                    "tooltip": "Custom mask. White=fill/regenerate, Black=keep. Overrides mask_mode."
                }),
                "depth_map": ("IMAGE", {
                    "tooltip": "Depth map image. White (close)=fill, Black (far)=keep."
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
                    "tooltip": "Depth cutoff. 0=use depth as gradient. Higher values=fill more of the foreground. 0.5=fill close objects, keep distant."
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
                # Person Mask Generation (based on a-person-mask-generator by David Bielejeski)
                "generate_person_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable built-in person/face mask generation using MediaPipe. Takes priority over external mask input."
                }),
                # Person Segmentation options
                "mask_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include face skin area in mask (requires generate_person_mask)"
                }),
                "mask_hair": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include hair in mask (requires generate_person_mask)"
                }),
                "mask_body": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include body/skin in mask (requires generate_person_mask)"
                }),
                "mask_clothes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include clothing in mask (requires generate_person_mask)"
                }),
                "mask_background": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include background in mask (requires generate_person_mask)"
                }),
                # Face Landmark options
                "mask_face_oval": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include full face oval outline (requires generate_person_mask)"
                }),
                "mask_eyes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include both eyes in mask (requires generate_person_mask)"
                }),
                "mask_eyebrows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include both eyebrows in mask (requires generate_person_mask)"
                }),
                "mask_lips": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include lips/mouth in mask (requires generate_person_mask)"
                }),
                "mask_pupils": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include pupils only in mask (requires generate_person_mask)"
                }),
                # Detection settings
                "mask_confidence": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold for person mask generation"
                }),
                "refine_mask_detection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Crop and re-run detection for better accuracy on smaller faces"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "MASK",)
    RETURN_NAMES = ("positive", "negative", "mask_out",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/Wan"
    DESCRIPTION = "Full-featured pixel-space masking with VAE re-encoding"

    def _generate_person_segmentation_mask(self, image_np: np.ndarray,
                                            include_face: bool, include_hair: bool,
                                            include_body: bool, include_clothes: bool,
                                            include_background: bool,
                                            confidence: float) -> np.ndarray:
        """
        Generate person segmentation mask using MediaPipe selfie_multiclass model.
        Returns mask where detected regions = 1.0, undetected = 0.0
        """
        if not MEDIAPIPE_AVAILABLE:
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

        model_path = get_mediapipe_model_path("selfie_multiclass_256x256.tflite")

        # Create segmenter
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            running_mode=mp_vision.RunningMode.IMAGE
        )

        with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
            # Convert to MediaPipe image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

            # Run segmentation
            result = segmenter.segment(mp_image)

            if not result.category_mask:
                return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

            # Get category mask (each pixel has class index)
            category_mask = result.category_mask.numpy_view()
            # Ensure 2D (some versions return 3D with extra dim)
            if len(category_mask.shape) == 3:
                category_mask = category_mask[:, :, 0]

            # Build combined mask from selected categories
            combined_mask = np.zeros_like(category_mask, dtype=np.float32)

            if include_background:
                combined_mask = np.where(category_mask == SEGMENT_BACKGROUND, 1.0, combined_mask)
            if include_hair:
                combined_mask = np.where(category_mask == SEGMENT_HAIR, 1.0, combined_mask)
            if include_body:
                combined_mask = np.where(category_mask == SEGMENT_BODY_SKIN, 1.0, combined_mask)
            if include_face:
                combined_mask = np.where(category_mask == SEGMENT_FACE_SKIN, 1.0, combined_mask)
            if include_clothes:
                combined_mask = np.where(category_mask == SEGMENT_CLOTHES, 1.0, combined_mask)

            return combined_mask

    def _generate_face_landmark_mask(self, image_np: np.ndarray,
                                      include_face_oval: bool, include_eyes: bool,
                                      include_eyebrows: bool, include_lips: bool,
                                      include_pupils: bool,
                                      confidence: float) -> np.ndarray:
        """
        Generate face landmark mask using MediaPipe face mesh.
        Returns mask where detected regions = 1.0, undetected = 0.0
        """
        if not MEDIAPIPE_AVAILABLE:
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

        # Check if any landmark options are enabled
        if not any([include_face_oval, include_eyes, include_eyebrows, include_lips, include_pupils]):
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

        model_path = get_mediapipe_model_path("face_landmarker.task")

        # Create face landmarker
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            num_faces=10,  # Support multiple faces
            running_mode=mp_vision.RunningMode.IMAGE
        )

        h, w = image_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)

        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            # Convert to MediaPipe image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

            # Run face landmark detection
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                return combined_mask

            # Process each detected face
            for face_landmarks in result.face_landmarks:
                # Convert landmarks to pixel coordinates
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

                # Draw filled polygons for each selected region
                from PIL import ImageDraw

                mask_img = Image.new('L', (w, h), 0)
                draw = ImageDraw.Draw(mask_img)

                if include_face_oval:
                    points = [landmarks[i] for i in FACE_OVAL_INDICES if i < len(landmarks)]
                    if len(points) >= 3:
                        draw.polygon(points, fill=255)

                if include_eyes:
                    # Left eye
                    left_eye_pts = [landmarks[i] for i in LEFT_EYE_INDICES if i < len(landmarks)]
                    if len(left_eye_pts) >= 3:
                        draw.polygon(left_eye_pts, fill=255)
                    # Right eye
                    right_eye_pts = [landmarks[i] for i in RIGHT_EYE_INDICES if i < len(landmarks)]
                    if len(right_eye_pts) >= 3:
                        draw.polygon(right_eye_pts, fill=255)

                if include_eyebrows:
                    # Left eyebrow
                    left_brow_pts = [landmarks[i] for i in LEFT_EYEBROW_INDICES if i < len(landmarks)]
                    if len(left_brow_pts) >= 3:
                        draw.polygon(left_brow_pts, fill=255)
                    # Right eyebrow
                    right_brow_pts = [landmarks[i] for i in RIGHT_EYEBROW_INDICES if i < len(landmarks)]
                    if len(right_brow_pts) >= 3:
                        draw.polygon(right_brow_pts, fill=255)

                if include_lips:
                    lip_pts = [landmarks[i] for i in LIPS_INDICES if i < len(landmarks)]
                    if len(lip_pts) >= 3:
                        draw.polygon(lip_pts, fill=255)

                if include_pupils:
                    # Draw small circles for pupils
                    pupil_radius = max(2, int(min(w, h) * 0.01))
                    for pupil_indices in [LEFT_PUPIL_INDICES, RIGHT_PUPIL_INDICES]:
                        valid_pts = [landmarks[i] for i in pupil_indices if i < len(landmarks)]
                        if valid_pts:
                            # Average the iris landmarks to find center
                            cx = sum(p[0] for p in valid_pts) // len(valid_pts)
                            cy = sum(p[1] for p in valid_pts) // len(valid_pts)
                            draw.ellipse(
                                [cx - pupil_radius, cy - pupil_radius,
                                 cx + pupil_radius, cy + pupil_radius],
                                fill=255
                            )

                # Merge this face's mask
                face_mask = np.array(mask_img, dtype=np.float32) / 255.0
                combined_mask = np.maximum(combined_mask, face_mask)

            return combined_mask

    def _generate_combined_person_mask(self, image: torch.Tensor,
                                        mask_face: bool, mask_hair: bool,
                                        mask_body: bool, mask_clothes: bool,
                                        mask_background: bool,
                                        mask_face_oval: bool, mask_eyes: bool,
                                        mask_eyebrows: bool, mask_lips: bool,
                                        mask_pupils: bool,
                                        confidence: float,
                                        refine_detection: bool) -> torch.Tensor:
        """
        Generate combined person/face mask from image tensor.
        Returns mask tensor [1, 1, H, W] where detected regions = 1.0
        """
        # Convert image tensor to numpy (assume [B, H, W, C] format, 0-1 range)
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]

        # Initialize combined mask
        combined_mask = np.zeros((h, w), dtype=np.float32)

        # Check if any segmentation options are enabled
        use_segmentation = any([mask_face, mask_hair, mask_body, mask_clothes, mask_background])
        use_landmarks = any([mask_face_oval, mask_eyes, mask_eyebrows, mask_lips, mask_pupils])

        # Generate person segmentation mask
        if use_segmentation:
            seg_mask = self._generate_person_segmentation_mask(
                img_np, mask_face, mask_hair, mask_body, mask_clothes, mask_background, confidence
            )
            combined_mask = np.maximum(combined_mask, seg_mask)

        # Generate face landmark mask
        if use_landmarks:
            landmark_mask = self._generate_face_landmark_mask(
                img_np, mask_face_oval, mask_eyes, mask_eyebrows, mask_lips, mask_pupils, confidence
            )
            combined_mask = np.maximum(combined_mask, landmark_mask)

            # Refine detection by cropping to face region and re-running
            if refine_detection and np.any(landmark_mask > 0):
                # Find bounding box of detected face regions
                ys, xs = np.where(landmark_mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = max(0, xs.min() - 50), min(w, xs.max() + 50)
                    y1, y2 = max(0, ys.min() - 50), min(h, ys.max() + 50)

                    # Only refine if crop is significantly smaller
                    if (x2 - x1) < w * 0.8 or (y2 - y1) < h * 0.8:
                        crop = img_np[y1:y2, x1:x2]
                        crop_h, crop_w = crop.shape[:2]

                        # Re-run detection on cropped region
                        refined_mask = self._generate_face_landmark_mask(
                            crop, mask_face_oval, mask_eyes, mask_eyebrows,
                            mask_lips, mask_pupils, confidence
                        )

                        # Place refined mask back in full image coords
                        if np.any(refined_mask > 0):
                            combined_mask[y1:y2, x1:x2] = np.maximum(
                                combined_mask[y1:y2, x1:x2], refined_mask
                            )

        # Convert to torch tensor [1, 1, H, W]
        mask_tensor = torch.from_numpy(combined_mask).float()
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.to(image.device)

        return mask_tensor

    def apply_mask(self, positive, negative, vae, image, mask=None, depth_map=None,
                   mask_mode="full", depth_threshold=0.5, fill_brightness=0.5,
                   tint_fill=False, tint_color="",
                   mask_strength=1.0, feather=0.0, invert_mask=False,
                   text_strength=1.0,
                   # Person mask generation parameters
                   generate_person_mask=False,
                   mask_face=True, mask_hair=False, mask_body=False,
                   mask_clothes=False, mask_background=False,
                   mask_face_oval=False, mask_eyes=False, mask_eyebrows=False,
                   mask_lips=False, mask_pupils=False,
                   mask_confidence=0.4, refine_mask_detection=True):
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
        # Priority: generate_person_mask > mask > depth_map > mask_mode preset
        if generate_person_mask:
            # Generate person/face mask using MediaPipe
            if not MEDIAPIPE_AVAILABLE:
                print("[WanI2V] Warning: MediaPipe not available. Install with: pip install mediapipe")
                print("[WanI2V] Falling back to other mask sources...")
                # Fall through to next priority
                generate_person_mask = False

        if generate_person_mask:
            # Generate internal person mask - ignore external mask input
            work_mask = self._generate_combined_person_mask(
                image,
                mask_face=mask_face, mask_hair=mask_hair,
                mask_body=mask_body, mask_clothes=mask_clothes,
                mask_background=mask_background,
                mask_face_oval=mask_face_oval, mask_eyes=mask_eyes,
                mask_eyebrows=mask_eyebrows, mask_lips=mask_lips,
                mask_pupils=mask_pupils,
                confidence=mask_confidence,
                refine_detection=refine_mask_detection
            )
            # Invert so detected = 0 (fill), undetected = 1 (keep)
            # This matches the convention: detected areas will be regenerated
            work_mask = 1.0 - work_mask
        elif mask is not None:
            # Use custom mask (invert so white=fill, black=keep)
            work_mask = mask.clone()
            if len(work_mask.shape) == 2:
                work_mask = work_mask.unsqueeze(0)
            if len(work_mask.shape) == 3:
                work_mask = work_mask.unsqueeze(1)
            # Resize to image dimensions
            work_mask = F.interpolate(work_mask, size=(img_h, img_w), mode='bilinear', align_corners=False)
            # Invert: white input (1) → fill (0), black input (0) → keep (1)
            work_mask = 1.0 - work_mask
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
            # Invert so bright/close = fill, dark/far = keep
            if depth_threshold > 0:
                # Hard threshold: above threshold = fill (0), below = keep (1)
                work_mask = (depth <= depth_threshold).float()
            else:
                # Use inverted depth values as gradient mask
                work_mask = 1.0 - depth
        else:
            # Create preset mask at image resolution
            work_mask = torch.ones((1, 1, img_h, img_w), device=image.device, dtype=image.dtype)

            if mask_mode == "top_half":
                work_mask[:, :, :img_h//2, :] = 0  # Fill top half
            elif mask_mode == "bottom_half":
                work_mask[:, :, img_h//2:, :] = 0  # Fill bottom half
            elif mask_mode == "left_half":
                work_mask[:, :, :, :img_w//2] = 0  # Fill left half
            elif mask_mode == "right_half":
                work_mask[:, :, :, img_w//2:] = 0  # Fill right half
            elif mask_mode == "center":
                work_mask[:, :, img_h//4:3*img_h//4, img_w//4:3*img_w//4] = 0  # Fill center
            elif mask_mode == "edges":
                work_mask[:, :, :img_h//4, :] = 0  # Fill edges
                work_mask[:, :, 3*img_h//4:, :] = 0
                work_mask[:, :, :, :img_w//4] = 0
                work_mask[:, :, :, 3*img_w//4:] = 0
            elif mask_mode == "gradient_top":
                for h in range(img_h):
                    work_mask[:, :, h, :] = h / img_h  # 0 at top (fill), 1 at bottom (keep)
            elif mask_mode == "gradient_bottom":
                for h in range(img_h):
                    work_mask[:, :, h, :] = 1.0 - (h / img_h)  # 1 at top (keep), 0 at bottom (fill)
            elif mask_mode == "gradient_left":
                for w in range(img_w):
                    work_mask[:, :, :, w] = w / img_w  # 0 at left (fill), 1 at right (keep)
            elif mask_mode == "gradient_right":
                for w in range(img_w):
                    work_mask[:, :, :, w] = 1.0 - (w / img_w)  # 1 at left (keep), 0 at right (fill)
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
