"""TryOn Pipeline."""

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fashn_human_parser import (
    CATEGORY_TO_BODY_COVERAGE,
    FashnHumanParser,
)

from PIL import Image
from tqdm.auto import tqdm

from .dwpose import DWposeDetector, draw_pose

from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)

from .tryon_mmdit import TryOnModel

from .utils import (
    get_dummy_dw_keypoints,
    get_rf_schedule,
    load_checkpoint,
    normalize_uint8_to_neg1_1,
    numpy_to_torch,
    setup_logger,
    tensor_to_pil,
)


@dataclass
class PipelineOutput:
    images: List[Image.Image]


class TryOnPipeline:

    CATEGORY_TO_LABEL = {
        "tops": 1,
        "bottoms": 2,
        "one-pieces": 3,
    }

    def __init__(
        self,
        weights_dir: str,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):

        self.weights_dir = os.path.abspath(weights_dir)

        self.logger = logger or setup_logger(
            "TryOnPipeline"
        )

        self.device = torch.device(
            device if device else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.logger.info(
            f"Using device: {self.device}"
        )

        self.inference_dtype = torch.float32

        if (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            self.inference_dtype = torch.bfloat16

        self.logger.info(
            f"Using dtype: {self.inference_dtype}"
        )

        self._validate_weights()

        self._setup_tryon_model()
        self._setup_pose_model()
        self._setup_hp_model()

        h, w = self.tryon_model.input_shape

        self.target_h = 768
        self.target_w = 512

        max_dim = max(h, w)

        self.pre_resize = AspectPreserveResize(
            target_size=(max_dim, max_dim),
            mode="fit",
            backend="pil",
        )

        self.resize_pad_fn = ResizePad(
            (self.target_w, self.target_h),
            backend="opencv",
        )

    def _validate_weights(self):

        tryon_path = os.path.join(
            self.weights_dir,
            "model.safetensors",
        )

        dwpose_dir = os.path.join(
            self.weights_dir,
            "dwpose",
        )

        yolox_path = os.path.join(
            dwpose_dir,
            "yolox_l.onnx",
        )

        dwpose_path = os.path.join(
            dwpose_dir,
            "dw-ll_ucoco_384.onnx",
        )

        missing = []

        if not os.path.exists(tryon_path):
            missing.append(tryon_path)

        if not os.path.exists(yolox_path):
            missing.append(yolox_path)

        if not os.path.exists(dwpose_path):
            missing.append(dwpose_path)

        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(
                    f"  - {p}" for p in missing
                )
            )

    def _setup_tryon_model(self):

        model_path = os.path.join(
            self.weights_dir,
            "model.safetensors",
        )

        self.logger.info(
            f"Loading TryOnModel from {model_path}"
        )

        self.tryon_model = TryOnModel()

        state_dict = load_checkpoint(
            model_path,
            device=str(self.device),
        )

        missing, unexpected = (
            self.tryon_model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        self.logger.info(
            f"Missing keys: {missing}"
        )

        self.logger.info(
            f"Unexpected keys: {unexpected}"
        )

        self.tryon_model.to(
            self.device,
            dtype=self.inference_dtype,
        ).eval()

        self.logger.info(
            "TryOnModel loaded"
        )

    def _setup_pose_model(self):

        dwpose_dir = os.path.join(
            self.weights_dir,
            "dwpose",
        )

        dwpose_device = (
            f"cuda:{self.device.index or 0}"
            if self.device.type == "cuda"
            else "cpu"
        )

        self.pose_model = DWposeDetector(
            checkpoints_dir=dwpose_dir,
            device=dwpose_device,
        )

        self.logger.info(
            "DWPose loaded"
        )

    def _setup_hp_model(self):

        hp_device = (
            "cuda"
            if self.device.type == "cuda"
            else "cpu"
        )

        self.hp_model = FashnHumanParser(
            device=hp_device
        )

        self.logger.info(
            "FashnHumanParser loaded"
        )

    def _resize_exact(
        self,
        img: np.ndarray,
        is_mask: bool = False,
    ):

        interpolation = (
            cv2.INTER_NEAREST
            if is_mask
            else cv2.INTER_LINEAR
        )

        img = cv2.resize(
            img,
            (
                self.target_w,
                self.target_h,
            ),
            interpolation=interpolation,
        )

        return img

    def _prepare_rgb_tensor(
        self,
        img: np.ndarray,
    ):

        img = self._resize_exact(img)

        img = img.astype(np.uint8)

        t = numpy_to_torch(img)

        t = t.unsqueeze(0)

        t = normalize_uint8_to_neg1_1(t)

        t = t.to(
            self.device,
            dtype=self.inference_dtype,
        )

        return t

    def _prepare_gray_tensor(
        self,
        img: np.ndarray,
    ):

        img = self._resize_exact(
            img,
            is_mask=True,
        )

        if len(img.shape) == 3:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY,
            )

        img = img.astype(np.float32)

        img = img / 127.5 - 1.0

        t = torch.from_numpy(img)

        t = t.unsqueeze(0)
        t = t.unsqueeze(0)

        t = t.to(
            self.device,
            dtype=self.inference_dtype,
        )

        return t

    def _fix_tensor_size(
        self,
        tensor: torch.Tensor,
    ):

        tensor = F.interpolate(
            tensor,
            size=(
                self.target_h,
                self.target_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        return tensor

    @torch.inference_mode()
    def _sample(
        self,
        *,
        ca_images: torch.Tensor,
        garment_images: torch.Tensor,
        person_poses: torch.Tensor,
        garment_poses: torch.Tensor,
        garment_categories: torch.Tensor,
        num_timesteps: int = 10,
        time_shift_mu: float = 1.5,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
    ):

        device = ca_images.device

        dtype = ca_images.dtype

        c, h, w = (
            self.tryon_model.channels_in,
            *self.tryon_model.input_shape,
        )

        images = torch.randn(
            (1, c, h, w),
            dtype=dtype,
            device=device,
        )

        timesteps = get_rf_schedule(
            num_steps=num_timesteps,
            mu=time_shift_mu,
        )

        model_kwargs = {
            "person_poses": person_poses,
            "garment_poses": garment_poses,
            "ca_images": ca_images,
            "garment_images": garment_images,
            "garment_categories": garment_categories,
        }

        for step_idx, (
            t_curr,
            t_prev,
        ) in enumerate(
            tqdm(
                zip(
                    timesteps[:-1],
                    timesteps[1:],
                ),
                total=len(timesteps) - 1,
                desc="Sampling",
            )
        ):

            dt = t_prev - t_curr

            t_vec = torch.full(
                (1,),
                t_curr,
                dtype=dtype,
                device=device,
            )

            pred = self.tryon_model.forward_for_cfg(
                images,
                t_vec,
                **model_kwargs,
            )

            v_c = pred["v_c"]
            v_u = pred["v_u"]

            # CRITICAL FIX
            if v_c.shape != images.shape:
                v_c = F.interpolate(
                    v_c,
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if v_u.shape != images.shape:
                v_u = F.interpolate(
                    v_u,
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if step_idx >= (
                num_timesteps
                - skip_cfg_last_n_steps
            ):
                v_guided = v_c
            else:
                v_guided = (
                    v_u
                    + guidance_scale
                    * (v_c - v_u)
                )

            images = images + dt * v_guided

        images = images.float().clamp_(
            -1.0,
            1.0,
        )

        return [
            tensor_to_pil(
                images[0],
                unnormalize=True,
            )
        ]

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Literal[
            "tops",
            "bottoms",
            "one-pieces",
        ],
        garment_photo_type: Literal[
            "model",
            "flat-lay",
        ] = "model",
        num_timesteps: int = 2,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
        seed: int = 42,
    ) -> PipelineOutput:

        torch.manual_seed(seed)

        np.random.seed(seed)

        person_image = self.pre_resize(
            person_image,
            allow_upsampling=False,
        )

        garment_image = self.pre_resize(
            garment_image,
            allow_upsampling=False,
        )

        person_image_np = np.array(
            person_image.convert("RGB")
        )

        garment_image_np = np.array(
            garment_image.convert("RGB")
        )

        person_pose = self.pose_model(
            person_image_np[..., ::-1]
        )

        garment_pose = (
            get_dummy_dw_keypoints()
            if garment_photo_type == "flat-lay"
            else self.pose_model(
                garment_image_np[..., ::-1]
            )
        )

        person_pose_img = draw_pose(
            person_pose,
            person_image_np.shape[0],
            person_image_np.shape[1],
            grayscale=True,
        )

        garment_pose_img = draw_pose(
            garment_pose,
            garment_image_np.shape[0],
            garment_image_np.shape[1],
            grayscale=True,
        )

        person_seg_pred = self.hp_model.predict(
            person_image_np
        )

        garment_seg_pred = self.hp_model.predict(
            garment_image_np
        )

        body_coverage = (
            CATEGORY_TO_BODY_COVERAGE.get(
                category
            )
        )

        labels_to_segment = (
            BODY_COVERAGE_TO_FASHN_LABELS.get(
                body_coverage
            )
        )

        labels_to_segment_indices = [
            FASHN_LABELS_TO_IDS[label]
            for label in labels_to_segment
        ]

        ca_image = (
            create_clothing_agnostic_image(
                img_np=person_image_np.copy(),
                seg_pred=person_seg_pred.copy(),
                labels_to_segment_indices=(
                    labels_to_segment_indices.copy()
                ),
                body_coverage=body_coverage,
                disable_masking=True,
                logger=self.logger,
            )
        )

        garment_image_processed = (
            create_garment_image(
                img_np=garment_image_np,
                seg_pred=garment_seg_pred,
                labels_to_segment_indices=(
                    labels_to_segment_indices.copy()
                ),
                disable_masking=(
                    garment_photo_type
                    == "flat-lay"
                ),
            )
        )

        ca_tensor = self._prepare_rgb_tensor(
            ca_image
        )

        garment_tensor = (
            self._prepare_rgb_tensor(
                garment_image_processed
            )
        )

        person_pose_tensor = (
            self._prepare_gray_tensor(
                person_pose_img
            )
        )

        garment_pose_tensor = (
            self._prepare_gray_tensor(
                garment_pose_img
            )
        )

        # FINAL SAFETY FIX
        ca_tensor = self._fix_tensor_size(
            ca_tensor
        )

        garment_tensor = self._fix_tensor_size(
            garment_tensor
        )

        person_pose_tensor = (
            self._fix_tensor_size(
                person_pose_tensor
            )
        )

        garment_pose_tensor = (
            self._fix_tensor_size(
                garment_pose_tensor
            )
        )

        self.logger.info(
            f"CA tensor shape: {ca_tensor.shape}"
        )

        self.logger.info(
            f"Garment tensor shape: "
            f"{garment_tensor.shape}"
        )

        self.logger.info(
            f"Person pose tensor shape: "
            f"{person_pose_tensor.shape}"
        )

        self.logger.info(
            f"Garment pose tensor shape: "
            f"{garment_pose_tensor.shape}"
        )

        garment_categories = torch.tensor(
            [self.CATEGORY_TO_LABEL[category]],
            device=self.device,
        )

        images = self._sample(
            ca_images=ca_tensor,
            garment_images=garment_tensor,
            person_poses=person_pose_tensor,
            garment_poses=garment_pose_tensor,
            garment_categories=garment_categories,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            skip_cfg_last_n_steps=(
                skip_cfg_last_n_steps
            ),
        )

        return PipelineOutput(
            images=images
        )