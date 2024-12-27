import torch
import numpy as np

from omni import OmniImage
from utils import *

class OmniRegionCropper(object):
    """
    OmniRegionCropper is part of a deprecated 360 tracking framework designed for processing
    omnidirectional (spherical) video or image data. It integrates with the XMem 
    (https://github.com/hkchengrex/XMem/tree/main) and provides functionality for cropping, remapping,
    and transforming images and masks based on a bounding field of view (bfov).

    Notes:
    - The framework is no longer maintained and has been replaced by `framework_torch.py`.
    - The integration XMem-360 is available at:
      https://github.com/XuYinzhe/XMem-360/tree/main.
    - Handles omnidirectional (360°) image processing using `OmniImage` for spherical geometry.
    - Maintains a `pre_bfov` state to store the previous bounding FOV for continuity between frames.
    - Extends field of view (FOV) dynamically, with constraints of 180° vertical and 360° horizontal.
    - Supports torch-numpy interoperability for processing with PyTorch and NumPy.
    - Designed to preprocess images/masks for XMem and post-process predictions for uncropping.
    """

    def __init__(self, ori_width=3840, ori_height=1920, out_width=1920, out_height=960, extend_ang=135):
        """
        Initializes the cropper with original and output dimensions, as well as FOV extension angles.
        - ori_width, ori_height: Size of the original omnidirectional image.
        - out_width, out_height: Size of the cropped output image.
        - extend_ang: Angle in degrees to extend the FOV for cropping (single value or tuple).
        """
        self.omniImage = OmniImage(img_h=ori_height, img_w=ori_width)
        self.ori_width = ori_width
        self.ori_height = ori_height
        self.out_width = out_width
        self.out_height = out_height
        
        self.pre_bfov: Bfov = None  # Stores the previous bounding FOV for continuity.
        
        if isinstance(extend_ang, (int, float)):
            self.extend_ang = (extend_ang, extend_ang)
        elif isinstance(extend_ang, tuple):
            self.extend_ang = extend_ang
            
    def clean_bfov(self):
        """
        Resets the previous bounding FOV (pre_bfov) to None.
        """
        self.pre_bfov: Bfov = None
        
    def extend_bfov(self, bfov: Bfov):
        """
        Dynamically extends the given bounding FOV (bfov) based on the configured extend_ang.
        Ensures the FOV does not exceed 180° vertical or 360° horizontal.
        """
        ang_v, ang_h = self.extend_ang
        ang_h *= 1.8  # Scales horizontal FOV extension.

        ang_v = min(180, ang_v)
        ang_h = min(360, ang_h)

        bfov.fov_h = ang_h if bfov.fov_h < ang_h else bfov.fov_h
        bfov.fov_v = ang_v if bfov.fov_v < ang_v else bfov.fov_v
        
        return bfov
    
    def process_rgbwmsk(self, rgb, msk):
        """
        Processes an RGB image and a mask to compute the bounding FOV (bfov) and crop the relevant
        region from both. Updates pre_bfov for continuity.
        """
        bfov = self.omniImage.mask2Bfov(msk, need_rotation=False)  # Compute bfov from the mask.
        self.pre_bfov = self.extend_bfov(bfov)  # Extend the FOV.
        rgb_remap, _, _ = self.omniImage.crop_bfov(
            rgb, self.pre_bfov, 
            num_sample_h=self.out_width, 
            num_sample_v=self.out_height
        )
        msk_remap, _, _ = self.omniImage.crop_bfov(
            msk, self.pre_bfov, 
            num_sample_h=self.out_width, 
            num_sample_v=self.out_height
        )
        return rgb_remap, msk_remap
    
    def numpy2torch(self, rgb, msk):
        """
        Converts RGB and mask data from NumPy format to PyTorch tensors.
        - RGB remains unchanged.
        - Mask is converted to a tensor and reshaped with an additional channel dimension.
        """
        res_rgb = rgb
        res_msk = torch.from_numpy(msk).unsqueeze(0) if msk is not None else None
        return res_rgb, res_msk
    
    def torch2numpy(self, rgb, msk):
        """
        Converts RGB and mask data from PyTorch tensors to NumPy arrays.
        - Removes the channel dimension from the mask if present.
        """
        np_rgb = np.array(rgb)
        np_msk = np.array(msk).squeeze(0) if msk is not None else None
        return np_rgb, np_msk
        
    def __call__(self, _rgb, _msk):
        """
        Main processing pipeline for cropping RGB and mask data:
        - Converts input data to NumPy format.
        - Computes the crop based on the bounding FOV (pre_bfov).
        - Converts the cropped data back to PyTorch tensors.
        """
        if _msk is not None:
            rgb, msk = self.torch2numpy(_rgb, _msk)
            rgb_remap, msk_remap = self.process_rgbwmsk(rgb, msk)
            res_rgb, res_msk = self.numpy2torch(rgb_remap, msk_remap)
        else:
            rgb, _ = self.torch2numpy(_rgb, _msk)
            rgb_remap, _, _ = self.omniImage.crop_bfov(
                rgb, self.pre_bfov, 
                num_sample_h=self.out_width, 
                num_sample_v=self.out_height
            )
            res_rgb, res_msk = self.numpy2torch(rgb_remap, None)
        
        return res_rgb, res_msk
    
    def insert_pred(self, pred):
        """
        Handles post-processing of predictions (e.g., segmentation maps):
        - Uncrops the prediction back to the original image.
        - Computes a new bounding FOV (bfov) from the uncropped prediction and updates pre_bfov.
        - If uncropping fails, returns an empty array and marks the prediction as invalid.
        """
        valid_pred = True
        try:
            uncrop_rgb, _, _ = self.omniImage.uncrop_bfov(
                pred, self.pre_bfov, out_width=self.ori_width, out_height=self.ori_height
            )
            uncrop_rgb = uncrop_rgb.astype('uint8')
            bfov = self.omniImage.mask2Bfov(uncrop_rgb, need_rotation=False)
            if bfov: 
                self.pre_bfov = self.extend_bfov(bfov)
        except:
            uncrop_rgb = np.zeros((self.ori_height, self.ori_width), dtype='uint8')
            valid_pred = False
        
        return uncrop_rgb, valid_pred