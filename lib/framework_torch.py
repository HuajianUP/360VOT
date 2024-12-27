import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np

from cc_torch import connected_components_labeling

from utils import *

class OmniImageTorch():
    def __init__(self, img_w=1920, img_h=960, device = 'cuda', check_device = True):
        """
        Initialize OmniImageTorch with image dimensions and device settings.

        Args:
            img_w (int): Image width in pixels.
            img_h (int): Image height in pixels.
            device (str): Compute device ('cuda' or 'cpu').
            check_device (bool): Whether to enforce device consistency.
        """
        
        self.device = torch.device('cuda') if device == 'cuda' else torch.device('cpu')
        self.check_device = check_device
    
        # numpy parameters
        self.img_w_np = int(img_w)
        self.img_h_np = int(img_h)
        
        self.fx_np = img_w / (2 * np.pi)
        self.fy_np = -img_h / np.pi
        self.cx_np = img_w / 2
        self.cy_np = img_h / 2

        # torch parameters
        self.fx = torch.tensor(self.fx_np, device=self.device, dtype=torch.float32)
        self.fy = torch.tensor(self.fy_np, device=self.device, dtype=torch.float32)
        self.cx = torch.tensor(self.cx_np, device=self.device, dtype=torch.float32)
        self.cy = torch.tensor(self.cy_np, device=self.device, dtype=torch.float32)
        self.img_w = torch.tensor(self.img_w_np, device=self.device, dtype=torch.float32)
        self.img_h = torch.tensor(self.img_h_np, device=self.device, dtype=torch.float32)
        self.min_mask = torch.tensor(8, device=self.device, dtype=torch.float32)

        # constants
        self.const_0 = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.const_0_5 = torch.tensor(0.5, device=self.device, dtype=torch.float32)
        self.const_2 = torch.tensor(2.0, device=self.device, dtype=torch.float32)
        self.const_360 = torch.tensor(360.0, device=self.device, dtype=torch.float32)
        self.const_180 = torch.tensor(180.0, device=self.device, dtype=torch.float32)
        self.const_neg90 = torch.tensor(-90.0, device=self.device, dtype=torch.float32)
        self.const_1 = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.const_neg1 = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        self.const_eps = torch.tensor(1e-6, device=self.device, dtype=torch.float32)
        self.const_255 = torch.tensor(255, device=self.device, dtype=torch.float32)

        # deprecated 
        self.rbbox_samples = torch.tensor(3000, device=self.device, dtype=torch.float32) # 2300

        # init xyz
        self.xyz = self._init_omni_image_cor()

    def _align_dim(self, *tensors, min_dim=2, max_dim=4, align_dim=4) -> torch.Tensor:
        """
        Align tensor dimensions to a specified number of dimensions.

        Args:
            *tensors: Variable number of tensors to align.
            min_dim (int): Minimum allowed dimensions.
            max_dim (int): Maximum allowed dimensions.
            align_dim (int): Target number of dimensions.

        Returns:
            Aligned tensor(s).
        """

        aligned_tensors = []
        for tensor in tensors:
            tensor = self._check_tensor(tensor).to(dtype=torch.float32)
            if tensor.dim() < min_dim or tensor.dim() > max_dim:
                raise ValueError(f"Tensor must have between {min_dim} and {max_dim} dimensions.")
            elif tensor.dim() == 3:
                tensor = tensor.permute(2, 0, 1)

            align_dim = max(min_dim, min(max_dim, align_dim))
            while tensor.dim() < align_dim: tensor = tensor.unsqueeze(0)
            while tensor.dim() > align_dim: tensor = tensor.squeeze(0)

            aligned_tensors.append(tensor)
        
        return aligned_tensors if len(aligned_tensors) > 1 else aligned_tensors[0]

    def _check_tensor(self, *args) -> torch.Tensor:
        """
        Ensure inputs are tensors on the correct device and dtype.

        Args:
            *args: Input tensors, numpy arrays, ints, or floats.

        Returns:
            Processed tensor(s).
        """

        tensors = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                if self.check_device:
                    arg = arg.to(self.device)
            elif isinstance(arg, (int, float)):
                arg = torch.tensor(arg, device=self.device if self.check_device else None)
            elif not isinstance(arg, torch.Tensor):
                raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, but got {type(arg)}")
            elif self.check_device and arg.device != self.device:
                arg = arg.to(self.device)
            tensors.append(arg)
        return tensors if len(tensors) > 1 else tensors[0]
    
    def _format_mask(self, mask, batch = False) -> torch.Tensor:
        """
        Format mask tensor to binary with shape (H, W) or (N, H, W).

        Args:
            mask (torch.Tensor or np.ndarray): Input mask.
            batch (bool): If True, treat mask as a batch.

        Returns:
            Formatted binary mask tensor.
        """

        mask = self._check_tensor(mask)
        if not mask.dim() == 2 and not mask.dim() == 3:
            raise ValueError(f"Tensor must have 2 or 3 dimensions.")
        
        # if input batch of masks, assume all masks are already in shape (H, W)
        if mask.dim() == 3 and not batch: mask = rgb_to_grayscale(mask, 2)

        mask = (mask.float() > self.const_0_5).float() * self.const_1

        return mask
    
    def _check_min_mask(self, mask, batch = False):
        """
        Check if mask regions meet the minimum size requirement.

        Args:
            mask (torch.Tensor): Binary mask.
            batch (bool): If True, treat mask as a batch.

        Returns:
            Boolean tensor indicating valid masks.
        """

        # if input batch of masks, assume in shape (N, H, W)
        mask = self._format_mask(mask, batch)
        return mask.sum(dim=(1, 2)) >= self.min_mask if batch else mask.sum()>=self.min_mask

    def uv2lonlat(self, u, v):
        """
        Convert pixel coordinates (u, v) to longitude and latitude.

        Args:
            u (torch.Tensor): Horizontal pixel coordinates.
            v (torch.Tensor): Vertical pixel coordinates.

        Returns:
            Tuple of tensors (longitude, latitude).
        """

        u, v = self._check_tensor(u, v)
        lon = ((u + self.const_0_5) - self.cx) / self.fx
        lat = ((v + self.const_0_5) - self.cy) / self.fy
        return lon, lat

    def lonlat2xyz(self, lon, lat):
        """
        Convert longitude and latitude to 3D Cartesian coordinates (x, y, z).

        Args:
            lon (torch.Tensor): Longitude angles.
            lat (torch.Tensor): Latitude angles.

        Returns:
            Tuple of tensors (x, y, z).
        """

        lon, lat = self._check_tensor(lon, lat)
        x = torch.cos(lat) * torch.sin(lon)
        y = torch.sin(-lat)
        z = torch.cos(lat) * torch.cos(lon)
        return x, y, z

    def uv2xyz(self, u, v):
        """
        Convert pixel coordinates (u, v) to 3D Cartesian coordinates (x, y, z).

        Args:
            u (torch.Tensor): Horizontal pixel coordinates.
            v (torch.Tensor): Vertical pixel coordinates.

        Returns:
            Tuple of tensors (x, y, z).
        """

        u, v = self._check_tensor(u, v)
        lon, lat = self.uv2lonlat(u, v)
        return self.lonlat2xyz(lon, lat)

    def xyz2lonlat(self, x, y, z, norm=False):
        """
        Convert 3D Cartesian coordinates to longitude and latitude.

        Args:
            x (torch.Tensor): X coordinates.
            y (torch.Tensor): Y coordinates.
            z (torch.Tensor): Z coordinates.
            norm (bool): If True, normalize latitude.

        Returns:
            Tuple of tensors (longitude, latitude).
        """

        x, y, z = self._check_tensor(x, y, z)
        lon = torch.atan2(x, z)
        if norm:
            lat = torch.asin(self.const_neg1 * y)
        else:
            lat = torch.atan2(self.const_neg1 * y, torch.sqrt(x**2 + z**2))
        return lon, lat

    def lonlat2uv(self, lon, lat):
        """
        Convert longitude and latitude to pixel coordinates (u, v).

        Args:
            lon (torch.Tensor): Longitude angles.
            lat (torch.Tensor): Latitude angles.

        Returns:
            Tuple of tensors (u, v).
        """

        lon, lat = self._check_tensor(lon, lat)
        u = lon * self.fx + self.cx - self.const_0_5
        v = lat * self.fy + self.cy - self.const_0_5
        return u, v

    def xyz2uv(self, x, y, z, norm=False):
        """
        Convert 3D Cartesian coordinates to pixel coordinates (u, v).

        Args:
            x (torch.Tensor): X coordinates.
            y (torch.Tensor): Y coordinates.
            z (torch.Tensor): Z coordinates.
            norm (bool): If True, normalize latitude.

        Returns:
            Tuple of tensors (u, v).
        """

        x, y, z = self._check_tensor(x, y, z)
        lon, lat = self.xyz2lonlat(x, y, z, norm)
        return self.lonlat2uv(lon, lat)

    def get_inverse_lonlat(self, R, u, v):
        """
        Apply inverse rotation to pixel coordinates and convert to longitude and latitude.

        Args:
            R (torch.Tensor): Rotation matrix.
            u (torch.Tensor): Horizontal pixel coordinates.
            v (torch.Tensor): Vertical pixel coordinates.

        Returns:
            Tuple of tensors (longitude, latitude).
        """
        
        R, u, v = self._check_tensor(R, u, v)
        x, y, z = self.uv2xyz(u, v)
        xyz = torch.matmul(R, torch.stack([x, y, z])).float()  # Apply rotation
        lon, lat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])
        return lon, lat

    def get_rough_FOV(self, bbox_w, bbox_h):
        """
        Estimate the field of view (FOV) based on bounding box dimensions.

        Args:
            bbox_w (torch.Tensor): Bounding box width in pixels.
            bbox_h (torch.Tensor): Bounding box height in pixels.

        Returns:
            Tuple of tensors (fov_width, fov_height) in degrees.
        """

        bbox_w, bbox_h = self._check_tensor(bbox_w, bbox_h)
        fov_w = bbox_w / self.img_w * self.const_360
        fov_h = bbox_h / self.img_h * self.const_180
        return fov_w, fov_h

    def rotate_zxy(self, lon, lat, rz):
        """
        Compute rotation matrix based on longitude, latitude, and rotation around Z-axis.

        Args:
            lon (torch.Tensor): Longitude angle in radians.
            lat (torch.Tensor): Latitude angle in radians.
            rz (torch.Tensor): Rotation angle around Z-axis in radians.

        Returns:
            Rotation matrix tensor of shape (3, 3).
        """

        if not (isinstance(lon, torch.Tensor) or isinstance(lat, torch.Tensor) or isinstance(rz, torch.Tensor)):
            R = rotate_y(lon) @ rotate_x(lat) @ rotate_z(rz)
            R = torch.from_numpy(R).to(device=self.device,dtype=torch.float32)
        else:
            lon, lat, rz = self._check_tensor(lon, lat, rz)
            # Precompute trigonometric terms
            cos_lon = torch.cos(lon)
            sin_lon = torch.sin(lon)
            cos_lat = torch.cos(lat)
            sin_lat = torch.sin(lat)
            cos_z = torch.cos(rz)
            sin_z = torch.sin(rz)

            # Rotation matrix around Z-axis
            Rz = torch.tensor([
                [cos_z, -sin_z, 0.0],
                [sin_z,  cos_z, 0.0],
                [0.0,    0.0,   1.0]
            ], dtype=torch.float32, device=self.device)

            # Rotation matrix around X-axis
            Rx = torch.tensor([
                [1.0,    0.0,       0.0],
                [0.0, cos_lat, -sin_lat],
                [0.0, sin_lat,  cos_lat]
            ], dtype=torch.float32, device=self.device)

            # Rotation matrix around Y-axis
            Ry = torch.tensor([
                [ cos_lon,  0.0, sin_lon],
                [ 0.0,      1.0, 0.0],
                [-sin_lon,  0.0, cos_lon]
            ], dtype=torch.float32, device=self.device)

            # Compute R = Ry @ (Rx @ Rz)
            R = torch.matmul(Ry, torch.mm(Rx, Rz)).float()

        return R

    def _init_omni_image_cor(self, fov_h=360, fov_v=180, num_sample_h=None, num_sample_v=None):
        """
        Initialize spherical coordinates for omni-directional images.

        Args:
            fov_h (float): Horizontal field of view in degrees.
            fov_v (float): Vertical field of view in degrees.
            num_sample_h (int): Number of horizontal samples.
            num_sample_v (int): Number of vertical samples.

        Returns:
            Tensor of shape (num_sample_v, num_sample_h, 3) representing (x, y, z).
        """

        fov_h = torch.deg2rad(fov_h) if isinstance(fov_h, torch.Tensor) else ang2rad(fov_h)
        fov_v = torch.deg2rad(fov_v) if isinstance(fov_v, torch.Tensor) else ang2rad(fov_v)

        ratio = fov_v / fov_h
        ratio = ratio.cpu().numpy() if isinstance(ratio, torch.Tensor) else ratio

        if num_sample_h is None: num_sample_h = self.img_w_np
        if num_sample_v is None: num_sample_v = int(num_sample_h * ratio)

        lon_range = fov_h / 2
        lat_range = fov_v / 2

        lon = torch.linspace(-lon_range, lon_range, num_sample_h, device=self.device)
        lat = torch.linspace(lat_range, -lat_range, num_sample_v, device=self.device)
        lon, lat = torch.meshgrid(lon, lat, indexing='xy')

        x, y, z = self.lonlat2xyz(lon, lat)
        xyz = torch.stack([x, y, z], dim=-1)
        return xyz
    
    def _init_perspective_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v=None):
        """
        Initialize perspective projection coordinates.

        Args:
            fov_h (float): Horizontal field of view in degrees.
            fov_v (float): Vertical field of view in degrees.
            num_sample_h (int): Number of horizontal samples.
            num_sample_v (int): Number of vertical samples.

        Returns:
            Tensor of shape (num_sample_v, num_sample_h, 3) representing (x, y, z).
        """

        fov_h = torch.deg2rad(fov_h) if isinstance(fov_h, torch.Tensor) else ang2rad(fov_h)
        fov_v = torch.deg2rad(fov_v) if isinstance(fov_v, torch.Tensor) else ang2rad(fov_v)

        ratio = fov_v / fov_h
        ratio = ratio.cpu().numpy() if isinstance(ratio, torch.Tensor) else ratio

        len_x = torch.tan(fov_h / self.const_2) if isinstance(fov_h, torch.Tensor) else np.tan(fov_h / 2)
        len_y = torch.tan(fov_v / self.const_2) if isinstance(fov_v, torch.Tensor) else np.tan(fov_v / 2)

        if num_sample_v is None: num_sample_v = int(num_sample_h * ratio)
        num_sample_h = int(num_sample_h)

        cx = torch.linspace(-len_x, len_x, num_sample_h, device=self.device)
        cy = torch.linspace(-len_y, len_y, num_sample_v, device=self.device)
        cx, cy = torch.meshgrid(cx, cy, indexing='xy')

        xyz = torch.stack([cx, cy, torch.ones_like(cx, device=self.device)], dim=-1)
        return xyz
    
    def _init_cylindrical_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v=None):
        """
        Initialize cylindrical projection coordinates.

        Args:
            fov_h (float): Horizontal field of view in degrees.
            fov_v (float): Vertical field of view in degrees.
            num_sample_h (int): Number of horizontal samples.
            num_sample_v (int): Number of vertical samples.

        Returns:
            Tensor of shape (num_sample_v, num_sample_h, 3) representing (x, y, z).
        """

        fov_h = torch.deg2rad(fov_h) if isinstance(fov_h, torch.Tensor) else ang2rad(fov_h)
        fov_v = torch.deg2rad(fov_v) if isinstance(fov_v, torch.Tensor) else ang2rad(fov_v)

        ratio = fov_v / fov_h
        ratio = ratio.cpu().numpy() if isinstance(ratio, torch.Tensor) else ratio

        len_x = fov_h / 2
        len_y = torch.tan(fov_v / self.const_2) if isinstance(fov_v, torch.Tensor) else np.tan(fov_v / 2)

        if num_sample_v is None: num_sample_v = int(num_sample_h * ratio)
        num_sample_h = int(num_sample_h)

        lon = torch.linspace(-len_x, len_x, num_sample_h, device=self.device)
        cy = torch.linspace(-len_y, len_y, num_sample_v, device=self.device)
        lon, cy = torch.meshgrid(lon, cy, indexing='xy')

        x = torch.sin(lon)
        z = torch.cos(lon)
        xyz = torch.stack([x, cy, z], dim=-1)  # Combine x, cy, z into a single tensor
        return xyz
    
    def _get_bfov_regin(self, bfov, projection_type=None, num_sample_h=500, num_sample_v=None):
        """
        Get the region of interest (bounding FOV) based on current bounding FOV and projection type.

        Args:
            bfov (Bfov): Bounding FOV object containing orientation and FOV properties.
            projection_type (int, optional): Type of projection (0: perspective, 1: cylindrical, 2: omni).
            num_sample_h (int): Number of horizontal samples.
            num_sample_v (int, optional): Number of vertical samples.

        Returns:
            Tuple of tensors (u, v) representing pixel coordinates in the original image.
        """

        c_lon = torch.deg2rad(bfov.clon) if isinstance(bfov.clon, torch.Tensor) else ang2rad(bfov.clon)
        c_lat = torch.deg2rad(bfov.clat) if isinstance(bfov.clat, torch.Tensor) else ang2rad(bfov.clat)
        rz = torch.deg2rad(bfov.rotation) if isinstance(bfov.rotation, torch.Tensor) else ang2rad(bfov.rotation)

        R = self.rotate_zxy(c_lon, c_lat, rz)
        if projection_type is None:
            if bfov.fov_h>90 or bfov.fov_v > 90:
                projection_type = 2
            else:
                projection_type = 0

        if projection_type == 0:
            xyz = self._init_perspective_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)
        elif projection_type == 1:
            xyz = self._init_cylindrical_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)
        else:
            xyz = self._init_omni_image_cor(bfov.fov_h, bfov.fov_v, num_sample_h, num_sample_v)

        xyz_new = torch.matmul(xyz, R.T).float()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2])
        u = u % (self.img_w + self.const_neg1).int()
        v = v % (self.img_h + self.const_neg1).int()
        return u, v
    
    def align_center_by_lonlat(self, img, lon, lat, rotation=0):
        """
        Align the image center based on longitude, latitude, and rotation.

        Args:
            img (torch.Tensor): Input image tensor.
            lon (torch.Tensor): Longitude angle in radians.
            lat (torch.Tensor): Latitude angle in radians.
            rotation (float or torch.Tensor): Rotation angle in radians.

        Returns:
            Tuple of aligned image tensor and rotation matrix.
        """

        if img.shape[0] != self.img_h_np or img.shape[1] != self.img_w_np:
            raise ValueError(f"Tensor must in shape {self.img_h_np}x{self.img_w_np} (HxW).")
        isRGB = len(img.shape) == 3
        img = self._align_dim(img)

        R = self.rotate_zxy(lon, lat, rotation)
        xyz_new = torch.matmul(self.xyz, R.T).float()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2], True)
        u_norm = (u.float() / (self.img_w + self.const_neg1)) * self.const_2 + self.const_neg1
        v_norm = (v.float() / (self.img_h + self.const_neg1)) * self.const_2 + self.const_neg1

        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)
        out = F.grid_sample(img, grid, mode='nearest', padding_mode='zeros', align_corners=True)

        if isRGB:
            out = out.squeeze(0)
            out = out.permute(1, 2, 0)
        else:
            out = out.squeeze(0).squeeze(0)

        return out, R
    
    def align_center(self, img, u, v, rotation=0):
        """
        Align the image center based on pixel coordinates and rotation.

        Args:
            img (torch.Tensor): Input image tensor.
            u (torch.Tensor): Horizontal pixel coordinates.
            v (torch.Tensor): Vertical pixel coordinates.
            rotation (float or torch.Tensor): Rotation angle in radians.

        Returns:
            Aligned image tensor and rotation matrix.
        """

        lon, lat = self.uv2lonlat(u, v)
        rotation = torch.deg2rad(rotation) if isinstance(rotation, torch.Tensor) else ang2rad(rotation)
        return self.align_center_by_lonlat(img, lon, lat, rotation)
    
    def compute_bbox_from_mask(self, mask, max_only = False):
        """
        Compute bounding boxes from a binary mask.

        Args:
            mask (torch.Tensor or np.ndarray): Binary mask (auto formatting).
            max_only (bool): If True, return only the largest bounding box.

        Returns:
            Tuple containing bounding box coordinates and center (int), or None if invalid.
        """

        mask = self._format_mask(mask)

        if max_only:
            mask_height = mask.shape[0]
            if mask_height > 960:
                resize_ratio = mask_height / 960.
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    scale_factor = 1. / resize_ratio,
                    mode='nearest'
                ).squeeze(0).squeeze(0)

            labeled_mask = connected_components_labeling(mask.to(dtype=torch.uint8))
            
            unique_labels, counts = torch.unique(labeled_mask, return_counts=True)

            non_zero_mask = unique_labels != 0
            unique_labels = unique_labels[non_zero_mask]
            counts = counts[non_zero_mask]

            if unique_labels.numel() == 0:
                return None

            topk_indices = torch.topk(counts, k=min(8, counts.size(0))).indices
            topk_labels = unique_labels[topk_indices]
            
            region_masks = (labeled_mask.unsqueeze(0) == topk_labels.view(-1, 1, 1))

            valid_mask = self._check_min_mask(region_masks, batch=True)  # Shape: (N,)
            if not valid_mask.any():
                return None

            region_masks = region_masks[valid_mask]
            # valid_labels = unique_labels[valid_mask]

            bboxes = masks_to_boxes(region_masks)

            if mask_height > 960:
                bboxes[:, [0, 2]] *= resize_ratio
                bboxes[:, [1, 3]] *= resize_ratio

            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            max_idx = torch.argmax(areas)
            largest_bbox = bboxes[max_idx]

            center_x = (largest_bbox[0] + largest_bbox[2]) / self.const_2
            center_y = (largest_bbox[1] + largest_bbox[3]) / self.const_2

            return largest_bbox.int(), (center_x.int(), center_y.int())

        else:
            if not self._check_min_mask(mask): return None

            bbox = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
            center_x = (bbox[0]+bbox[2])/self.const_2
            center_y = (bbox[1]+bbox[3])/self.const_2
            
            return bbox.int(), (center_x.int(), center_y.int())

    # deprecated    
    def compute_rbbox_from_mask_svd(self, mask):
        """
        Compute rotated bounding box from mask using SVD (deprecated).

        Args:
            mask (torch.Tensor or np.ndarray): Binary mask (auto formatting).

        Returns:
            Rotated bounding box parameters or None if invalid.
        """

        mask = self._format_mask(mask)
        if not self._check_min_mask(mask): return None

        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        if len(x_coords) == 0: return None

        if torch.numel(x_coords)>self.rbbox_samples:
            sample_rate = (torch.numel(x_coords)/self.rbbox_samples).int()
            x_coords = x_coords[::sample_rate]
            y_coords = y_coords[::sample_rate]

        points = torch.stack([y_coords, x_coords], dim=1).float()
        mean = points.mean(dim=0)
        centered_points = points - mean

        _, _, vh = torch.linalg.svd(centered_points)
        principal_axes = vh.T

        projections = centered_points @ principal_axes
        min_proj, _ = projections.min(dim=0)
        max_proj, _ = projections.max(dim=0)

        wh = max_proj - min_proj

        angle = torch.atan2(principal_axes[1, 0], principal_axes[0, 0])
        angle = torch.rad2deg(angle)

        if wh[0] < wh[1]:
            angle += self.const_neg90

        mean = mean.flip(0)

        return mean, wh, angle
    
    def compute_rbbox_from_mask(self, mask):
        """
        Compute rotated bounding box from mask using convex hull and rotating calipers.

        Args:
            mask (torch.Tensor or np.ndarray): Binary mask (auto formatting).

        Returns:
            Tuple containing rotated bbox parameters or None if invalid.
        """

        mask = self._format_mask(mask)
        if not self._check_min_mask(mask): return None

        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        if len(x_coords) < 3: return None

        points = torch.stack([x_coords, y_coords], dim=1).float()

        hull = quickhull(points)

        if hull.shape[0] < 3: return None

        rbbox = rotating_calipers(hull)

        return (torch.tensor([rbbox[0], rbbox[1]]), torch.tensor([rbbox[2], rbbox[3]]), rbbox[4])
    
    def crop_bfov(self, img, bfov, projection_type=None, num_sample_h=1000, num_sample_v=None, return_uv = True):
        """
        Crop the image based on bounding FOV.

        Args:
            img (torch.Tensor): Input image tensor.
            bfov (Bfov): Bounding FOV object.
            projection_type (int, optional): Type of projection.
            num_sample_h (int): Number of horizontal samples.
            num_sample_v (int, optional): Number of vertical samples.
            return_uv (bool): Whether to return pixel coordinates.

        Returns:
            Cropped image tensor and optionally pixel coordinates.
        """

        # input and output shape: rgb (H, W, 3), msk (H, W)
        isRGB = len(img.shape) == 3
        
        img = self._align_dim(img)

        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h, num_sample_v)
        u_norm = (u / (self.img_w + self.const_neg1)) * self.const_2 + self.const_neg1
        v_norm = (v / (self.img_h + self.const_neg1)) * self.const_2 + self.const_neg1

        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)
        out = F.grid_sample(img, grid, mode='nearest', padding_mode='zeros', align_corners=True)
        
        if isRGB:
            out = out.squeeze(0)
            out = out.permute(1, 2, 0)
        else:
            out = out.squeeze(0).squeeze(0)

        if return_uv:
            return out, u, v
        else:
            return out
    
    def uncrop_bfov(self, cropped_img, bfov, projection_type=None, out_width=3840, out_height=1920, ker_size = 3, iter = 1, return_uv = True):
        """
        Reproject the cropped image back to the original frame.

        Args:
            cropped_img (torch.Tensor): Cropped image tensor.
            bfov (Bfov): Bounding FOV object.
            projection_type (int, optional): Type of projection.
            out_width (int): Output image width.
            out_height (int): Output image height.
            ker_size (int): Kernel size for morphological operations.
            iter (int): Number of iterations for morphological operations.
            return_uv (bool): Whether to return pixel coordinates.

        Returns:
            Reprojected image tensor and optionally pixel coordinates.
        """

        cropped_img = self._check_tensor(cropped_img)
        if not cropped_img.dim() == 2:
            raise ValueError(f"Tensor must have 2 dimensions.")
        
        u, v = self._get_bfov_regin(bfov, projection_type, cropped_img.shape[1], cropped_img.shape[0])
        u_flat = u.flatten()
        v_flat = v.flatten()

        x_indices = torch.arange(v.shape[1], device=self.device).repeat(v.shape[0], 1).flatten()
        y_indices = torch.arange(v.shape[0], device=self.device).unsqueeze(1).repeat(1, u.shape[1]).flatten()

        map_x = torch.full((out_height, out_width), -1, dtype=torch.float32, device=self.device)
        map_y = torch.full((out_height, out_width), -1, dtype=torch.float32, device=self.device)

        valid_mask = (u_flat >= 0) & (u_flat < out_width) & (v_flat >= 0) & (v_flat < out_height)
        valid_u = u_flat[valid_mask].long()
        valid_v = v_flat[valid_mask].long()
        valid_x = x_indices[valid_mask].float()
        valid_y = y_indices[valid_mask].float()

        map_x[valid_v, valid_u] = valid_x
        map_y[valid_v, valid_u] = valid_y

        map_x = torch.where(map_x == -1, self.const_0, map_x)
        map_y = torch.where(map_y == -1, self.const_0, map_y)

        map_x_norm = (map_x / (cropped_img.shape[1] - 1.)) * self.const_2 + self.const_neg1
        map_y_norm = (map_y / (cropped_img.shape[0] - 1.)) * self.const_2 + self.const_neg1

        map_x_norm = torch.clamp(map_x_norm, -1.0, 1.0)
        map_y_norm = torch.clamp(map_y_norm, -1.0, 1.0)

        grid = torch.stack((map_x_norm, map_y_norm), dim=-1).unsqueeze(0)
        cropped_img = self._align_dim(cropped_img)

        # Shape: (1, 1, H, W)
        new_img = F.grid_sample(cropped_img, grid, mode='nearest', padding_mode='zeros', align_corners=True)

        new_img = new_img.int()
        out_img = torch.zeros_like(new_img, device=self.device, dtype=torch.int)
        
        kernel = torch.ones((1, 1, ker_size, ker_size), device=self.device, dtype=torch.float32)
        padding = ker_size // 2
        
        unique_labels = new_img.unique().int()
        for label in unique_labels:
            if label == 0: continue

            label_mask = (new_img == label).float()
            for _ in range(iter):
                padded_mask = F.pad(label_mask, (padding, padding, padding, padding), mode='constant', value=0)
                dilated = F.conv2d(padded_mask, kernel, padding=0, groups=1)
                label_mask = (dilated > 0).float()

                padded_dilated = F.pad(1 - label_mask, (padding, padding, padding, padding), mode='replicate')
                eroded = F.conv2d(padded_dilated, kernel, padding=0, groups=1)
                label_mask = 1 - (eroded > 0).float()
            
            out_img[label_mask > 0] = label

        out_img = out_img.squeeze()
        
        if return_uv:
            return out_img, u, v
        else:
            return out_img
    
    def mask2Bfov(self, mask, need_rotation=True):
        """
        Convert a binary mask to a bounding FOV (Bfov) with orientation.

        Args:
            mask (torch.Tensor or np.ndarray): Binary mask (auto formatting).
            need_rotation (bool): Whether to compute rotation based on bbox.

        Returns:
            Bfov object or None if mask is invalid.
        """

        if mask.shape[0] != self.img_h_np or mask.shape[1] != self.img_w_np:
            raise ValueError(f"Tensor must in shape ({self.img_h_np}, {self.img_w_np}) -> (H, W).")
        
        mask = self._format_mask(mask)
        if mask.sum() < self.const_eps: return None

        _check_bbox = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
        _check_bbox_area = (_check_bbox[2]-_check_bbox[0])*(_check_bbox[3]-_check_bbox[1])
        if _check_bbox_area < self.min_mask: return None

        bbox1 = self.compute_bbox_from_mask(mask, max_only=True)
        if bbox1 is None: return None
        _, ct1 = bbox1
        # rough rotation step 2
        mask_image_rotation, R = self.align_center(mask, ct1[0], ct1[1]) 
        # adjust the image according to the centroid of the mask. as mask may cross image
        v, u = torch.nonzero(mask_image_rotation > self.const_0_5, as_tuple=True)
        clon, clat = self.get_inverse_lonlat(R, u.float().mean(), v.float().mean())
        # adjust original image again, Step 2.1
        mask_image_rotation2, R2 = self.align_center_by_lonlat(mask, clon, clat)
        # get the rotated bbox, ensure the target not cross edge
        # _, ct2 = self.compute_bbox_from_mask(mask_image_rotation2)

        # get the final bfov or rbov estimation
        rbbox2 = self.compute_rbbox_from_mask(mask_image_rotation2)
        if rbbox2 is None: return None
        rect_xy, rect_wh, rect_ang = rbbox2
        clon2, clat2 = self.get_inverse_lonlat(R2, rect_xy[0], rect_xy[1])
        if need_rotation:
            rotation_angle = rect_ang if rect_wh[0] > rect_wh[1] else self.const_neg90 + rect_ang
        else: rotation_angle = self.const_0
        
        mask_image_rotation3, R3 = self.align_center_by_lonlat(mask, clon2, clat2, torch.deg2rad(rotation_angle))
        v, u = torch.nonzero(mask_image_rotation3 > self.const_0_5, as_tuple=True)

        min_u, max_u = u.min(), u.max()
        min_v, max_v = v.min(), v.max()

        # based on the bbox, calculate the center and fov of fovbbox
        cu = (min_u + max_u) * self.const_0_5
        cv = (min_v + max_v) * self.const_0_5
        clon3, clat3 = self.get_inverse_lonlat(R3, cu, cv)
        clon = torch.rad2deg(clon3)
        clat = torch.rad2deg(clat3)
        # use a bbox to approximate
        min_lon, min_lat = self.uv2lonlat(min_u, max_v)
        max_lon, max_lat = self.uv2lonlat(max_u, min_v)
        fov_h = torch.rad2deg(max_lon - min_lon)
        fov_v = torch.rad2deg(max_lat - min_lat)

        bfov = Bfov(clon, clat, fov_h, fov_v, rotation_angle)
        return bfov


class OmniFrameworkTorch(OmniImageTorch):
    def __init__(self, ori_width = 3840, ori_height = 1920, 
                 out_width = 1920, out_height = 960, 
                 sr_ratio = 2, sr_min = 60, sr_max = 360, 
                 dense_iter = 1, dense_kernel = 3, max_loss = 4,
                 save_inter = False, device='cuda', check_device=True):
        """
        Initialize OmniFrameworkTorch with tracking and search region parameters.

        Args:
            ori_width (int): Original image width.
            ori_height (int): Original image height.
            out_width (int): Output search region width.
            out_height (int): Output search region height.
            sr_ratio (float): Search region scaling ratio.
            sr_min (float): Minimum search region FOV.
            sr_max (float): Maximum search region FOV.
            dense_iter (int): Iterations for densifing masks.
            dense_kernel (int): Kernel size for densifing masks.
            max_loss (int): Maximum frames to keep search region after target loss.
            save_inter (bool): Whether to save intermediate results for debugging.
            device (str): Compute device ('cuda' or 'cpu').
            check_device (bool): Whether to enforce device consistency.
        """
        
        super().__init__(img_w=ori_width, img_h=ori_height, device=device, check_device=check_device)

        # input size
        self.ori_width = ori_width
        self.ori_height = ori_height
        self.out_width = out_width
        self.out_height = out_height

        # search region settings
        self.sr_ratio_np = sr_ratio
        self.sr_min_np = sr_min
        self.sr_max_np = sr_max if sr_max < 360 else 360
        self.sr_ratio = torch.tensor(sr_ratio, device=self.device, dtype=torch.float32)
        self.sr_min = torch.tensor(sr_min, device=self.device, dtype=torch.float32)
        self.sr_max = torch.tensor(sr_max, device=self.device, dtype=torch.float32)

        # other settings
        self.dense_iter = dense_iter
        self.dense_kernel = dense_kernel

        ## intermediates
        # required
        self.save_inter = save_inter    # if save optional values, for debug
        self.max_loss_sr = max_loss     # max frames to keep search region after loss
        self.valid_sr = False           # if valid search region
        self.full_sr = False            # if search region covers full frame
        self.loss_sr = 0                # counter for consecutive loss frames
        self.pre_bfov_sr:Bfov = None    # previous search region bfov
        self.cur_bfov_sr:Bfov = None    # current search region bfov
        # optional
        self.pre_bfov:Bfov = None       # previous target bfov
        self.cur_bfov:Bfov = None       # current target bfov
        self.cur_rgb_sr = None          # current rgb in search region
        self.cur_msk_sr = None          # current msk in search region (gt or pred msk)
        self.pre_rgb_sr = None          # ...
        self.pre_msk_sr = None          # ...
        self.cur_prd = None             # current predicted msk in original frame
        self.pre_prd = None             # previous predicted msk in original frame
        self.cur_prd_bfov = None        # current bfov of predicted msk in original frame
        self.pre_prd_bfov = None        # previous bfov of predicted msk in original frame
        self.cur_prd_sr = None          # current predicted msk in search region
        self.pre_prd_sr = None          # previous predicted msk in search region

    def _format_framework_rgb(self, rgb: torch.Tensor, ori_rgb: torch.Tensor = None) -> torch.Tensor:
        """
        Format RGB image tensor to a consistent shape for the framework.

        Args:
            rgb (torch.Tensor): Input RGB tensor.
            ori_rgb (torch.Tensor, optional): Original RGB tensor for reference.

        Returns:
            Formatted RGB tensor.
        """

        rgb = self._check_tensor(rgb).float()
        if ori_rgb is None:
            # Formatting to (H, W, 3)
            if rgb.dim() == 4:
                if rgb.size(1) == 3:  # (1, 3, H, W)
                    rgb = rgb.squeeze(0).permute(1, 2, 0)
                elif rgb.size(3) == 3:  # (1, H, W, 3)
                    rgb = rgb.squeeze(0)
                else:
                    raise ValueError("Invalid RGB tensor shape for formatting.")
            elif rgb.dim() == 3:
                if rgb.size(0) == 3:  # (3, H, W)
                    rgb = rgb.permute(1, 2, 0)
                elif rgb.size(2) == 3:  # (H, W, 3)
                    pass
                else:
                    raise ValueError("Invalid RGB tensor shape for formatting.")
            else:
                raise ValueError("Invalid RGB tensor shape for formatting.")
        else:
            if not isinstance(ori_rgb, torch.Tensor): ori_rgb = self._check_tensor(ori_rgb)
            # Formatting to (3, H, W)
            if ori_rgb.dim() == 4:
                if ori_rgb.size(1) == 3:  # (1, 3, H, W)
                    rgb = rgb.unsqueeze(0) # rgb.permute(2, 0, 1).unsqueeze(0)
                elif ori_rgb.size(3) == 3:  # (1, H, W, 3)
                    rgb = rgb.unsqueeze(0).permute(2, 0, 1) # rgb.unsqueeze(0)
            elif ori_rgb.dim() == 3:
                if ori_rgb.size(0) == 3:  # (3, H, W)
                    pass # rgb = rgb.permute(2, 0, 1)
                elif ori_rgb.size(2) == 3:  # (H, W, 3)
                    rgb = rgb.permute(2, 0, 1) # pass
            else:
                raise ValueError("Invalid original RGB tensor shape.")
        
        return rgb

    def _format_framework_msk(self, msk: torch.Tensor, ori_msk: torch.Tensor = None) -> torch.Tensor:
        """
        Format mask tensor to a consistent shape for the framework.

        Args:
            msk (torch.Tensor): Input mask tensor.
            ori_msk (torch.Tensor, optional): Original mask tensor for reference.

        Returns:
            Formatted mask tensor.
        """

        msk = self._check_tensor(msk).float()
        if ori_msk is None:
            # Formatting to (H, W)
            if msk.dim() == 4 and msk.size(0) == 1 and msk.size(1) == 1:  # (1, 1, H, W)
                msk = msk.squeeze(0).squeeze(0)
            elif msk.dim() == 3 and msk.size(0) == 1:  # (1, H, W)
                msk = msk.squeeze(0)
            elif msk.dim() == 2:  # Already (H, W)
                pass
            else:
                raise ValueError("Invalid mask tensor shape for formatting.")
        else:
            if not isinstance(ori_msk, torch.Tensor): ori_msk = self._check_tensor(ori_msk)
            # Recovering to original shape
            if ori_msk.dim() == 4 and ori_msk.size(0) == 1 and ori_msk.size(1) == 1:  # (1, 1, H, W)
                msk = msk.unsqueeze(0).unsqueeze(0)
            elif ori_msk.dim() == 3 and ori_msk.size(0) == 1:  # (1, H, W)
                msk = msk.unsqueeze(0)
            elif ori_msk.dim() == 2:  # (H, W)
                pass  # Already in desired format
            else:
                raise ValueError("Invalid original mask tensor shape.")
        
        return msk

    def clean_search_region(self):
        """
        Reset the search region and related states.
        """

        self.valid_sr = False
        self.full_sr = False
        self.loss_sr = 0
        self.pre_bfov_sr:Bfov = None
        self.cur_bfov_sr:Bfov = None
        if self.save_inter:
            self.pre_bfov:Bfov = None
            self.cur_bfov:Bfov = None
            self.cur_rgb_sr = None
            self.cur_msk_sr = None
            self.pre_rgb_sr = None
            self.pre_msk_sr = None
            self.cur_prd = None
            self.pre_prd = None
            self.cur_prd_bfov = None
            self.pre_prd_bfov = None
            self.cur_prd_sr = None
            self.pre_prd_sr = None
            torch.cuda.empty_cache() 

    def search_region_bfov(self, bfov: Bfov):
        """
        Adjust the bounding FOV for the search region based on current target FOV.

        Args:
            bfov (Bfov): Current target bounding FOV.

        Returns:
            Adjusted search region bounding FOV.
        """

        clon, clat, fov_h, fov_v, rotation = self._check_tensor(*bfov.tolist())

        fov_ratio = fov_h / fov_v

        fov_h_sr = fov_h * self.sr_ratio
        fov_v_sr = fov_v * self.sr_ratio

        v_sr_min = self.sr_min
        h_sr_min = torch.max(self.sr_min * self.const_2, v_sr_min * fov_ratio)
        if fov_h_sr > self.sr_max: 
            fov_h_sr = self.sr_max
        elif fov_h_sr < h_sr_min: 
            fov_h_sr = h_sr_min

        v_sr_max = torch.min(self.sr_max, self.const_180)
        if fov_v_sr > v_sr_max:
            fov_v_sr = v_sr_max
        elif fov_v_sr < v_sr_min: 
            fov_v_sr = v_sr_min

        self.full_sr = fov_h_sr == self.sr_max and fov_v_sr == v_sr_max

        bfov = Bfov(clon, clat, fov_h_sr, fov_v_sr, rotation)
        return bfov
    
    def obtain_search_region(self, raw_rgb, raw_msk):
        """
        Obtain the current search region based on the input RGB and mask.

        Args:
            raw_rgb (torch.Tensor or np.ndarray): Original RGB image.
            raw_msk (torch.Tensor or np.ndarray or None): Ground truth mask or None.

        Returns:
            Tuple of search region RGB and mask tensors.
        """

        rgb = self._format_framework_rgb(raw_rgb) # expect (H, W, 3)
        if not raw_msk is None: 
            # case 1: current rgb has gt msk
            msk = self._format_framework_msk(raw_msk) # expect (H, W)
            cur_bfov = self.mask2Bfov(msk, need_rotation=False)
            if cur_bfov is None: raise ValueError("Invalid ground truth mask.")
            self.valid_sr = True # has gt msk, of course has valid search region
            self.cur_bfov_sr = self.search_region_bfov(cur_bfov)
            rgb_sr = self.crop_bfov(rgb, self.cur_bfov_sr, 
                                    num_sample_h=self.out_width, 
                                    num_sample_v=self.out_height,
                                    return_uv=False)
            msk_sr = self.crop_bfov(msk, self.cur_bfov_sr, 
                                    num_sample_h=self.out_width, 
                                    num_sample_v=self.out_height,
                                    return_uv=False)
            msk_sr = self._format_framework_msk(msk_sr, raw_msk)
        else: # case 2: current rgb don't have gt msk
            if self.valid_sr and not self.full_sr: 
                # case 2.1: valid search region bfov is obtained from the predicted msk of previous frame
                rgb_sr = self.crop_bfov(rgb, self.cur_bfov_sr, 
                                        num_sample_h=self.out_width, 
                                        num_sample_v=self.out_height,
                                        return_uv=False)
            else: 
                # case 2.2: previous frame losing target or predicting msk invalid, resize the whole image as the search region
                rgb_sr = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), (self.out_height, self.out_width), mode='nearest')
                rgb_sr = rgb_sr.squeeze(0).permute(1, 2, 0)
            cur_bfov = None
            msk_sr = None
        rgb_sr = self._format_framework_rgb(rgb_sr, raw_rgb)
        rgb_sr = rgb_sr / self.const_255

        if self.save_inter:
            self.pre_rgb_sr = self.cur_rgb_sr #if self.cur_rgb_sr else self.pre_rgb_sr
            self.pre_msk_sr = self.cur_msk_sr #if self.cur_msk_sr else self.pre_msk_sr
            self.cur_rgb_sr = rgb_sr.int().cpu().numpy()
            self.cur_msk_sr = msk_sr.int().cpu().numpy() if not msk_sr is None else None
            self.pre_bfov = self.cur_bfov
            self.cur_bfov = cur_bfov
        
        torch.cuda.empty_cache()
        return rgb_sr, msk_sr

    def reproject_search_region(self, raw_prd):
        """
        Reproject the predicted mask from search region back to the original frame.

        Args:
            raw_prd (torch.Tensor or np.ndarray): Predicted mask in search region.

        Returns:
            Tuple of reprojected mask as numpy array and validity flag.
        """

        prd_sr = self._format_framework_msk(raw_prd)
        # convert predicted mask on search region back to original image
        if self.valid_sr and not self.full_sr: 
            # case 1 or 2.1: recover the prediction on search region back to original frame
            prd = self.uncrop_bfov(prd_sr, self.cur_bfov_sr,
                                        out_height=self.ori_height, 
                                        out_width=self.ori_width,
                                        ker_size=self.dense_kernel,
                                        iter=self.dense_iter,
                                        return_uv=False)
        else: 
            # case 2.2: prediction is on whole image, so just resize it
            prd = F.interpolate(prd_sr.unsqueeze(0).unsqueeze(0), 
                                (self.ori_height, self.ori_width), mode='nearest')
            prd = prd.squeeze()
        
        prd_bfov = self.mask2Bfov(prd, need_rotation=False)
        self.pre_bfov_sr = self.cur_bfov_sr

        if not prd_bfov is None:
             # case 2.1: prediction on current frame valid
            self.valid_sr = True
            self.loss_sr = 0
            # compute search region for next frame
            self.cur_bfov_sr = self.search_region_bfov(prd_bfov)
        else: 
            # case 2.2: loss the target
            if self.loss_sr < self.max_loss_sr: 
                # case 2.2.1: only loss several frames, e.g. 4, don't change the search region
                self.valid_sr = True
                self.loss_sr += 1
            elif self.loss_sr < self.max_loss_sr * 2:
                # case 2.2.2: loss more frames, e.g. 8, increase the search region
                self.valid_sr = True
                self.cur_bfov_sr = self.search_region_bfov(self.cur_bfov_sr)
                self.loss_sr += 1
                if self.full_sr == True: self.loss_sr = self.max_loss_sr * 2
            else:
                # case 2.2.3: still loss frames, e.g. 9, search on whole frame
                self.valid_sr = False
                self.cur_bfov_sr = None
            prd = torch.zeros((self.ori_height, self.ori_width))

        if self.save_inter:
            self.pre_prd = self.cur_prd
            self.cur_prd = prd.int().cpu().numpy()
            self.pre_prd_bfov = self.cur_prd_bfov
            self.cur_prd_bfov = prd_bfov
            self.pre_prd_sr = self.cur_prd_sr
            self.cur_prd_sr = prd_sr.int().cpu().numpy()

        return (prd.cpu().numpy()).astype(np.uint8), self.valid_sr
