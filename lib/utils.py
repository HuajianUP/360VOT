import numpy as np
import shapely.geometry as sgeo 
import cv2
import torch

class Bfov():
    def __init__(self, lon, lat, fov_h, fov_v, rotation=0):
        # center position : lon, lat
        # horizontal and vertical fov : fov_h, fov_v
        # rotation： positive -> anticlockwise； negative -> clockwise
        # in angle
        self.clon = lon
        self.clat = lat
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.rotation = rotation

    def iou(self, target_bfov):
        pass

    def todict(self):
        return {"clon": self.clon, "clat": self.clat,
                "fov_h": self.fov_h, "fov_v": self.fov_v,
                "rotation": self.rotation}
    def tolist(self):
        return (self.clon, self.clat, self.fov_h, self.fov_v, self.rotation)


class Bbox():
    def __init__(self, cx, cy, w, h, rotation=0):
        # center position : cx, cy
        # bbox width and heigh: w, h
        # rotation： positive-> clockwise； negative -> anticlockwise
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.rotation = rotation

        istensor = isinstance(cx, torch.Tensor) or isinstance(cy, torch.Tensor) or \
            isinstance(w, torch.Tensor) or isinstance(h, torch.Tensor) or \
            isinstance(rotation, torch.Tensor)
        
        if not istensor: self._init_corner()

    def _init_corner(self):
        rotation = ang2rad(self.rotation)
        ## rotation matrix R
        # cos -sin
        # sin cos
        # opencv is based on the bottom-left [-w/2, h/2]
        w_2 = self.w/2
        h_2 = self.h/2
        w_2cos = w_2*np.cos(rotation)
        w_2sin = w_2*np.sin(rotation)
        h_2cos = h_2*np.cos(rotation)
        h_2sin = h_2*np.sin(rotation)

        self.bottomLeft = [self.cx - w_2cos - h_2sin, self.cy - w_2sin + h_2cos] # R@[-w/2, h/2].t()
        self.topLeft = [self.cx - w_2cos + h_2sin, self.cy - h_2cos - w_2sin] # R@[-w/2, -h/2].t() 
        self.topRight = [2*self.cx - self.bottomLeft[0], 2*self.cy - self.bottomLeft[1]]
        self.bottomRight = [2*self.cx - self.topLeft[0], 2*self.cy - self.topLeft[1]]
        

    def iou(self, target_bbox):
        a = sgeo.Polygon([self.topLeft, self.topRight, self.bottomRight, self.bottomLeft])
        b = sgeo.Polygon([target_bbox.topLeft, target_bbox.topRight, target_bbox.bottomRight, target_bbox.bottomLeft])
        iou = a.intersection(b).area / a.union(b).area
        return iou
    
    def todict(self):
        return {"cx": self.cx, "cy": self.cy,
                "w": self.w, "h": self.h,
                "rotation": self.rotation}

    def tolist_xywh(self):
        return (self.topLeft[0], self.topLeft[1], self.w, self.h)

    def tolist(self):
        return (self.cx, self.cy, self.w, self.h, self.rotation)

def convert_mask_to_polygon(mask, max_only=False, integrate=False):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]
    #cv.drawContours(img, contours, -1, (0,255,0), 3)
    #print(len(contours))
    if max_only:
        contours = np.array(max(contours, key=lambda arr: arr.size)).reshape(-1, 2)
        return contours

    if integrate:
        group = []
        for contour in contours:
            if contour.size > 3 * 2:
                group.append(contour)
        contours = np.concatenate(group, axis = 0).reshape(-1, 2)
        return contours

    return contours

def scaleBFoV(BFoV, scale):
    fov_h = max(0, min(BFoV.fov_h * scale, 360))
    fov_v = max(0, min(BFoV.fov_v * scale, 180))
    return Bfov(BFoV.clon, BFoV.clat, fov_h, fov_v)

def dict2Bfov(bfov_dict):
    return Bfov(bfov_dict["clon"], bfov_dict["clat"], bfov_dict["fov_h"], bfov_dict["fov_v"], bfov_dict["rotation"])

def dict2Bbox(bbox_dict):
    return Bbox(bbox_dict["cx"], bbox_dict["cy"], bbox_dict["w"], bbox_dict["h"], bbox_dict["rotation"])

def x1y1wh2bbox(bbox_list):
    rotation = bbox_list[4] if len(bbox_list) == 5 else 0
    cx = bbox_list[0] + bbox_list[2] * 0.5
    cy = bbox_list[1] + bbox_list[3] * 0.5
    return Bbox(cx, cy, bbox_list[2], bbox_list[3], rotation)

def ang2rad(a):
    return a/180*np.pi

def rad2ang(r):
    return r/np.pi*180

def rotation_2d(angle):
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

def rotate_yx(angle_x, angle_y):
    r = np.identity(3)
    sin_x = np.sin(angle_x)
    cos_x = np.cos(angle_x)
    sin_y = np.sin(angle_y)
    cos_y = np.cos(angle_y)

    r[0, 0] = cos_y
    r[0, 1] = sin_y * sin_x
    r[0, 2] = sin_y * cos_x
    r[1, 1] = cos_x
    r[1, 2] = -sin_x
    r[2, 0] = -sin_y
    r[2, 1] = cos_y * sin_x
    r[2, 2] = cos_y * cos_x
    return r

def rotate_x(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[1, 1] = cos_a
    r_mat[2, 2] = cos_a
    r_mat[1, 2] = -sin_a
    r_mat[2, 1] = sin_a
    return r_mat

def rotate_y(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[0, 0] = cos_a
    r_mat[2, 2] = cos_a
    r_mat[0, 2] = sin_a
    r_mat[2, 0] = -sin_a
    return r_mat

def rotate_z(angle):
    r_mat = np.identity(3)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    r_mat[0, 0] = cos_a
    r_mat[1, 1] = cos_a
    r_mat[0, 1] = -sin_a
    r_mat[1, 0] = sin_a
    return r_mat

def mask_dilate(mask, kernel_size = 3):
    mask = mask.astype('uint8')
    objs = np.unique(mask)
    res_mask = np.zeros_like(mask)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for obj in objs[1:]:  
        binary_mask = np.where(mask == obj, 1, 0).astype(np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        res_mask[dilated_mask > 0] = obj
        
    return res_mask

def rotating_calipers(hull: torch.Tensor):
    """
    Finds the minimum area rectangle enclosing the convex hull using the Rotating Calipers method.

    Args:
        hull (torch.Tensor): Convex hull points in counter-clockwise order, shape (M, 2).

    Returns:
        torch.Tensor: Tensor containing [center_x, center_y, width, height, angle_deg].
                      center_x and center_y are the coordinates of the rectangle center,
                      width and height are the dimensions of the rectangle,
                      angle_deg is the angle of rotation in degrees.
    """
    M = hull.shape[0]
    if M < 3:
        raise ValueError("Convex hull must have at least 3 points.")

    # Compute edges and their lengths
    edges = hull[(torch.arange(M) + 1) % M] - hull  # Shape: (M, 2)
    edge_lengths = torch.norm(edges, dim=1)  # Shape: (M,)

    # Filter out zero-length edges
    valid_edges = edge_lengths > 1e-6
    edges = edges[valid_edges]
    hull = hull[valid_edges]
    M = hull.shape[0]

    # Compute angles of edges relative to the x-axis
    angles = torch.atan2(edges[:, 1], edges[:, 0])  # Shape: (M,)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Rotation matrices for each edge
    R = torch.stack([torch.stack([cos_angles, -sin_angles], dim=1),
                     torch.stack([sin_angles, cos_angles], dim=1)], dim=1)  # Shape: (M, 2, 2)

    # Rotate all points for each edge
    rotated_hulls = torch.stack([
        torch.matmul(hull - hull[i], R[i]) for i in range(M)
    ])  # Shape: (M, M, 2)

    # Compute bounding boxes in rotated space
    mins = rotated_hulls.min(dim=1).values  # Shape: (M, 2)
    maxs = rotated_hulls.max(dim=1).values  # Shape: (M, 2)
    widths = maxs[:, 0] - mins[:, 0]  # Shape: (M,)
    heights = maxs[:, 1] - mins[:, 1]  # Shape: (M,)
    areas = widths * heights  # Shape: (M,)

    # Find the rotation with the minimum area
    min_idx = torch.argmin(areas)
    min_width = widths[min_idx]
    min_height = heights[min_idx]
    min_angle = angles[min_idx]

    # Compute the center of the rectangle in rotated space
    center_rotated = (mins[min_idx] + maxs[min_idx]) / 2  # Shape: (2,)
    # Transform the center back to the original space
    center = torch.matmul(center_rotated, R[min_idx].T) + hull[min_idx]  # Shape: (2,)

    # Adjust angle to be in the range [-90, 90)
    min_angle_deg = torch.rad2deg(min_angle).item()
    if min_angle_deg < 0:
        min_angle_deg += 180
    if min_angle_deg >= 90:
        min_angle_deg -= 90
        min_width, min_height = min_height, min_width

    return torch.tensor([*center, min_width, min_height, min_angle_deg], device=hull.device, dtype=hull.dtype)

def quickhull(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the convex hull of a set of 2D points using the QuickHull algorithm.
    Supports both CPU and GPU tensors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), where each row is a 2D point.

    Returns:
        torch.Tensor: Convex hull points in counter-clockwise order, shape (M, 2).
    """
    # Ensure input is a 2D tensor with shape (N, 2)
    assert points.ndim == 2 and points.shape[1] == 2, "Input must be a tensor of shape (N, 2)"
    N = points.shape[0]

    if N < 3:
        raise ValueError("The input points are not enough for searching convex hull, expect 3 or more.")

    # Helper function: Cross product to determine orientation
    def cross_product(o, a, b):
        """
        Computes the cross product of vectors OA and OB.
        Positive -> counter-clockwise, Negative -> clockwise, Zero -> collinear.
        """
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    # Helper function: Find the farthest point from the line formed by points p1 and p2
    def farthest_point_index(p1, p2, subset):
        """
        Finds the index of the point in `subset` that is farthest from the line (p1, p2).
        """
        distances = torch.abs(cross_product(p1, p2, subset))  # Perpendicular distances
        return torch.argmax(distances)

    # Helper function: Recursive QuickHull
    def quickhull_recursive(subset, p1, p2):
        """
        Recursive QuickHull subroutine to find points on one side of the line (p1, p2).
        """
        if subset.shape[0] == 0:  # No points left
            return []

        # Find the farthest point from line (p1, p2)
        farthest_idx = farthest_point_index(p1, p2, subset)
        farthest = subset[farthest_idx]

        # Partition points into two subsets: left of (p1, farthest) and (farthest, p2)
        left_of_p1_far = subset[cross_product(p1, farthest, subset) > 0]
        left_of_far_p2 = subset[cross_product(farthest, p2, subset) > 0]

        # Recursively find hull points
        return (quickhull_recursive(left_of_p1_far, p1, farthest) +
                [farthest] +
                quickhull_recursive(left_of_far_p2, farthest, p2))

    # Step 1: Find the two extreme points (leftmost and rightmost)
    leftmost_idx = torch.argmin(points[:, 0])
    rightmost_idx = torch.argmax(points[:, 0])
    leftmost = points[leftmost_idx]
    rightmost = points[rightmost_idx]

    # Partition points into two subsets: above and below the line (leftmost, rightmost)
    above = points[cross_product(leftmost, rightmost, points) > 0]
    below = points[cross_product(rightmost, leftmost, points) > 0]

    # Step 2: Recurse to find points on the upper and lower hulls
    upper_hull = quickhull_recursive(above, leftmost, rightmost)
    lower_hull = quickhull_recursive(below, rightmost, leftmost)

    # Combine hull points and include the extreme points
    hull_points = [leftmost] + upper_hull + [rightmost] + lower_hull

    # Convert hull points to a tensor and return
    return torch.stack(hull_points)

# deprecated, too slow
def convexhull(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the convex hull of a set of 2D points using Andrew's Monotone Chain algorithm.
    Supports both CPU and GPU tensors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) containing 2D points.

    Returns:
        torch.Tensor: Convex hull points in counter-clockwise order, shape (M, 2).
    """
    # Ensure input is a 2D tensor with shape (N, 2)
    assert points.ndim == 2 and points.shape[1] == 2, "Input must be a tensor of shape (N, 2)"
    N = points.shape[0]

    if N < 3:
        raise ValueError("The input points are not enough for searching convex hull, expect 3 or more.")

    # Sort points lexicographically by x-coordinate, then by y-coordinate
    # Sorting works on both CPU and GPU
    sorted_points, _ = torch.sort(points, dim=0)  # Sort by x-coordinates
    sorted_indices = torch.argsort(sorted_points[:, 1], stable=True)  # Then by y-coordinates
    points = points[sorted_indices]

    # Helper function to compute cross product
    def cross_product(o, a, b):
        """
        Cross product of vectors OA and OB:
        > 0 if counter-clockwise, < 0 if clockwise, 0 if collinear.
        """
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    # Build the lower hull
    lower = []
    for i in range(N):
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], points[i]) <= 0:
            lower.pop()
        lower.append(points[i])

    # Build the upper hull
    upper = []
    for i in range(N - 1, -1, -1):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], points[i]) <= 0:
            upper.pop()
        upper.append(points[i])

    # Concatenate lower and upper hulls, removing duplicates
    hull = torch.stack(lower[:-1] + upper[:-1])

    return hull