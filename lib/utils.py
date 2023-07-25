import numpy as np
import shapely.geometry as sgeo 
import cv2

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
        self._init_corner()

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