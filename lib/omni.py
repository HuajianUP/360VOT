import cv2
import numpy as np
import shapely.geometry as sgeo 

from utils import *

class OmniCam():
    """
    Spherical camera model
    u[0:image_w] -> lon[-pi:pi]
    v[0:image_h] -> lat[pi/2:-pi/2]
    camera coordinate (opencv convention): x_axis: right; y_axis: down; z_axis: outwards
    """
    def __init__(self, img_w=1920, img_h=960):
        self.img_w = img_w
        self.img_h = img_h
        self.fx = img_w / (2 * np.pi)
        self.fy = -img_h / np.pi
        self.cx = img_w / 2
        self.cy = img_h / 2
    
    def uv2lonlat(self, u, v):
        lon = ((u + 0.5) - self.cx) / self.fx
        lat = ((v + 0.5) - self.cy) / self.fy
        return lon, lat

    def lonlat2xyz(self, lon, lat):
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(-lat)
        z = np.cos(lat) * np.cos(lon)
        return x, y, z

    def uv2xyz(self, u, v):
        lon, lat = self.uv2lonlat(u, v)
        return self.lonlat2xyz(lon, lat)

    def xyz2lonlat(self, x, y, z, norm=False):
        lon = np.arctan2(x, z)
        lat = np.arcsin(-y) if norm else np.arctan2(-y, np.sqrt(x**2 + z**2)) 
        return lon, lat

    def lonlat2uv(self, lon, lat):
        u = lon * self.fx + self.cx - 0.5
        v = lat * self.fy + self.cy - 0.5
        return u, v

    def xyz2uv(self, x, y, z, norm=False):
        lon, lat = self.xyz2lonlat(x, y, z, norm)
        return self.lonlat2uv(lon, lat)

    def get_inverse_lonlat(self, R, u, v):
        x, y, z = self.uv2xyz(u, v)
        xyz = R @ np.array([x, y, z])
        lon, lat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])
        return lon, lat
    
    def get_rough_FOV(self, bbox_w, bbox_h):
        return bbox_w/self.img_w * 360, bbox_h/self.img_h * 180
    

class OmniImage(OmniCam):
    def __init__(self, img_w=1920, img_h=960):
        super().__init__(img_w, img_h)

        self.xyz = self._init_omni_image_cor()

    def _init_omni_image_cor(self, fov_h=360, fov_v=180, num_sample_h=None, num_sample_v=None):
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        if num_sample_h is None:
            num_sample_h = self.img_w
        if num_sample_v is None:   
            num_sample_v = int(num_sample_h * (fov_v / fov_h))

        lon_range = fov_h / 2
        lat_range = fov_v / 2

        lon, lat = np.meshgrid(np.linspace(-lon_range, lon_range, num_sample_h), np.linspace(lat_range, -lat_range, num_sample_v))
        x, y, z = self.lonlat2xyz(lon, lat)
   
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1) 
        return xyz

    def _init_perspective_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v):
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        # initi tangent image
        len_x = np.tan(fov_h/2)
        len_y = np.tan(fov_v/2)
        if num_sample_v is None:
            num_sample_v = int(num_sample_h * (fov_v / fov_h))

        cx, cy = np.meshgrid(np.linspace(-len_x, len_x, num_sample_h), np.linspace(-len_y, len_y, num_sample_v))
        xyz = np.concatenate([cx[..., None], cy[..., None], np.ones_like(cx)[..., None]], axis=-1)
        return xyz

    def _init_cylindrical_image_cor(self, fov_h, fov_v, num_sample_h, num_sample_v):
        fov_h = ang2rad(fov_h)
        fov_v = ang2rad(fov_v)

        # initi tangent image
        len_x = fov_h/2
        len_y = np.tan(fov_v/2)
        if num_sample_v is None:
            num_sample_v = int(num_sample_h * fov_v / fov_h)

        lon, cy = np.meshgrid(np.linspace(-len_x, len_x, num_sample_h), np.linspace(-len_y, len_y, num_sample_v))
        x = np.sin(lon)
        z = np.cos(lon)

        xyz = np.concatenate([x[..., None], cy[..., None], z[..., None]], axis=-1)

        return xyz

    def _get_bfov_regin(self, bfov, projection_type=None, num_sample_h=500, num_sample_v=None):
        c_lon = ang2rad(bfov.clon)
        c_lat = ang2rad(bfov.clat)
        rz = ang2rad(bfov.rotation)
        R = rotate_y(c_lon)  @ rotate_x(c_lat) @ rotate_z(rz)
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

        xyz_new =  xyz @ R.transpose()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2])
        u = u.astype(np.float32)%(self.img_w-1)
        v = v.astype(np.float32)%(self.img_h-1)
        return u, v
    
    def crop_bfov(self, img, bfov, projection_type=None, num_sample_h=1000, num_sample_v=None):
        # supposed the input is in angle 
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h, num_sample_v)
        out = cv2.remap(img, u, v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) #INTER_LINEAR
        #print(out.shape)
        return out, u, v
    
    def plot_bfov(self, img, bfov, projection_type=None, num_sample_h=1000, num_sample_v=None, border_only=True, color=(255, 0, 0), size = 10):
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h, num_sample_v) # img_h, 
        img = self.plot_uv(img, u, v, border_only, color, size)
        return img

    def plot_uv(self, img, u, v, border_only=True, color=(255, 0, 0), size = 2):
        if border_only:
            for j in range(u.shape[1]):
                cv2.circle(img, (int(u[0, j]), int(v[0, j])), 1, color, size)
                cv2.circle(img, (int(u[-1, j]), int(v[-1, j])), 1, color, size)
            
            for i in range(u.shape[0]):
                cv2.circle(img, (int(u[i, 0]), int(v[i, 0])), 1, color, size)
                cv2.circle(img, (int(u[i, -1]), int(v[i, -1])), 1, color, size)
        else:
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    cv2.circle(img, (int(u[i, j]), int(v[i, j])), 1, color, 1)
        return img    

    def crop_bbox(self, img, bbox, borderMode=cv2.BORDER_CONSTANT, needBoderValue=False):
        # only consider horizontal rotation, and pad the vertical
        u, v = np.meshgrid(np.linspace(-bbox.w*0.5, bbox.w*0.5, int(bbox.w)), np.linspace(-bbox.h*0.5, bbox.h*0.5, int(bbox.h)))
        R = rotation_2d(ang2rad(bbox.rotation))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1) @ R.transpose()
        u = uv[..., 0] + bbox.cx
        v = uv[..., 1] + bbox.cy

        u = u.astype(np.float32)%(self.img_w-1)
        v = v.astype(np.float32)#%self.img_h
    
        borderValue = cv2.mean(img) if needBoderValue else [0,0,0]
        out = cv2.remap(img, u, v, interpolation=cv2.INTER_NEAREST, borderMode=borderMode, borderValue=borderValue)
        return out, u, v

    def plot_bbox(self, img, bbox, color=(255, 0, 0), size = 10):
        # according to the characteristic of 360, it is impossible to cross top and bottom, thus only consider cross left and right
        #print("plot_bbox", bbox.todict(), bbox.topLeft, bbox.topRight, bbox.topRight, bbox.bottomLeft )
        img_h, img_w = img.shape[:2]
        topleft = bbox.topLeft.copy()
        topRight = bbox.topRight.copy()
        bottomRight = bbox.bottomRight.copy()
        bottomLeft = bbox.bottomLeft.copy()

        topLine = sgeo.LineString([topleft, topRight])
        rightLine = sgeo.LineString([topRight, bottomRight])
        bottomLine = sgeo.LineString([bottomRight, bottomLeft])
        leftLine = sgeo.LineString([bottomLeft, topleft])

        leftBorder = sgeo.LineString([(-1, 0), (-1, img_h)])
        rightBorder = sgeo.LineString([(img_w, 0), (img_w, img_h)])

        lines = []

        if topLine.intersects(leftBorder):
            point = topLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                lines.append([topleft, [img_w-1, point.y]])
                lines.append([[0, point.y], topRight])
        elif topLine.intersects(rightBorder):
            point = topLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                lines.append([topleft, [img_w-1, point.y]])
                lines.append([[0, point.y], topRight])
        else:
            lines.append([topleft, topRight])

        
        if bottomLine.intersects(leftBorder):
            point = bottomLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                lines.append([bottomLeft, [img_w-1, point.y]])
                lines.append([[0, point.y], bottomRight])
        elif bottomLine.intersects(rightBorder):
            point = bottomLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                lines.append([bottomLeft, [img_w-1, point.y]])
                lines.append([[0, point.y], bottomRight])
        else:
            lines.append([bottomLeft, bottomRight])


        if rightLine.intersects(leftBorder):
            point = rightLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                if topRight[0] < 0:
                    lines.append([topRight, [img_w-1, point.y]])
                    lines.append([[0, point.y], bottomRight])
                else:
                    lines.append([topRight, [0, point.y]])
                    lines.append([[img_w-1, point.y], bottomRight])

        elif rightLine.intersects(rightBorder):
            point = rightLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                if topRight[0] < img_w:
                    lines.append([topRight, [img_w-1, point.y]])
                    lines.append([[0, point.y], bottomRight])
                else:
                    lines.append([topRight, [0, point.y]])
                    lines.append([[img_w-1, point.y], bottomRight])
        else:
            lines.append([topRight, bottomRight])


        if leftLine.intersects(leftBorder):
            point = leftLine.intersection(leftBorder)
            if isinstance(point, sgeo.Point):
                if topleft[0] < 0:
                    lines.append([topleft, [img_w-1, point.y]])
                    lines.append([[0, point.y], bottomLeft])
                else:
                    lines.append([topleft, [0, point.y]])
                    lines.append([[img_w-1, point.y], bottomLeft])

        elif leftLine.intersects(rightBorder):
            point = leftLine.intersection(rightBorder)
            if isinstance(point, sgeo.Point):
                if topleft[0] < img_w:
                    lines.append([topleft, [img_w-1, point.y]])
                    lines.append([[0, point.y], bottomLeft])
                else:
                    lines.append([topleft, [0, point.y]])
                    lines.append([[img_w-1, point.y], bottomLeft])
        else:
            lines.append([topleft, bottomLeft])

        for line in lines:
            start, end = line.copy()
            start = np.intp(start)
            end = np.intp(end)
            start[0] %= img_w
            end[0] %= img_w
            cv2.line(img, start, end, color, size)
        return img

    def _get_global_coordinate(self, bbox, ref_u, ref_v):
        # convert local box coordinate to coordinate

        u_init, v_init = np.meshgrid(np.linspace(-bbox.w*0.5, bbox.w*0.5, int(bbox.w)), np.linspace(-bbox.h*0.5, bbox.h*0.5, int(bbox.h)))
        R = rotation_2d(ang2rad(bbox.rotation))
        uv_init = np.concatenate([u_init[..., None], v_init[..., None]], axis=-1) @ R.transpose()
        u_local = uv_init[..., 0] + bbox.cx
        v_local = uv_init[..., 1] + bbox.cy
        u_local = u_local.astype(np.float32)
        v_local = v_local.astype(np.float32)
        u_global = cv2.remap(ref_u, u_local, v_local, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        v_global = cv2.remap(ref_v, u_local, v_local, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)

        c_u = ref_u[int(bbox.cy), int(bbox.cx)]
        c_v = ref_v[int(bbox.cy), int(bbox.cx)]
        return c_u, c_v, u_global, v_global

    def uv2Bbox(self, u, v, cx, need_rotation):

        shift = cx - self.img_w * 0.5

        u_new = (u - shift)%(self.img_w-1)
        v_new = v
        uv = np.concatenate([u_new[..., None], v_new[..., None]], axis=-1).reshape(-1, 2)
        rotation_angle = 0
        if need_rotation:
            rect_xy, rect_wh, rect_ang = rect = cv2.minAreaRect(uv) 
            cx, cy = rect_xy
            w, h = rect_wh
            rotation_angle = rect_ang
            #print(cx, cy, w, h, rect_ang)
        else:
            lx, ly, w, h = cv2.boundingRect(uv)
            #print(lx, ly, w, h)
            cx = lx + w * .5
            cy = ly + h * .5

        """
        x1, y1 = np.min(u_new), np.min(v_new)
        x2, y2 = np.max(u_new), np.max(v_new)
        w = (x2 - x1)#%self.img_w 
        h = (y2 - y1)#%self.img_h
        #w = w if w < self.img_w else self.img_
        assert w > 0 and h > 0
        cx = (x2+x1) * 0.5 - shift if w < self.img_w-1 else self.img_w * 0.5
        cy = (y2+y1) * 0.5
        """
        cx = cx + shift if w < self.img_w-1 else self.img_w * 0.5
        
        return Bbox(cx, cy, w, h, rotation_angle)

    def localBbox2Bfov(self, bbox, ref_u, ref_v, need_rotation=True):
        # Args: 
        # bbox: supposed to be axis-aligned
        # ref_u, ref_v: coordinate of the local region with respect to origin 360 image
        # Return: Bfov 
        # consider the case of rotated local bbox
        c_u, c_v, u_global, v_global = self._get_global_coordinate(bbox, ref_u, ref_v)

        c_lon, c_lat = self.uv2lonlat(c_u, c_v)
        R = rotate_y(c_lon)  @ rotate_x(c_lat) 

        u = np.concatenate([u_global[0, :], u_global[:, -1], u_global[-1, :], u_global[:, 0]])
        v = np.concatenate([v_global[0, :], v_global[:, -1], v_global[-1, :], v_global[:, 0]])

        #print(uv)
        x, y, z = self.uv2xyz(u, v)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ R
        lon, lat = self.xyz2lonlat(xyz[:, 0], xyz[:, 1], xyz[:, 2], True)
 
        rotation_angle = 0
        if need_rotation:
            shift = 1000
            scale = 360
            lon  = np.intp(lon * scale + shift)
            lat  = np.intp(lat * scale + shift)
            #print(lon)
            lonlat = np.concatenate([lon[..., None], lat[..., None]], axis=-1).reshape(-1, 2)
            rect_xy, rect_wh, rect_ang = cv2.minAreaRect(lonlat) 
            c_lon, c_lat = rect_xy
            c_lon = (c_lon - shift) / scale
            c_lat = (c_lat - shift) / scale
            fov_h, fov_v = rect_wh
            fov_h /= scale
            fov_v /= scale
            rotation_angle = -rect_ang

            if abs(rect_ang + bbox.rotation) > 85:
                temp = fov_v
                fov_v = fov_h
                fov_h = temp
                rotation_angle += 90
                #print("swap")
            #print(c_lon, c_lat, fov_h, fov_v, rect_ang)
        else:
            lon_max, lon_min = np.max(lon), np.min(lon)
            lat_max, lat_min = np.max(lat), np.min(lat)
            fov_h = lon_max - lon_min
            fov_v = lat_max - lat_min
            c_lon = (lon_max + lon_min)*0.5
            c_lat = (lat_max + lat_min)*0.5

        x, y, z = self.lonlat2xyz(c_lon, c_lat)
        xyz = R @ np.array([x, y, z])
        c_lon, c_lat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])
        bfov = Bfov(rad2ang(c_lon), rad2ang(c_lat), rad2ang(fov_h), rad2ang(fov_v), rotation_angle)

        return bfov #, u_global, v_global

    def localBbox2Bbox(self, bbox, ref_u, ref_v, need_rotation=True):
        # get the global bbox to cover the area of local bbox.
        c_u, c_v, u_global, v_global = self._get_global_coordinate(bbox, ref_u, ref_v)
        return self.uv2Bbox(u_global, v_global, c_u, need_rotation)

    def bbox2Bfov(self, bbox, need_rotation=False):
        # Args: the bbox with respect to the 360 image
        # Return: Bfov
        # get the boundary 
        topLine = sgeo.LineString([bbox.topLeft, bbox.topRight])
        rightLine = sgeo.LineString([bbox.topRight, bbox.bottomRight])
        bottomLine = sgeo.LineString([bbox.bottomRight, bbox.bottomLeft])
        leftLine = sgeo.LineString([bbox.bottomLeft, bbox.topLeft])
        
        uv1 = np.array([topLine.interpolate(dis).xy for dis in np.linspace(0, topLine.length, bbox.w)])
        uv2 = np.array([rightLine.interpolate(dis).xy for dis in np.linspace(0, rightLine.length, bbox.h)])
        uv3 = np.array([bottomLine.interpolate(dis).xy for dis in np.linspace(0, bottomLine.length, bbox.w)])
        uv4 = np.array([leftLine.interpolate(dis).xy for dis in np.linspace(0, leftLine.length, bbox.h)])
        #print(uv1.shape, uv2.shape)
        uv = np.concatenate([uv1, uv2, uv3, uv4]).reshape(-1, 2)
        #print(uv.shape)
        u = uv[:, 0]%(self.img_w-1)
        v = uv[:, 1]%(self.img_h-1)
        
        clon, clat = self.uv2lonlat(bbox.cx, bbox.cy)
        rotation_angle = bbox.rotation if need_rotation else 0

        R = rotate_y(clon) @ rotate_x(clat) @ rotate_z(ang2rad(rotation_angle))

        x, y, z = self.uv2xyz(u, v)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ R
        lon, lat = self.xyz2lonlat(xyz[..., 0], xyz[..., 1], xyz[..., 2], True)

        clon = (np.max(lon) + np.min(lon))*0.5
        clat = (np.max(lat) + np.min(lat))*0.5
        x, y, z = self.lonlat2xyz(clon, clat)
        xyz = R @ np.array([x, y, z])
        clon, clat = self.xyz2lonlat(xyz[0], xyz[1], xyz[2])

        clon = rad2ang((clon))
        clat = rad2ang((clat))
        
        fov_h = rad2ang(np.max(lon) - np.min(lon))
        fov_v = rad2ang(np.max(lat) - np.min(lat))

        return Bfov(clon, clat, fov_h, fov_v, rotation_angle), u, v

    def bfov2Bbox(self, bfov, need_rotation=False, projection_type=None):
        # Args: Bfov 
        # Return: the bbox with respect to the 360 image
        u, v = self._get_bfov_regin(bfov, projection_type, num_sample_h=500) # img_h,
        cx, _ = self.lonlat2uv(ang2rad(bfov.clon), 0)
        return self.uv2Bbox(u, v, cx, need_rotation)

    def align_center_by_lonlat(self, img, lon, lat, rotation=0):
        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            self.img_w = img.shape[1] 
            self.img_h = img.shape[0] 
            self._init_image_cor()
        R = rotate_y(lon) @ rotate_x(lat) @ rotate_z(rotation)

        xyz_new =  self.xyz @ R.transpose()

        u, v = self.xyz2uv(xyz_new[..., 0], xyz_new[..., 1], xyz_new[..., 2], True)
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        out_img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        return out_img, R

    def align_center_by_lonlatangle(self, img, lon, lat, rotation=0):
        lat = ang2rad(lat)
        lon = ang2rad(lon)
        rotation = ang2rad(rotation)
        return self.align_center_by_lonlat(img, lon, lat, rotation)

    def align_center(self, img, u, v, rotation=0):
        lon, lat = self.uv2lonlat(u, v)
        rotation = ang2rad(rotation)
        return self.align_center_by_lonlat(img, lon, lat, rotation)
    
    def rot_image(self, img, pitch, yaw, roll=0):
        """
        rot the image by the angle
        positive pitch pulls up the (original center of) image
        positive yaw shifts the (original center of) image to the right
        positive roll clockwise rotates the image along the center
        """
        lat = -ang2rad(pitch)
        lon = -ang2rad(yaw)
        rotation = -ang2rad(roll)
        return self.align_center_by_lonlat(img, lon, lat, rotation)

    def mask2Bfov(self, mask_image, need_rotation=True):
        assert self.img_w == mask_image.shape[1] and self.img_h == mask_image.shape[0]
         
        if len(mask_image.shape)>2:   
            mask = mask_image[:, :, 0].copy()
        else:
            mask = mask_image.copy()
        test_v, test_u= np.where(mask > 127)
        if len(test_v)<8:
            return None
        # need to consider disapper case
        contours1 = convert_mask_to_polygon(mask, max_only=True) 
        cx, cy = np.mean(contours1, axis=0)
        # rough rotation step 2
        mask_image_rotation, R = self.align_center(mask, cx, cy) 
        # adjust the image according to the centroid of the mask. as mask may cross image
        v, u = np.where(mask_image_rotation>127)
        clon, clat = self.get_inverse_lonlat(R, np.mean(u), np.mean(v))
        # adjust original image again, Step 2.1
        mask_image_rotation2, R2 = self.align_center_by_lonlat(mask, clon, clat)
        # get the rotated bbox, ensure the target not cross edge
        contours2 = convert_mask_to_polygon(mask_image_rotation2, integrate=True)

        # get the final bfov or rbov estimation
        rect_xy, rect_wh, rect_ang = cv2.minAreaRect(contours2)
        clon2, clat2 = self.get_inverse_lonlat(R2, rect_xy[0], rect_xy[1])
        rotation_angle = 0
        if need_rotation:
            rotation_angle = rect_ang if rect_wh[0] > rect_wh[1] else -90 + rect_ang
        
        mask_image_rotation3, R3 = self.align_center_by_lonlat(mask, clon2, clat2, ang2rad(rotation_angle))
        contours3 = convert_mask_to_polygon(mask_image_rotation3, integrate=True)
        u = contours3[:, 0]
        v = contours3[:, 1]

        min_u, max_u = np.min(u), np.max(u)
        min_v, max_v = np.min(v), np.max(v)

        # based on the bbox, calculate the center and fov of fovbbox
        cu = (min_u+max_u)*0.5
        cv = (min_v+max_v)*0.5
        clon3, clat3 = self.get_inverse_lonlat(R3, cu, cv)
        clon = rad2ang(clon3)
        clat = rad2ang(clat3)
        # use a bbox to approximate
        min_lon, min_lat = self.uv2lonlat(min_u, max_v)
        max_lon, max_lat = self.uv2lonlat(max_u, min_v)
        fov_h = rad2ang(max_lon - min_lon)
        fov_v = rad2ang(max_lat - min_lat)

        bfov = Bfov(clon, clat, fov_h, fov_v, rotation_angle)
        return bfov
    
    def mask2Bbox(self, mask_image, need_rotation=True):
        assert self.img_w == mask_image.shape[1] and self.img_h == mask_image.shape[0]
            
        if len(mask_image.shape)>2:   
            mask = mask_image[:, :, 0].copy()
        else:
            mask = mask_image.copy()

        test_v, test_u= np.where(mask > 127)
        if len(test_v)<8:
            return None

        # Step 1
        contours1 = convert_mask_to_polygon(mask, max_only=True)
        cx, cy = np.mean(contours1, axis=0)
        c_lon, c_lat = self.uv2lonlat(cx, cy)
        # horizontal shift, Step 2
        mask_image_rotation, R = self.align_center_by_lonlat(mask, c_lon, 0)
        shift = cx -  self.img_w * 0.5
        # get the final bbox or rbbox
        contours2 = convert_mask_to_polygon(mask_image_rotation, integrate=True)
        rotation_angle = 0
        if need_rotation:
            rect_xy, rect_wh, rotation_angle = cv2.minAreaRect(contours2)
            cx, cy = rect_xy
            w, h = rect_wh
        else:
            lx, ly, w, h = cv2.boundingRect(contours2)
            cx  = lx + w * 0.5
            cy  = ly + h * 0.5

        cx = (cx + shift)%self.img_w if w < self.img_w-1 else self.img_w * 0.5

        return Bbox(cx, cy, w, h, rotation_angle)
    
