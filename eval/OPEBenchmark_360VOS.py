from sphgrid import SphGrid

import cv2
import numpy as np

class VOSMetrics:
    def __init__(self, img_h=960, img_w=1920):
        self.img_w = img_w
        self.img_h = img_h
        self.sphGrid = SphGrid(img_h, img_w)
        
    def eval_iou(self, gt, result, sph = False):
        gt = gt.astype(bool)
        result = result.astype(bool)

        inters = np.sum((result & gt) * self.sphGrid.integr_grid) if sph else np.sum((result & gt)) 
        union = np.sum((result | gt) * self.sphGrid.integr_grid) if sph else np.sum((result | gt)) 

        if np.isclose(union, 0): return 1
        j = inters / union
        return j
        
    def eval_boundary(self, gt, result, sph=False, bound_th=0.008):

        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(result.shape))

        # Get the pixel boundaries of both masks
        fg_boundary = seg2bmap(result)
        gt_boundary = seg2bmap(gt)

        from skimage.morphology import disk

        fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
        gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            if sph:
                precision = np.sum(fg_match * self.sphGrid.integr_grid) / np.sum(fg_boundary * self.sphGrid.integr_grid)
                recall = np.sum(gt_match * self.sphGrid.integr_grid) / np.sum(gt_boundary * self.sphGrid.integr_grid)
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2 * precision * recall / (precision + recall)

        return F
        
        
def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + np.floor((y - 1) + height / h)
                    i = 1 + np.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap
