import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from equilib import Equi2Pers, Equi2Equi
import time
import json

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, ".."))
sys.path.append(os.path.join(this_dir, "..", "lib"))
from lib.utils import *
from lib.omni import OmniImage


def compare_bfov(img, omni=None):
    if omni is None:
        omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color1 = (48,188,227)
    color2 = (224,161,101)
    bfov1 = Bfov(0, 30, 60, 60)
    bfov2 = Bfov(0, 30, 120, 120)
    bfov3 = Bfov(0, 30, 170, 170)

    img_tangent_bfov1 = omni.plot_bfov(img.copy(), bfov1, 0, color=color1, size=30)
    img_tangent_bfov2 = omni.plot_bfov(img.copy(), bfov2, 0, color=color1, size=30)
    img_tangent_bfov3 = omni.plot_bfov(img.copy(), bfov3, 0, color=color1, size=30)

    img_tangent_bfov_region1, _, _ = omni.crop_bfov(img_tangent_bfov1, bfov1, 0)
    img_tangent_bfov_region2, _, _ = omni.crop_bfov(img_tangent_bfov2, bfov2, 0)
    img_tangent_bfov_region3, _, _ = omni.crop_bfov(img_tangent_bfov3, bfov3, 0)

    img_extend_bfov1 = omni.plot_bfov(img.copy(), bfov1, color=color2, size=30)
    img_extend_bfov2 = omni.plot_bfov(img.copy(), bfov2, color=color2, size=30)
    img_extend_bfov3 = omni.plot_bfov(img.copy(), bfov3, color=color2, size=30)

    img_extend_bfov_region1, _, _ = omni.crop_bfov(img_extend_bfov1, bfov1)
    img_extend_bfov_region2, _, _ = omni.crop_bfov(img_extend_bfov2, bfov2)
    img_extend_bfov_region3, _, _ = omni.crop_bfov(img_extend_bfov3, bfov3)

    fig = plt.figure("Comparison of BFoV")

    ax1 = plt.subplot2grid((2, 9), (0, 0), colspan=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = plt.subplot2grid((2, 9), (0, 2))
    ax2.set_axis_off()
    ax3 = plt.subplot2grid((2, 9), (0, 3), colspan=2)
    ax3.set_axis_off()
    ax4 = plt.subplot2grid((2, 9), (0, 5))
    ax4.set_axis_off()
    ax5 = plt.subplot2grid((2, 9), (0, 6), colspan=2)
    ax5.set_axis_off()
    ax6 = plt.subplot2grid((2, 9), (0, 8))
    ax6.set_axis_off()

    ax7 = plt.subplot2grid((2, 9), (1, 0), colspan=2)
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8 = plt.subplot2grid((2, 9), (1, 2))
    ax8.set_axis_off()
    ax9 = plt.subplot2grid((2, 9), (1, 3), colspan=2)
    ax9.set_axis_off()
    ax10 = plt.subplot2grid((2, 9), (1, 5))
    ax10.set_axis_off()
    ax11 = plt.subplot2grid((2, 9), (1, 6), colspan=2)
    ax11.set_axis_off()
    ax12 = plt.subplot2grid((2, 9), (1, 8))
    ax12.set_axis_off()


    ax1.set_title("(0, 30, 60, 60)")
    ax1.set_ylabel("Tangent")
    ax1.imshow(img_tangent_bfov1)
    ax3.set_title("(0, 30, 120, 120)")
    ax3.imshow(img_tangent_bfov2)
    ax5.set_title("(0, 30, 170, 170)")
    ax5.imshow(img_tangent_bfov3)
    ax2.imshow(img_tangent_bfov_region1)
    ax4.imshow(img_tangent_bfov_region2)
    ax6.imshow(img_tangent_bfov_region3)

    ax7.set_ylabel("Extended")
    ax7.imshow(img_extend_bfov1)
    ax9.imshow(img_extend_bfov2)
    ax11.imshow(img_extend_bfov3)
    ax8.imshow(img_extend_bfov_region1)
    ax10.imshow(img_extend_bfov_region2)
    ax12.imshow(img_extend_bfov_region3)

    plt.show()

def compare_anno(img_path, anno_file, omni=None):
    img_name = img_path.split("/")[-1].split("_")[-1] #"000216.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if omni is None:
        omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])
    with open(anno_file, "r") as output:
        annotations = json.load(output)

    anno = annotations[img_name]
    bfov = dict2Bfov(anno["bfov"])
    rbfov = dict2Bfov(anno["rbfov"])
    bbox = dict2Bbox(anno["bbox"])
    rbbox = dict2Bbox(anno["rbbox"])

    img_bfov = omni.plot_bfov(img.copy(), bfov, color=(224,161,101), size=20)
    img_rbfov = omni.plot_bfov(img.copy(), rbfov, color=(224,107,96), size=20)
    img_bbox = omni.plot_bbox(img.copy(), bbox, color=(48,188,227), size=20)
    img_rbbox = omni.plot_bbox(img.copy(), rbbox, color=(121,224,167), size=20)

    omni.plot_bfov(img, bfov, color=(224,161,101), size=20)
    omni.plot_bfov(img, rbfov, color=(224,107,96), size=20)
    omni.plot_bbox(img, bbox, color=(48,188,227), size=20)
    omni.plot_bbox(img, rbbox, color=(121,224,167), size=20)

    img_bfov_crop, _, _ = omni.crop_bfov(img_bfov, bfov)
    img_rbfov_crop, _, _ = omni.crop_bfov(img_rbfov, rbfov)

    img_bbox_crop, _, _ = omni.crop_bbox(img_bbox, bbox)
    img_rbbox_crop, _, _ = omni.crop_bbox(img_rbbox, rbbox)

    fig = plt.figure("Comparison between BFoV and BBox")

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax1.set_axis_off()
    ax2 = plt.subplot2grid((2, 4), (0, 2))
    ax2.set_axis_off()
    ax3 = plt.subplot2grid((2, 4), (0, 3))
    ax3.set_axis_off()
    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax4.set_axis_off()
    ax5 = plt.subplot2grid((2, 4), (1, 3))
    ax5.set_axis_off()

    ax1.imshow(img)
    ax1.set_title("Tracking a train")
    ax2.imshow(img_bbox_crop)
    ax2.set_title("BBox")
    ax3.imshow(img_rbbox_crop)
    ax3.set_title("rBBox")
    ax4.imshow(img_bfov_crop)
    ax4.set_title("BFoV")
    ax5.imshow(img_rbfov_crop)
    ax5.set_title("rBFoV")

    plt.show()

def rotate_img(img, omni=None):
    # move the marker to image center
    if omni is None:
        omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # marker
    u, v = omni.lonlat2uv(lon=60/180*np.pi, lat=80/180*np.pi)
    cv2.circle(img, (int(u), int(v)), 1, (255, 0, 0), 50)
    u2, v2 = omni.lonlat2uv(lon=-60/180*np.pi, lat=-80/180*np.pi)
    cv2.circle(img, (int(u2), int(v2)), 1, (0, 0, 255), 50)
    u3, v3 = omni.lonlat2uv(lon=0/180*np.pi, lat=80/180*np.pi)
    cv2.circle(img, (int(u3), int(v3)), 1, (0, 255, 255), 50)

    rot_img1, rotation_mat1 = omni.rot_image(img, pitch=0, yaw=80, roll=0)
    rot_img2, rotation_mat2 = omni.rot_image(img, pitch=0, yaw=-80, roll=0)
    rot_img3, rotation_mat3 = omni.rot_image(img, pitch=60, yaw=0, roll=0)
    rot_img4, rotation_mat4 = omni.rot_image(img, pitch=-60, yaw=0, roll=0)
    rot_img5, rotation_mat5 = omni.rot_image(img, pitch=0, yaw=0, roll=30)
    rot_img6, rotation_mat6 = omni.rot_image(img, pitch=0, yaw=0, roll=-30)
    rot_img7, rotation_mat7 = omni.rot_image(img, pitch=60, yaw=80, roll=0)
    rot_img8, rotation_mat8 = omni.rot_image(img, pitch=-60, yaw=-80, roll=0)
    rot_img9, rotation_mat9 = omni.rot_image(img, pitch=60, yaw=80, roll=30)
    rot_img10, rotation_mat10 = omni.rot_image(img, pitch=-60, yaw=-80, roll=-30)
    rot_img11, rotation_mat11 = omni.rot_image(img, pitch=-60, yaw=-80, roll=-88)

    fig = plt.figure("Rotating 360 image")

    ax1 = plt.subplot(3, 4, 1)
    ax1.set_axis_off()
    ax1.set_title("0, 0, 0")
    ax1.imshow(img)
    ax1 = plt.subplot(3, 4, 2)
    ax1.set_axis_off()
    ax1.set_title("0, 80, 0")
    ax1.imshow(rot_img1)
    ax1 = plt.subplot(3, 4, 3)
    ax1.set_axis_off()
    ax1.set_title("0, -80, 0")
    ax1.imshow(rot_img2)
    ax1 = plt.subplot(3, 4, 4)
    ax1.set_axis_off()
    ax1.set_title("60, 0, 0")
    ax1.imshow(rot_img3)
    ax1 = plt.subplot(3, 4, 5)
    ax1.set_axis_off()
    ax1.set_title("-60, 0, 0")
    ax1.imshow(rot_img4)
    ax1 = plt.subplot(3, 4, 6)
    ax1.set_axis_off()
    ax1.set_title("0, 0, 30")
    ax1.imshow(rot_img5)
    ax1 = plt.subplot(3, 4, 7)
    ax1.set_axis_off()
    ax1.set_title("0, 0, -30")
    ax1.imshow(rot_img6)
    ax1 = plt.subplot(3, 4, 8)
    ax1.set_axis_off()
    ax1.set_title("60, 80, 0")
    ax1.imshow(rot_img7)
    ax1 = plt.subplot(3, 4, 9)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, 0")
    ax1.imshow(rot_img8)
    ax1 = plt.subplot(3, 4, 10)
    ax1.set_axis_off()
    ax1.set_title("60, 80, 30")
    ax1.imshow(rot_img9)
    ax1 = plt.subplot(3, 4, 11)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, -30")
    ax1.imshow(rot_img10)
    ax1 = plt.subplot(3, 4, 12)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, -88")
    ax1.imshow(rot_img11)
    #plt.savefig("rotate_img.jpg")
    plt.show()

"""
def rotate_img_equilib(img):

    equi2equi = Equi2Equi(img.shape[1], img.shape[0])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
 
    rot_img1 = equi2equi(img, rots={"pitch":0, "yaw":80/180*np.pi, "roll":0})
    rot_img2 = equi2equi(img, rots={"pitch":0, "yaw":-80/180*np.pi, "roll":0})
    rot_img3 = equi2equi(img, rots={"pitch":60/180*np.pi, "yaw":0, "roll":0})
    rot_img4 = equi2equi(img, rots={"pitch":-60/180*np.pi, "yaw":0, "roll":0})
    rot_img5 = equi2equi(img, rots={"pitch":0, "yaw":0, "roll":30/180*np.pi})
    rot_img6 = equi2equi(img, rots={"pitch":0, "yaw":0, "roll":-30/180*np.pi})
    rot_img7 = equi2equi(img, rots={"pitch":60/180*np.pi, "yaw":80/180*np.pi, "roll":0})
    rot_img8 = equi2equi(img, rots={"pitch":-60/180*np.pi, "yaw":-80/180*np.pi, "roll":0})
    rot_img9 = equi2equi(img, rots={"pitch":60/180*np.pi, "yaw":80/180*np.pi, "roll":30/180*np.pi})
    rot_img10 = equi2equi(img, rots={"pitch":-60/180*np.pi, "yaw":-80/180*np.pi, "roll":-30/180*np.pi})
    rot_img11 = equi2equi(img, rots={"pitch":-60/180*np.pi, "yaw":-80/180*np.pi, "roll":-88/180*np.pi})

    fig = plt.figure("Rotating 360 image")

    ax1 = plt.subplot(3, 4, 1)
    ax1.set_axis_off()
    ax1.set_title("0, 0, 0")
    ax1.imshow(np.transpose(img, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 2)
    ax1.set_axis_off()
    ax1.set_title("0, 80, 0")
    ax1.imshow(np.transpose(rot_img1, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 3)
    ax1.set_axis_off()
    ax1.set_title("0, -80, 0")
    ax1.imshow(np.transpose(rot_img2, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 4)
    ax1.set_axis_off()
    ax1.set_title("60, 0, 0")
    ax1.imshow(np.transpose(rot_img3, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 5)
    ax1.set_axis_off()
    ax1.set_title("-60, 0, 0")
    ax1.imshow(np.transpose(rot_img4, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 6)
    ax1.set_axis_off()
    ax1.set_title("0, 0, 30")
    ax1.imshow(np.transpose(rot_img5, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 7)
    ax1.set_axis_off()
    ax1.set_title("0, 0, -30")
    ax1.imshow(np.transpose(rot_img6, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 8)
    ax1.set_axis_off()
    ax1.set_title("60, 80, 0")
    ax1.imshow(np.transpose(rot_img7, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 9)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, 0")
    ax1.imshow(np.transpose(rot_img8, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 10)
    ax1.set_axis_off()
    ax1.set_title("60, 80, 30")
    ax1.imshow(np.transpose(rot_img9, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 11)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, -30")
    ax1.imshow(np.transpose(rot_img10, (1, 2, 0)))
    ax1 = plt.subplot(3, 4, 12)
    ax1.set_axis_off()
    ax1.set_title("-60, -80, -88")
    ax1.imshow(np.transpose(rot_img11, (1, 2, 0)))
    plt.savefig("rotate_img_equilib.jpg")
    #plt.show()
"""
    
def localAnno2GlobalAnno(img, anno_file, omni=None):
    img_name = img_path.split("/")[-1].split("_")[-1] #"000193.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if omni is None:
        omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])
    with open(anno_file, "r") as output:
        annotations = json.load(output)

    last_img_name = "{:06d}.jpg".format(int(img_name.split(".")[0])-1)
    #print(last_img_name)
    anno = annotations[last_img_name]
    bfov = dict2Bfov(anno["bfov"])
    rbfov = dict2Bfov(anno["rbfov"])
    bfov = scaleBFoV(bfov, 5)

    img_bfov, u_bfov, v_bfov = omni.crop_bfov(img, bfov)

    # stag 2
    bbox = Bbox(490, 312, 215, 150, 0)
    rbbox = Bbox(485, 300, 230, 102, 30)
    local_bbox_img = img_bfov.copy()
    local_rbbox_img = img_bfov.copy()
    omni.plot_bbox(local_bbox_img, bbox, color=(255, 0, 0), size=8)
    omni.plot_bbox(local_rbbox_img, rbbox, color=(255, 0, 0), size=8)

    bfovFromBbox = omni.localBbox2Bfov(bbox, u_bfov, v_bfov, need_rotation=False)
    rbfovFromBbox = omni.localBbox2Bfov(bbox, u_bfov, v_bfov, need_rotation=True)
    bbox_global = omni.localBbox2Bbox(bbox, u_bfov, v_bfov, need_rotation=False)
    rbbox_global = omni.localBbox2Bbox(bbox, u_bfov, v_bfov, need_rotation=True)

    img_bfovFromBbox = omni.plot_bfov(img.copy(), bfovFromBbox, color=(224,161,101), size=50)
    img_rbfovFromBbox = omni.plot_bfov(img.copy(), rbfovFromBbox, color=(224,107,96), size=50)
    img_bbox_global = omni.plot_bbox(img.copy(), bbox_global, color=(48,188,227), size=50)
    img_rbbox_global = omni.plot_bbox(img.copy(), rbbox_global, color=(121,224,167), size=50)


    bfovFromrBbox= omni.localBbox2Bfov(rbbox, u_bfov, v_bfov, need_rotation=False)
    rbfovFromrBbox= omni.localBbox2Bfov(rbbox, u_bfov, v_bfov, need_rotation=True)
    bbox_global_rot = omni.localBbox2Bbox(bbox, u_bfov, v_bfov, need_rotation=False)
    rbbox_global_rot = omni.localBbox2Bbox(rbbox, u_bfov, v_bfov, need_rotation=True)

    img_bfovFromrBbox = omni.plot_bfov(img.copy(), bfovFromrBbox, color=(224,161,101), size=50)
    img_rbfovFromrBbox = omni.plot_bfov(img.copy(), rbfovFromrBbox, color=(224,107,96), size=50)
    img_bbox_global_rot = omni.plot_bbox(img.copy(), bbox_global_rot, color=(48,188,227), size=50)
    img_rbbox_global_rot = omni.plot_bbox(img.copy(), rbbox_global_rot, color=(121,224,167), size=50)

    fig = plt.figure("Local anno to global anno")
    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.title("bbox on the local region")
    plt.imshow(local_bbox_img)  
    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.title("local bbox to global bbox")
    plt.imshow(img_bbox_global)
    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.title("local bbox to bfov")
    plt.imshow(img_bfovFromBbox)

    plt.subplot(2, 3, 4)
    plt.axis("off")
    plt.title("rbbox on the local region")
    plt.imshow(local_rbbox_img)
    plt.subplot(2, 3, 5)
    plt.axis("off")
    plt.title("local rbbox to rbfov")
    plt.imshow(img_rbbox_global_rot)
    plt.subplot(2, 3, 6)
    plt.axis("off")
    plt.title("local rbbox to rbfov")
    plt.imshow(img_rbfovFromrBbox)
    plt.show()

def visualizeMask2AnnoProcess(img, img_mask, omni=None):
    if omni is None:
        omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])
    img_mask_raw = img_mask.copy()
    mask = img_mask[:, :, 0].copy() # > 127
    #mask = mask.astype(np.uint8)
    img_mask[np.array(mask)>0]=[0, 0 , 255]

    # get the blended image
    blended_img = cv2.addWeighted(img, 0.3, img_mask, 0.7, 0.0)
    # get the initial position
    contours1 = convert_mask_to_polygon(mask, max_only=True) # need to consider disapper case
    cx_contours, cy_contours = np.mean(contours1, axis=0)
    blended_img_with_point = cv2.circle(blended_img.copy(), (int(cx_contours), int(cy_contours)), 1, [255, 0 , 0], 60)
    
    # step 2 move c1 to the horizontal center
    c_lon, c_lat = omni.uv2lonlat(cx_contours, cy_contours)
    mask_rotation_bbox, R = omni.align_center_by_lonlat(mask, c_lon, 0)
    mask_image_rotation_horizontal, R = omni.align_center_by_lonlat(blended_img, c_lon, 0)

    contours2 = convert_mask_to_polygon(mask_rotation_bbox, integrate=True)
    # bbox
    lx, ly, w, h = cv2.boundingRect(contours2)
    cx  = lx + w * 0.5
    cy  = ly + h * 0.5
    mask_image_rotation_horizontal_bbox = omni.plot_bbox(mask_image_rotation_horizontal.copy(), Bbox(cx, cy, w, h, 0), size=50)
    # rbbox
    rect_xy, rect_wh, rotation_angle = cv2.minAreaRect(contours2)
    cx, cy = rect_xy
    w, h = rect_wh
    mask_image_rotation_horizontal_rbbox= omni.plot_bbox(mask_image_rotation_horizontal.copy(), Bbox(cx, cy, w, h, rotation_angle), size=50)

    # directly get the bbox from the mask with the function.
    bbox_final = omni.mask2Bbox(img_mask_raw, need_rotation=False)
    rbbox_final = omni.mask2Bbox(img_mask_raw)
    #if bbox_final is None:
    image_bbox_final = omni.plot_bbox(img.copy(), bbox_final, color=(227,188,48), size=50)
    image_rbbox_final = omni.plot_bbox(img.copy(), rbbox_final, color=(167,224,121), size=50)


    #  Step 2 (r)bfov
    mask_rotation_bfov, R = omni.align_center(mask, cx_contours, cy_contours)
    mask_image_rotation_center, R = omni.align_center(blended_img, cx_contours, cy_contours)
    v, u = np.where(mask_rotation_bfov>127)
    clon, clat = omni.get_inverse_lonlat(R, np.mean(u), np.mean(v))
    cv2.circle(mask_image_rotation_center, (int(np.mean(u)), int(np.mean(v))), 1, [255, 0 , 0], 60)
    # Step 2.1
    mask_image_rotation2, R2 = omni.align_center_by_lonlat(mask, clon, clat)
    mask_image_rotation_recenter, R = omni.align_center_by_lonlat(blended_img, clon, clat)
    contours2 = convert_mask_to_polygon(mask_image_rotation2, integrate=True)
    rect_xy, rect_wh, rect_ang = cv2.minAreaRect(contours2)
    clon2, clat2 = omni.get_inverse_lonlat(R2, rect_xy[0], rect_xy[1])
    #cv2.circle(mask_image_rotation_recenter, (int(rect_xy[0]), int(rect_xy[1])), 1, [255, 0 , 0], 60)
    
    # Step 3
    # bfov
    rotation_angle = 0
    mask_image_rotation3, R3 = omni.align_center_by_lonlat(mask, clon2, clat2, ang2rad(rotation_angle))
    mask_image_rotation_finalcenter_bfov, R3 = omni.align_center_by_lonlat(blended_img, clon2, clat2, ang2rad(rotation_angle))
    contours3 = convert_mask_to_polygon(mask_image_rotation3, integrate=True)
    u = contours3[:, 0]
    v = contours3[:, 1]
    min_u, max_u = np.min(u), np.max(u)
    min_v, max_v = np.min(v), np.max(v)
    cv2.rectangle(mask_image_rotation_finalcenter_bfov, (min_u, min_v), (max_u, max_v), (255, 0, 0), 50)
    # based on the bbox, calculate the center and fov of bfov
    cu = (min_u+max_u)*0.5
    cv = (min_v+max_v)*0.5
    clon3, clat3 = omni.get_inverse_lonlat(R3, cu, cv)
    clon = rad2ang(clon3)
    clat = rad2ang(clat3)
    # use a bbox to approximate
    min_lon, min_lat = omni.uv2lonlat(min_u, max_v)
    max_lon, max_lat = omni.uv2lonlat(max_u, min_v)
    fov_h = rad2ang(max_lon - min_lon)
    fov_v = rad2ang(max_lat - min_lat)
    bfov = Bfov(clon, clat, fov_h, fov_v, rotation_angle)

    # rbfov
    rotation_angle = rect_ang if rect_wh[0] > rect_wh[1] else -90 + rect_ang
    mask_image_rotation3, R3 = omni.align_center_by_lonlat(mask, clon2, clat2, ang2rad(rotation_angle))
    mask_image_rotation_finalcenter_rbfov, R3 = omni.align_center_by_lonlat(blended_img, clon2, clat2, ang2rad(rotation_angle))
    contours3 = convert_mask_to_polygon(mask_image_rotation3, integrate=True)
    u = contours3[:, 0]
    v = contours3[:, 1]
    min_u, max_u = np.min(u), np.max(u)
    min_v, max_v = np.min(v), np.max(v)
    cv2.rectangle(mask_image_rotation_finalcenter_rbfov, (min_u, min_v), (max_u, max_v), (255, 0, 0), 50)
    # based on the bbox, calculate the center and fov of bfov
    cu = (min_u+max_u)*0.5
    cv = (min_v+max_v)*0.5
    clon3, clat3 = omni.get_inverse_lonlat(R3, cu, cv)
    clon = rad2ang(clon3)
    clat = rad2ang(clat3)
    # use a bbox to approximate
    min_lon, min_lat = omni.uv2lonlat(min_u, max_v)
    max_lon, max_lat = omni.uv2lonlat(max_u, min_v)
    fov_h = rad2ang(max_lon - min_lon)
    fov_v = rad2ang(max_lat - min_lat)
    rbfov = Bfov(clon, clat, fov_h, fov_v, rotation_angle)

    #directly get the bfov from the mask with the function.
    bfov_final = omni.mask2Bfov(img_mask_raw, need_rotation=False)
    rbfov_final = omni.mask2Bfov(img_mask_raw)
    print("bfov: ", bfov.tolist())
    print("bfov_final: ", bfov_final.tolist())
    print("rbfov: ", rbfov.tolist())
    print("rbfov_final: ", rbfov_final.tolist())

    image_bfov_final = omni.plot_bfov(img.copy(), bfov, color=(101,161,224), size=50)
    image_rbfov_final = omni.plot_bfov(img.copy(), rbfov, color=(96,107,224), size=50)
    
    fig = plt.figure("Visualize mask to anno process")
    
    plt.subplot2grid((4, 5), (1, 0))
    plt.axis("off")
    plt.title("image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (2, 0))
    plt.axis("off")
    plt.title("masked image")
    plt.imshow(cv2.cvtColor(blended_img_with_point, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (0, 1), rowspan=2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_horizontal, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (0, 3))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_horizontal_bbox, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (1, 3))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_horizontal_rbbox, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (0, 4))
    plt.axis("off")
    plt.title("BBox")
    plt.imshow(cv2.cvtColor(image_bbox_final, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (1, 4))
    plt.axis("off")
    plt.title("rBBox")
    plt.imshow(cv2.cvtColor(image_rbbox_final, cv2.COLOR_BGR2RGB))

    plt.subplot2grid((4, 5), (2, 1), rowspan=2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_center, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (2, 2), rowspan=2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_recenter, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (2, 3))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_finalcenter_rbfov, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (3, 3))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(mask_image_rotation_finalcenter_bfov, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (2, 4))
    plt.axis("off")
    plt.title("BFoV")
    plt.imshow(cv2.cvtColor(image_bfov_final, cv2.COLOR_BGR2RGB))
    plt.subplot2grid((4, 5), (3, 4))
    plt.axis("off")
    plt.title("rBFoV")
    plt.imshow(cv2.cvtColor(image_rbfov_final, cv2.COLOR_BGR2RGB))

    plt.show()


if __name__ == "__main__":
    img = cv2.imread(os.path.join(this_dir, "..", "asset", "0030_000064.jpg"))
    omni = OmniImage(img_h=img.shape[0], img_w=img.shape[1])
    # Figure 3
    compare_bfov(img, omni)
    
    # test image rotation
    #t0 = time.time()
    rotate_img(img, omni)
    #print("rotate_img takes {}".format(time.time()-t0))
    #t0 = time.time()
    #rotate_img_equilib(img)
    #print("rotate_img_equilib takes {}".format(time.time()-t0))

    # Figure 2
    img_path = os.path.join(this_dir, "..", "asset", "0115_000216.jpg")
    anno_file = os.path.join(this_dir, "..", "asset", "0115_label.json")
    compare_anno(img_path, anno_file, omni)

    # Figure 4
    img_path = os.path.join(this_dir, "..", "asset", "0098_000193.jpg")
    anno_file = os.path.join(this_dir, "..", "asset", "0098_label.json")
    localAnno2GlobalAnno(img_path, anno_file, omni)

    # Figure 3 in supp
    img = cv2.imread(os.path.join(this_dir, "..", "asset", "0117_000211.jpg"))
    img_mask = cv2.imread(os.path.join(this_dir, "..", "asset", "0117_000211_0.png"))
    visualizeMask2AnnoProcess(img, img_mask, omni)