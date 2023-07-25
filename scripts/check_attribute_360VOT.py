import os
import math
import json
import argparse
import tqdm
import numpy as np
from rich.console import Console
from rich.table import Table

headers_full = ["length", "IV(illumination variation)", "BC(background clutter)",
                "DEF(deformable target)", "MB(motion blur)", "CM(camera motion)",
                "ROT(rotation)", "POC(partial occlusion)", "FOC(full occlusion)",
                "ARC(aspect ratio change)", "SV(scale variation)", "FM(fast motion)",
                "LR(low resolution)", "HR(high resolution)", "SA(stitching artifact)",
                "CB(cross border)", "FMS(fast motion on the sphere)", "LFoV(large FoV)",
                "LV(latitude variant)", "HL(high latitude)", "LD(large distortion)"]

headers_quantitative = ["length",  "FOC(full occlusion)", "ARC(aspect ratio change)", 
                        "SV(scale variation)", "FM(fast motion)", "LR(low resolution)", 
                        "HR(high resolution)", "CB(cross border)", 
                        "FMS(fast motion on the sphere)", "LFoV(large FoV)", 
                        "LV(latitude variant)", "HL(high latitude)"]

def checkOcclusion(annos):
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        if bbox["w"] < 1 or bbox["h"] < 1:
            return True
    
    return False

def checkAspectRatio2(annos):
    def getAR(w, h):
        if w > 0 and h > 0:
            return w / h
        return None

    first_ar_bbox = None
    first_ar_rbbox = None
    first_ar_bfov = None
    first_ar_rbfov = None

    arc_bbox = False
    arc_rbbox = False
    arc_bfov = False
    arc_rbfov = False
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        rbbox = anno["rbbox"]
        bfov = anno["bfov"]
        rbfov = anno["rbfov"]

        ar_bbox = getAR(bbox["w"], bbox["h"])
        ar_rbbox = getAR(rbbox["w"], rbbox["h"])
        ar_bfov = getAR(bfov["fov_v"], bfov["fov_h"])
        ar_rbfov = getAR(rbfov["fov_v"], rbfov["fov_h"])

        if first_ar_bbox is None and ar_bbox:
            first_ar_bbox = ar_bbox
        elif ar_bbox:
            arc = first_ar_bbox / ar_bbox
            if arc < 0.5 or arc > 2:
                arc_bbox = True

        if first_ar_rbbox is None and ar_rbbox:
            first_ar_rbbox = ar_rbbox
        elif ar_rbbox:
            arc = first_ar_rbbox / ar_rbbox
            if arc < 0.5 or arc > 2:
                arc_rbbox = True

        if first_ar_bfov is None and ar_bfov:
            first_ar_bfov = ar_bfov
        elif ar_bfov:
            arc = first_ar_bfov / ar_bfov
            if arc < 0.5 or arc > 2:
                arc_bfov = True

        if first_ar_rbfov is None and ar_rbfov:
            first_ar_rbfov = ar_rbfov
        elif ar_rbfov:
            arc = first_ar_rbfov / ar_rbfov
            if arc < 0.5 or arc > 2:
                arc_rbfov = True

    return arc_rbbox and arc_bbox and arc_rbfov and arc_bfov
         
def checkAspectRatio(annos):
    def getAR(w, h):
        if w > 0 and h > 0:
            return w / h
        return None

    first_ar_bbox = None
    first_ar_bbox_r = None
    first_ar_fovbbox = None
    first_ar_fovbbox_r = None

    arc_bbox = False
    arc_bbox_r = False
    arc_fovbbox = False
    arc_fovbbox_r = False
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        bbox_rotated = anno["rbbox"]
        fovBbox = anno["bfov"]
        fovBbox_rotated = anno["rbfov"]

        ar_bbox = getAR(bbox["w"], bbox["h"])
        ar_bbox_r = getAR(bbox_rotated["w"], bbox_rotated["h"])
        ar_fovbbox = getAR(fovBbox["fov_v"], fovBbox["fov_h"])
        ar_fovbbox_r = getAR(fovBbox_rotated["fov_v"], fovBbox_rotated["fov_h"])

        if first_ar_bbox is None and ar_bbox:
            first_ar_bbox = ar_bbox
        elif ar_bbox:
            arc = first_ar_bbox / ar_bbox
            if arc < 0.5 or arc > 2:
                arc_bbox = True

        if first_ar_bbox_r is None and ar_bbox_r:
            first_ar_bbox_r = ar_bbox_r
        elif ar_bbox_r:
            arc = first_ar_bbox_r / ar_bbox_r
            if arc < 0.5 or arc > 2:
                arc_bbox_r = True

        if first_ar_fovbbox is None and ar_fovbbox:
            first_ar_fovbbox = ar_fovbbox
        elif ar_fovbbox:
            arc = first_ar_fovbbox / ar_fovbbox
            if arc < 0.5 or arc > 2:
                arc_fovbbox = True

        if first_ar_fovbbox_r is None and ar_fovbbox_r:
            first_ar_fovbbox_r = ar_fovbbox_r
        elif ar_fovbbox_r:
            arc = first_ar_fovbbox_r / ar_fovbbox_r
            if arc < 0.5 or arc > 2:
                arc_fovbbox_r = True

    return arc_bbox #arc_bbox_r and arc_bbox and arc_fovbbox_r and arc_fovbbox
                       
def checkScaleVariant(annos):
    bbox_area_pre = 0
    rbbox_area_pre = 0 
    sv_b = False
    sv_br = False
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        rbbox = anno["rbbox"]

        bbox_area = bbox["w"] * bbox["h"]
        rbbox_area = rbbox["w"] * rbbox["h"]

        if bbox_area_pre > 1:
            if bbox_area > 1:
                sv = bbox_area_pre / bbox_area
                if sv < 0.5 or sv > 2:
                    sv_b = True
        elif bbox_area > 1:
            bbox_area_pre = bbox_area

        if rbbox_area_pre > 1:
            if rbbox_area > 1:
                sv = rbbox_area_pre / rbbox_area
                if sv < 0.5 or sv > 2:
                    sv_br = True
        elif rbbox_area > 1:
            rbbox_area_pre = rbbox_area   
    return sv_br and sv_b

def checkFastMotion(annos):
    center_x = None
    center_y = None
    w = 0
    h = 0

    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        if bbox["h"] < 1 or bbox["w"] < 1:
            continue

        if center_x is not None or center_y is not None:
            motion_x = bbox["cx"] - center_x
            motion_y = bbox["cy"] - center_y
            motion_x = 3840 - abs(motion_x) if abs(motion_x) > 2000 else motion_x
            motion = motion_x * motion_y
            #if motion > bbox["w"] * bbox["h"]:
            #    return True
            if abs(motion_y) > bbox["h"] or abs(motion_x) > bbox["w"]:
                return True

        center_x = bbox["cx"]
        center_y = bbox["cy"]
        w = bbox["w"]
        h = bbox["h"]

    return False
     
def checkResolution(annos, low_th, high_th):
    low = False
    high = False
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"] #bbox_rotated
        area = bbox["h"] * bbox["w"]
        if area < low_th:
            low = True
        if area > high_th:
            high = True
        if low and high:
            break
    return low, high

def checkCross(annos, left=0, right=3840, sequence=""):
    for index in annos:
        anno = annos[index]
        bbox = anno["bbox"]
        rbbox = anno["rbbox"]

        if bbox["cx"] > -1 and bbox["cx"] < right:
            pass
        else:
            print("bbox", sequence, index, bbox["cx"])
        if rbbox["cx"] > -1 and rbbox["cx"] < right:
            pass
        else:
            print("rbbox:",sequence, index, rbbox["cx"])
         
        x1 = bbox["cx"] - bbox["w"] * 0.5
        x2 = bbox["cx"] + bbox["w"] * 0.5

        if x1 < left or x2 > right:
            return True

    return False

def checkFastMotionSphere(annos):
    def lonlat2xyz(lon, lat):
        lon = lon / 180 * np.pi
        lat = lat / 180 * np.pi
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(-lat)
        z = np.cos(lat) * np.cos(lon)
        return x, y, z

    center_x = None
    center_y = None
    center_z = None
    fov_h = 0
    fov_v = 0

    for index in annos:
        anno = annos[index]
        bfov = anno["bfov"]
        x, y, z = lonlat2xyz(bfov["clon"], bfov["clat"])

        if bfov["fov_v"] < 1 or bfov["fov_h"] < 1:
            continue
        
        if center_x is not None:
            ab = center_x * x + center_y * y + center_z * z 
            a = math.sqrt(center_x**2 + center_y**2 + center_z**2)
            b = math.sqrt(x**2 + y**2 + z**2)
            cos_ab = ab / (a * b) 
            cos_ab = 1 if cos_ab > 1 else cos_ab
            angle_ab = np.arccos(cos_ab) / np.pi * 180
            #print(angle_ab)
            if angle_ab > 5 and (angle_ab > fov_h or angle_ab>fov_v):
                return True

        center_x = x
        center_y = y
        center_z = z
        fov_v = bfov["fov_v"]
        fov_h = bfov["fov_h"]

    return False

def checkFoV(annos, fov_th=90):
    for index in annos:
        anno = annos[index]
        bfov = anno["bfov"]
        rbfov = anno["rbfov"]

        if bfov["fov_h"] > fov_th or bfov["fov_v"] > fov_th:
            return True

    return False

def checkLatitude(annos, range_th=50):
    lat = []
    for index in annos:
        anno = annos[index]
        bfov = anno["bfov"]
        clat = bfov["clat"]
        lat.append(clat)
    lat_min, lat_max = np.min(lat), np.max(lat)
    LV = (lat_max - lat_min) > 50
    HL = lat_max > 66.5 or lat_min < -66.5
    return LV, HL

def getbenchmark(excel_file):
    import pandas as pd
    data = pd.read_excel(excel_file)
    #print(data);exit(0)
    count = 1
    files = []
    meta = {}
    for index, row in data.iterrows():
        sequence = row["sequence"]
        files.append(sequence)   
        meta.update({"{:04d}".format(sequence): row})
    #print(meta)
    return files, meta

def checkAttribute(args): 
    if args.excel:
        benchmark_list, benchmark_attribute = getbenchmark(args.excel)
    
    sequences =  sorted(os.listdir(args.dir), key=lambda x: x.lower())

    table = Table(title="360VOT Attribute")
    table.add_column("Seq", justify="right", no_wrap=True)
    style_str=["cyan", "magenta"]
    for i, header in enumerate(headers_quantitative):
        table.add_column(header.split("(")[0], justify="center", style=style_str[i%2])

    for sequence in tqdm.tqdm(sequences):
        attribute_dict = {}
        #print(sequence)

        image_path = os.path.join(args.dir, sequence, "image")
        images = sorted(os.listdir(image_path))
        anno_files = os.path.join(args.dir, sequence, "label.json")
        with open(anno_files, "r") as output:
            annotations = json.load(output)

        #target
        #length
        #IV(illumination variation)
        #BC(background clutter)
        #DEF(deformable target)
        #MB(motion blur)       
        #CM(camera motion)            
        #SA(stitching artifact)
        #ROT(rotation) the target rotates related to the frames. 
        #attribute_dict["ROT(rotation)"] = 1 if checkRotation(annotations) else math.nan
        #POC(partial occlusion) the target is partially occluded, need to manually annotate
        #LD(large distortion) the target suffers large distortion on the omnidirectional image
        #attribute["LD(large distortion)"] = 1 if LV or HL else math.nan
        #attribute["LD(large distortion)"] = 1 if checkDistortion(annotations) else math.nan

        attribute_dict["length"] = len(images)

        #FOC(full occlusion) the target is fully occluded, check if the area of the box is zero
        attribute_dict["FOC(full occlusion)"] = 1 if checkOcclusion(annotations) else math.nan

        #ARC(aspect ratio change) the ratio of the bounding-box aspect ratio of the first and the current frame is outside the range[0.5, 2]
        attribute_dict["ARC(aspect ratio change)"] = 1 if checkAspectRatio(annotations) else math.nan

        #SV(scale variation) the ratio of the bounding-box area of the first and the current frame is "outside" the range [0.5, 2]
        attribute_dict["SV(scale variation)"] = 1 if checkScaleVariant(annotations) else math.nan

        #FM(fast motion) the motion of the traget annotation center between contiguous frames exceed its own size.
        attribute_dict["FM(fast motion)"] = 1 if checkFastMotion(annotations) else math.nan

        low, high = checkResolution(annotations, 1000, 250000)
        #LR(low resolution) the area of the target annotation is less than 1000 pixels in at least one frame.
        attribute_dict["LR(low resolution)"] = 1 if low else math.nan
        #HR(high resolution) the area of the target (rotated) (fov) bbox /annotation is larger than 500^2 pixels in at least one frame
        attribute_dict["HR(high resolution)"] = 1 if high else math.nan

        #CB(cross border) the target cross the border of the frame and pratially appear on the other side.
        attribute_dict["CB(cross border)"] = 1 if checkCross(annotations, sequence=sequence) else math.nan
        
        #FMS(fast motion on the sphere) the motion angle on the spherical surface of the target center is larger than its last BFOV
        attribute_dict["FMS(fast motion on the sphere)"] = 1 if checkFastMotionSphere(annotations) else math.nan
            
        #LFoV(large FoV) the vertical or horizontal FoV of the bfov is larger than 90 degree.
        attribute_dict["LFoV(large FoV)"] = 1 if checkFoV(annotations) else math.nan

        LV, HL = checkLatitude(annotations)
        #LV(latitude variant) the range of the latitude of the target center is larger than 50 degree
        attribute_dict["LV(latitude variant)"] = 1 if LV else math.nan

        #HL(high latitude) the center of the target lies in the "frigid zone"
        attribute_dict["HL(high latitude)"] = 1 if HL else math.nan

        #verify 360vot attribute excel
        if args.excel:
            attribute_360VOT = benchmark_attribute[sequence]
            for att in attribute_dict:
                assert attribute_dict[att] == attribute_360VOT[att] or \
                (math.isnan(attribute_dict[att]) and math.isnan(attribute_360VOT[att])),\
                    ("sequence: {}, attr: {}, {}, {}".format(sequence, att, attribute_dict[att], 
                                                        attribute_360VOT[att]))

        table_row = [sequence]
        for att in attribute_dict:
            if att=="length":
                table_row.append(str(attribute_dict[att]))
            elif attribute_dict[att] == 1:
                table_row.append("1")
            else:
                table_row.append("")
        table.add_row(*table_row)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dir', type=str, default="/homes/Dataset/360tracking/benchmark", help='path to dataset')
    parser.add_argument('--excel', type=str, default="", help='path to save excel')

    args = parser.parse_args()
    checkAttribute(args)
