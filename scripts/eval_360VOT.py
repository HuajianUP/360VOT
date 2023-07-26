import os
import sys
import argparse
import json
import numpy as np
import tqdm
import pandas as pd
from multiprocessing import Pool
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, ".."))
sys.path.append(os.path.join(this_dir, "..", "eval"))
sys.path.append(os.path.join(this_dir, "..", "lib"))
from eval.OPEBenchmark_360VOT import eval_success, eval_precision, eval_norm_precision, eval_angle_precision, show_result
import eval.draw_success_precision as plt_sp

import lib.utils as utils

def opt():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-d', '--dataset_dir', type=str, default="/homes/Dataset/360tracking/360VOT-release/360VOT",help='dataset path')
    parser.add_argument('-b', '--bbox_dir', type=str, default="",help='bbox result path')
    parser.add_argument('-rb', '--rbbox_dir', type=str, default="",help='rotated bbox result path')
    parser.add_argument('-f', '--bfov_dir', type=str, default="",help='bfov result path')
    parser.add_argument('-rf', '--rbfov_dir', type=str, default="",help='rotated bfov result path')
    parser.add_argument('-a', '--attribute', type=str, default="",help='attribute excel path') #360VOT attribute final.xlsx
    parser.add_argument('-v', '--show_video_level', dest='show_video_level',action='store_true')
    parser.add_argument('-p', '--plot_curve', dest='plot_results',action='store_true')
    parser.add_argument('-s', '--save_path', type=str, default=None,help='path to save the plot figure') 

    parser.set_defaults(show_video_level=False)
    parser.set_defaults(plot_curve=False)

    return parser.parse_args()

def loadDataset(dir):
    sequences = sorted(os.listdir(dir))
    gt_bbox_dict = {}
    gt_rbbox_dict = {}
    gt_bfov_dict = {}
    gt_rbfov_dict = {}

    for sequence in tqdm.tqdm(sequences):
        gt_bbox = []
        gt_rbbox = []
        gt_bfov = []
        gt_rbfov = []
        
        
        target_anno_file = os.path.join(dir, sequence, "label.json")
        with open(target_anno_file, "r") as output:
            target_annos = json.load(output)

        gt_bbox_absent = np.ones(len(target_annos))
        gt_rbbox_absent = np.ones(len(target_annos))
        gt_bfov_absent = np.ones(len(target_annos))
        gt_rbfov_absent = np.ones(len(target_annos))
        for i, index in enumerate(target_annos):
            annotation = target_annos[index]

            fovBbox = annotation["bfov"]
            fovBbox_rotated = annotation["rbfov"]
            bbox = annotation["bbox"]
            bbox_rotated = annotation["rbbox"]

            fovBbox = utils.dict2Bfov(fovBbox)
            fovBbox_rotated = utils.dict2Bfov(fovBbox_rotated)
            bbox = utils.dict2Bbox(bbox)
            bbox_rotated = utils.dict2Bbox(bbox_rotated)
            
            gt_bbox.append(bbox.tolist_xywh())
            gt_rbbox.append(bbox_rotated.tolist())
            gt_bfov.append(fovBbox.tolist())
            gt_rbfov.append(fovBbox_rotated.tolist())

            if bbox.w ==0 or bbox.h == 0:
                gt_bbox_absent[i] = 0

            if bbox_rotated.w ==0 or bbox_rotated.h == 0:
                gt_rbbox_absent[i] = 0

            if fovBbox.fov_h == 0 or fovBbox.fov_v == 0:
                gt_bfov_absent[i] = 0

            if fovBbox_rotated.fov_h == 0 or fovBbox_rotated.fov_v == 0:
                gt_rbfov_absent[i] = 0
                


        gt_bbox_dict.update({sequence: (np.array(gt_bbox), gt_bbox_absent)})
        gt_rbbox_dict.update({sequence: (np.array(gt_rbbox), gt_rbbox_absent)})
        gt_bfov_dict.update({sequence: (np.array(gt_bfov), gt_bfov_absent)})
        gt_rbfov_dict.update({sequence: (np.array(gt_rbfov), gt_rbfov_absent)})
    
    return sequences, gt_bbox_dict, gt_rbbox_dict, gt_bfov_dict, gt_rbfov_dict

def loadResult(dir, sequences):
    trackers = sorted(os.listdir(dir))
    total_results ={}
    for name in tqdm.tqdm(trackers):
        result = {}
        for sequence in sequences:
            traj_file = os.path.join(dir, name, sequence+'.txt')
            assert os.path.exists(traj_file)
            with open(traj_file, 'r') as f :
                pred_traj = []
                for x in f.readlines():
                    anno = x.strip().split(',')             
                    if len(anno)<2:
                        anno = x.strip().split()  #re.split('; |, ', x)
                    #print(bbox)
                    pred_traj.append(list(map(float, anno)))
            result.update({sequence: np.array(pred_traj)})
        
        total_results.update({name: result})
    
    return trackers, total_results

def eval_bbox(args, sequences, gt_bbox_dict, attribute_sequence, attribute_data, attribute_name):
    trackers_bbox, bbox_results = loadResult(args.bbox_dir, sequences)
    pbar = tqdm.tqdm(total=len(trackers_bbox))
    pbar_update = lambda *args: pbar.update()

    success_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bbox:
            eval_result = pool.apply_async(func=eval_success, args=(gt_bbox_dict, eval_tracker, bbox_results, ), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        success_ret.update(eval_result.get())
    
    pbar.reset()
    precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bbox:
            eval_result = pool.apply_async(func=eval_precision, args=(gt_bbox_dict, eval_tracker, bbox_results, ), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        precision_ret.update(eval_result.get())
    
    pbar.reset()
    norm_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bbox:
            eval_result = pool.apply_async(func=eval_norm_precision, args=(gt_bbox_dict, eval_tracker, bbox_results, ), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        norm_precision_ret.update(eval_result.get())

    pbar.reset()
    angle_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bbox:
            eval_result = pool.apply_async(func=eval_angle_precision, args=(gt_bbox_dict, eval_tracker, bbox_results, ), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        angle_precision_ret.update(eval_result.get())

    print("bbox result")
    show_result(success_ret, precision_ret, norm_precision_ret, angle_precision_ret, show_video_level=args.show_video_level, postfix="BBox")
    if args.plot_curve:
        plt_sp.draw_success_precision(success_ret, "360VOT", sequences, attr="ALL", precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                                    angle_precision_ret=angle_precision_ret, path=args.save_path)

    if attribute_sequence:
        # too much figure, prefer save rather than preview
        if args.save_path is None:
            args.save_path = "./bbox_result_fig_per_attribute"
            os.makedirs(args.save_path, exist_ok=True)
        for name in attribute_name:
            col = attribute_data.loc[:, name].to_numpy()
            vaild = np.where(col==1)[0]
            eval_sequences = [attribute_sequence[i] for i in vaild]
            plt_sp.draw_success_precision(success_ret, "360VOT", eval_sequences, attr=name.split("(")[0], precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                                            angle_precision_ret=angle_precision_ret, path=args.save_path, vis=args.plot_curve)

def eval_rbbox(args, sequences, gt_rbbox_dict, attribute_sequence, attribute_data, attribute_name):
    trackers_rbbox, rbbox_results = loadResult(args.rbbox_dir, sequences)
    pbar = tqdm.tqdm(total=len(trackers_rbbox))
    pbar_update = lambda *args: pbar.update()

    success_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbbox:
            eval_result = pool.apply_async(func=eval_success, args=(gt_rbbox_dict, eval_tracker, rbbox_results, ), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        success_ret.update(eval_result.get())


    precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbbox:
            eval_result = pool.apply_async(func=eval_precision, args=(gt_rbbox_dict, eval_tracker, rbbox_results, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        precision_ret.update(eval_result.get())
    
    pbar.reset()
    norm_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbbox:
            eval_result = pool.apply_async(func=eval_norm_precision, args=(gt_rbbox_dict, eval_tracker, rbbox_results, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        norm_precision_ret.update(eval_result.get())
    
    pbar.reset()
    angle_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbbox:
            eval_result = pool.apply_async(func=eval_angle_precision, args=(gt_rbbox_dict, eval_tracker, rbbox_results, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        angle_precision_ret.update(eval_result.get())


    print("rotated bbox result")
    show_result(success_ret, precision_ret, norm_precision_ret, angle_precision_ret, show_video_level=args.show_video_level, postfix="rBBox")
    if args.plot_curve:
        plt_sp.draw_success_precision(success_ret, "360VOT", sequences, attr="ALL", precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                                    angle_precision_ret=angle_precision_ret, path=args.save_path)

    if attribute_sequence:
        # too much figure, prefer save rather than preview
        if args.save_path is None:
            args.save_path = "./rbbox_result_fig_per_attribute"
            os.makedirs(args.save_path, exist_ok=True)
        for name in attribute_name:
            col = attribute_data.loc[:, name].to_numpy()
            vaild = np.where(col==1)[0]
            eval_sequences = [attribute_sequence[i] for i in vaild]
            plt_sp.draw_success_precision(success_ret, "360VOT", eval_sequences, attr=name.split("(")[0], precision_ret=precision_ret, norm_precision_ret=norm_precision_ret, 
                                            angle_precision_ret=angle_precision_ret, path=args.save_path, vis=args.plot_curve)

def eval_bfov(args, sequences, gt_bfov_dict, attribute_sequence, attribute_data, attribute_name):
    trackers_bfov, bfov_results = loadResult(args.bfov_dir, sequences)

    pbar = tqdm.tqdm(total=len(trackers_bfov))
    pbar_update = lambda *args: pbar.update()
    # only estimate the success rate and angle precesion
    success_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bfov:
            eval_result = pool.apply_async(func=eval_success, args=(gt_bfov_dict, eval_tracker, bfov_results, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        success_ret.update(eval_result.get())

    pbar.reset()
    angle_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_bfov:
            eval_result = pool.apply_async(func=eval_angle_precision, args=(gt_bfov_dict, eval_tracker, bfov_results, True, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        angle_precision_ret.update(eval_result.get())
    
    print("bfov result")
    show_result(success_ret, angle_precision_ret=angle_precision_ret, show_video_level=args.show_video_level, given_sphere=True, postfix="BFoV")
    
    if args.plot_curve:
        plt_sp.draw_success_precision(success_ret, "360VOT", sequences, attr="ALL", angle_precision_ret=angle_precision_ret, path=args.save_path)

    if attribute_sequence:
        # too much figure, prefer save rather than preview
        if args.save_path is None:
            args.save_path = "./bfov_result_fig_per_attribute"
            os.makedirs(args.save_path, exist_ok=True)
        for name in attribute_name:
            col = attribute_data.loc[:, name].to_numpy()
            vaild = np.where(col==1)[0]
            eval_sequences = [attribute_sequence[i] for i in vaild]
            plt_sp.draw_success_precision(success_ret, "360VOT", eval_sequences, attr=name.split("(")[0], 
                                            angle_precision_ret=angle_precision_ret, path=args.save_path, vis=args.plot_curve)

def eval_rbfov(args, sequences, gt_rbfov_dict, attribute_sequence, attribute_data, attribute_name):
    trackers_rbfov, rbfov_results = loadResult(args.rbfov_dir, sequences)
    
    pbar = tqdm.tqdm(total=len(trackers_rbfov))
    pbar_update = lambda *args: pbar.update()
    # only estimate the success rate and angle precesion
    success_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbfov:
            eval_result = pool.apply_async(func=eval_success, args=(gt_rbfov_dict, eval_tracker, rbfov_results, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        success_ret.update(eval_result.get())

    pbar.reset()
    angle_precision_ret = {}
    eval_results = []
    with Pool(processes=20) as pool:
        for eval_tracker in trackers_rbfov:
            eval_result = pool.apply_async(func=eval_angle_precision, args=(gt_rbfov_dict, eval_tracker, rbfov_results, True, True), callback=pbar_update)
            eval_results.append(eval_result)
        pool.close()
        pool.join()
    for eval_result in eval_results:
        angle_precision_ret.update(eval_result.get())
    print("rotated bfov result")
    show_result(success_ret, angle_precision_ret=angle_precision_ret, show_video_level=args.show_video_level, given_sphere=True, postfix="rBFoV")
    
    if args.plot_curve:
        plt_sp.draw_success_precision(success_ret, "360VOT", sequences, attr="ALL", angle_precision_ret=angle_precision_ret, path=args.save_path)

    if attribute_sequence:
        # too much figure, prefer save rather than preview
        if args.save_path is None:
            args.save_path = "./rbfov_result_fig_per_attribute"
            os.makedirs(args.save_path, exist_ok=True)
        for name in attribute_name:
            col = attribute_data.loc[:, name].to_numpy()
            vaild = np.where(col==1)[0]
            eval_sequences = [attribute_sequence[i] for i in vaild]
            plt_sp.draw_success_precision(success_ret, "360VOT", eval_sequences, attr=name.split("(")[0], 
                                            angle_precision_ret=angle_precision_ret, path=args.save_path, vis=args.plot_curve)


def main():
    args = opt()

    print("# loading dataset")
    sequences, gt_bbox_dict, gt_rbbox_dict, gt_bfov_dict, gt_rbfov_dict = loadDataset(args.dataset_dir)
    attribute_name = ["IV(illumination variation)", "BC(background clutter)", "DEF(deformable target)", "MB(motion blur)",
    	            "SA(stitching artifact)", "ROT(rotation)", "FM(fast motion)", "LR(low resolution)", "HR(high resolution)", 
                    "ARC(aspect ratio change)", "SV(scale variation)", "FOC(full occlusion)", "POC(partial occlusion)", 
                    "FMS(fast motion on the sphere)", "CB(cross border)", "HL(high latitude)", "LV(latitude variant)", "LFoV(large FoV)", "CM(camera motion)", "LD(large distortion)"]

    attribute_data = []
    attribute_sequence = []
    if args.attribute:
        if os.path.isfile(args.attribute):
            attribute_data = pd.read_excel(args.attribute)
            attribute_sequence = attribute_data.loc[:, "sequence"].values.tolist()
            attribute_sequence = ["{:04d}".format(name) for name in attribute_sequence if name is not None]

    print("# evaluating results")
    
    if args.bbox_dir:
        eval_bbox(args, sequences, gt_bbox_dict, attribute_sequence, attribute_data, attribute_name)

    if args.rbbox_dir:
        eval_rbbox(args, sequences, gt_rbbox_dict, attribute_sequence, attribute_data, attribute_name)

    if args.bfov_dir:
        eval_bfov(args, sequences, gt_bfov_dict, attribute_sequence, attribute_data, attribute_name)

    if args.rbfov_dir:
        eval_rbfov(args, sequences, gt_rbfov_dict, attribute_sequence, attribute_data, attribute_name)


if __name__ == "__main__":
    main()
