import os
import cv2
import sys
import tqdm
import argparse
import numpy as np
import multiprocessing as mul

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, ".."))
sys.path.append(os.path.join(this_dir, "..", "lib"))
from lib.omni import OmniImage
from lib.utils import x1y1wh2bbox, Bfov, Bbox
from eval_360VOT import loadDataset, loadResult

COLOR_norm=((0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745))
COLOR = np.array(COLOR_norm)*255

def opt():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-d', '--dataset_dir', type=str, default="/homes/Dataset/360tracking/360VOT",help='dataset path')
    parser.add_argument('-p', '--path', type=str, default="",help='the path for video saving')

    parser.add_argument('-b', '--bbox_dir', type=str, default="",help='bbox result path')
    parser.add_argument('-rb', '--rbbox_dir', type=str, default="",help='rotated bbox result path')
    parser.add_argument('-f', '--bfov_dir', type=str, default="",help='bfov result path')
    parser.add_argument('-rf', '--rbfov_dir', type=str, default="",help='rotated bfov result path')
    parser.add_argument('-ss', '--show_sequence', type=str, default="",help='plot specific sequence, e.g., 0001')
    parser.add_argument('-t', '--plot_top', dest='plot_top', action='store_false')
    parser.set_defaults(plot_top=True)

    return parser.parse_args()

def plot_anno(args, vis_sequence, trackers, results_dict, gt_dict, fourcc, img_w, img_h, plot_top, omni, save_image=False):
    """
    Input
    vis_sequence:[str] the visualized sequence
    trackers: [list] the compared trackers
    results_dict: [dict] 
    gt_dict: [dict]
    """
    
    image_path = os.path.join(args.dataset_dir, vis_sequence, "image")
    images = sorted(os.listdir(image_path))
    save_path = os.path.join(args.path) 
    os.makedirs(save_path, exist_ok=True)
    if save_image:
        os.makedirs(os.path.join(save_path, vis_sequence), exist_ok=True)
    video_file = os.path.join(save_path, vis_sequence+".mp4")
    if os.path.isfile(video_file):
        return
    print(video_file)
    video_writer = cv2.VideoWriter(video_file, fourcc, 15.0, (img_w, img_h))


    anno_classes = results_dict.keys() if len(results_dict) > 0 else gt_dict.keys()
    text_position = 115 if plot_top else 1795
    rect_position = 0 if plot_top else 1680
    font_size = 20

    for i, image_name in enumerate(images):
        img = cv2.imread(os.path.join(image_path, image_name))
        
        cv2.rectangle(img, (0, rect_position), (3840, rect_position+150), (240,240,240), -1) #95 240
        cv2.putText(img=img, text="{:06d} GT: ".format(i), org=(20, text_position), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(111,223,224), thickness=5)
        gt_char_size = 30
        gt_cur_pos = 760
        for anno_class in anno_classes:
            gt_anno, gt_anno_absent  = gt_dict[anno_class][vis_sequence]
            if gt_anno_absent[i] < 1:
                continue
            if anno_class == "bbox":
                anno = x1y1wh2bbox(gt_anno[i])
                omni.plot_bbox(img, anno, color=(227,188,48), size=font_size)
                cv2.putText(img=img, text="BBox", org=(gt_cur_pos, text_position), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(227,188,48), thickness=7)
            elif anno_class == "rbbox":
                anno = Bbox(*gt_anno[i])
                omni.plot_bbox(img, anno, color=(167,224,121), size=font_size)
                cv2.putText(img=img, text="rBBox", org=(gt_cur_pos, text_position), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(167,224,121), thickness=7)
            elif anno_class == "bfov" :
                anno = Bfov(*gt_anno[i])
                omni.plot_bfov(img, anno, color=(101,161,224), size=font_size)
                cv2.putText(img=img, text="BFoV", org=(gt_cur_pos, text_position), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(101,161,224), thickness=7)
            elif anno_class == "rbfov":
                anno = Bfov(*gt_anno[i])
                omni.plot_bfov(img, anno, color=(96,107,224), size=font_size)
                cv2.putText(img=img, text="rBFoV", org=(gt_cur_pos, text_position), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(96,107,224), thickness=7)
            
            gt_cur_pos += 400

            #start_point = [40, 490, 1190, 1545, 2120, 2450, 3040, 3520]
            if len(trackers) > 0:
                cv2.rectangle(img, (0, rect_position+150), (3840, rect_position+150+80), (240,240,240), -1) 
            char_size = 55
            cur_pos = 10

            for j, tracker in enumerate(trackers):
                if j > 8:
                    continue
                cv2.putText(img=img, text=tracker.upper(), org=(cur_pos, text_position+95), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=COLOR[j%20], thickness=5)
        
                anno_pred = results_dict[anno_class][tracker][vis_sequence][i]
                if anno_class == "bbox":
                    bbox = x1y1wh2bbox(anno_pred)
                    omni.plot_bbox(img, bbox, color=COLOR[j%20], size=font_size)
                elif anno_class == "rbbox":
                    rbbox = Bbox(*anno_pred)
                    omni.plot_bbox(img, rbbox, color=COLOR[j%20], size=font_size)
                elif anno_class == "bfov" :
                    anno = Bfov(*anno_pred)
                    omni.plot_bfov(img, anno, color=COLOR[j%20], size=font_size)
                elif anno_class == "rbfov":
                    anno = Bfov(*anno_pred)
                    omni.plot_bfov(img, anno, color=COLOR[j%20], size=font_size)
                cur_pos += char_size*(len(tracker)+1) 

        if save_image:
            img = cv2.resize(img, (960, 480))
            cv2.imwrite(os.path.join(save_path, image_name), img)
            #exit()
            #cv2.imshow("test", img)
            #cv2.waitKey(1)
        video_writer.write(img)
    video_writer.release()



def main():
    args = opt()
    print("# loading dataset")
    sequences, gt_bbox_dict, gt_rbbox_dict, gt_bfov_dict, gt_rbfov_dict = loadDataset(args.dataset_dir)
    vis_sequences = [args.show_sequence] if args.show_sequence else sequences
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    img_w = 3840
    img_h = 1920
    omni = OmniImage(img_w, img_h)
    plot_top=args.plot_top

    pool_count = len(vis_sequences) if len(vis_sequences) < 20 else 20
    processingPool = mul.Pool(pool_count)
    pbar = tqdm.tqdm(total=len(vis_sequences))
    pbar_update = lambda *args: pbar.update()

    gt_dict={"bbox": gt_bbox_dict,
             "rbbox": gt_rbbox_dict,
             "bfov": gt_bfov_dict,
             "rbfov": gt_rbfov_dict
    }
    results_dict={}
    trackers = []

    if args.bbox_dir:
        trackers, results = loadResult(args.bbox_dir, sequences)
        results_dict={"bbox": results}
    
    if args.rbbox_dir:
        trackers, results = loadResult(args.rbbox_dir, sequences)
        results_dict={"rbbox": results}
    
    if args.bfov_dir:
        trackers, results = loadResult(args.bfov_dir, sequences)
        results_dict={"bfov": results}
    
    if args.rbfov_dir:
        trackers, results = loadResult(args.rbfov_dir, sequences)
        results_dict={"rbfov": results}
    
    print("start processing")
    for vis_sequence in vis_sequences:
        processingPool.apply_async(func=plot_anno, args=(args, vis_sequence, trackers, results_dict, 
                                                             gt_dict, fourcc, img_w, img_h, plot_top, 
                                                             omni, False, ), callback=pbar_update)
        #plot_anno(args, vis_sequence, trackers, results_dict, gt_dict, fourcc, img_w, img_h, plot_top, omni, save_image=False)

    processingPool.close()
    processingPool.join()
    processingPool.terminate()

if __name__ == "__main__":
    main()