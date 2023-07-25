import numpy as np
from sphiou import SphIoU

def overlap_ratio(rect1, rect2, rotated=False):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
        rect:2d array of N x [x,y,w,h, r]
    Return:
        iou
    '''
    if rotated:
        from Rotated_IoU.oriented_iou_loss import cal_iou
        import torch

        assert rect1.shape[-1] == rect2.shape[-1] == 5, "wrong rotated bbox"
        rect1 = torch.from_numpy(rect1).unsqueeze(0).to(torch.float32).cuda()
        rect2 = torch.from_numpy(rect2).unsqueeze(0).to(torch.float32).cuda()

        #print(rect1.shape, rect1.dtype)
        iou, _, _, _ = cal_iou(rect1, rect2) 
        iou = iou.detach().cpu().squeeze().numpy()
        iou = np.maximum(np.minimum(1, iou), 0)
        #print(iou)
    else:
        left = np.maximum(rect1[:,0], rect2[:,0])
        right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
        top = np.maximum(rect1[:,1], rect2[:,1])
        bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

        intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
        union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
        iou = intersect / union
        iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def overlap_ratio_sphere(bfov1, bfov2):
    #print(bfov1.shape)

    bfov1 = bfov1 / 180 * np.pi
    bfov2 = bfov2 / 180 * np.pi
    #print(bfov1.shape)
    sphIoU = SphIoU().IOU(bfov1, bfov2)
    #print(bfov2.shape, sphIoU.shape);exit()
    return sphIoU

def dual_success_overlap(gt_bb, result_bb, n_frame, img_w):
    rotated = True if gt_bb.shape[-1] == 5 else False

    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    left_iou = np.ones(len(gt_bb)) * (-1)
    right_iou = np.ones(len(gt_bb)) * (-1)

    mask = np.sum(gt_bb[:, 2:4] > 0, axis=1) == 2
    left_gt_bbox = gt_bb.copy()
    right_gt_bbox = gt_bb.copy()
    left_gt_bbox[:, 0] -= img_w
    right_gt_bbox[:, 0] += img_w

    left_iou[mask] = overlap_ratio(left_gt_bbox[mask], result_bb[mask], rotated)
    right_iou[mask] = overlap_ratio(left_gt_bbox[mask], result_bb[mask], rotated)
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask], rotated)

    iou = np.max(np.column_stack([left_iou, iou, right_iou]), axis=1)
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def success_overlap(gt_bb, result_bb, n_frame, sphIoU=False):
    rotated = True if gt_bb.shape[-1] == 5 else False
        
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    # mask = np.sum(gt_bb > 0, axis=1) == 4 #TODO check all dataset
    mask = np.sum(gt_bb[:, 2:4] > 0, axis=1) == 2
    if sphIoU:
        iou[mask] = overlap_ratio_sphere(gt_bb[mask], result_bb[mask])
    else:
        iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask], rotated)

    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def eval_success(gt, eval_trackers, results, sphIoU=False, img_w=3840):
    """
    Args: 
        eval_trackers: list of tracker name or single tracker name
        sphereIou: have to be true for bfov and rbofv
        img_w: image width of 360 image
    Return:
        res: dict of results
    """
    if isinstance(eval_trackers, str):
        eval_trackers = [eval_trackers]

    success_ret = {}
    for tracker_name in eval_trackers:
        success_ret_ = {}
        result = results[tracker_name]
        for sequence_name in result:
            gt_traj, gt_vaild = gt[sequence_name]
            tracker_traj = result[sequence_name]
            n_frame = len(gt_traj)
            gt_traj = gt_traj[np.where(gt_vaild == 1)]
            tracker_traj = tracker_traj[np.where(gt_vaild == 1)]
            if sphIoU:
                success_ret_[sequence_name] = success_overlap(gt_traj, tracker_traj, n_frame, sphIoU)
            else:
                success_ret_[sequence_name] = dual_success_overlap(gt_traj, tracker_traj, n_frame, img_w)
        success_ret[tracker_name] = success_ret_
    return success_ret


def dual_success_error(gt_center, result_center, thresholds, n_frame, img_w=3840):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    dist_left = np.ones(len(gt_center)) * (-1)
    dist_right = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    left_gt_bbox = gt_center.copy()
    right_gt_bbox = gt_center.copy()
    left_gt_bbox[:, 0] -= img_w
    right_gt_bbox[:, 0] += img_w

    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    dist_left[mask] = np.sqrt(np.sum(
        np.power(left_gt_bbox[mask] - result_center[mask], 2), axis=1))
    dist_right[mask] = np.sqrt(np.sum(
        np.power(right_gt_bbox[mask] - result_center[mask], 2), axis=1))
    
    dist = np.min(np.column_stack([dist_left, dist, dist_right]), axis=1)
    for i in range(len(thresholds)):
        success[i] = np.sum(dist <= thresholds[i]) / float(n_frame)
    return success

def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                        (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

def eval_precision(gt, eval_trackers, results, given_center=False):
    """
    for evaluation of bbox and rbbox. bbox[x1, y1, w, h], rbbox[cx, cy, w, h, r]
    Args:
        gt: dict
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """
    if isinstance(eval_trackers, str):
        eval_trackers = [eval_trackers]
    thresholds = np.arange(0, 51, 1)
    precision_ret = {}
    for tracker_name in eval_trackers:
        precision_ret_ = {}
        result = results[tracker_name]
        for sequence_name in result:
            gt_traj, gt_vaild = gt[sequence_name]
            tracker_traj = result[sequence_name]
            n_frame = len(gt_traj)
            gt_traj = gt_traj[np.where(gt_vaild == 1)]
            tracker_traj = tracker_traj[np.where(gt_vaild == 1)]
            if given_center:
                assert tracker_traj.shape[-1] == 5, "make sure you are evaluating rbbox in [cx, cy, w, h, r]"
                gt_center = gt_traj[:, :2]
                tracker_center = tracker_traj[:, :2]
            else:
                assert tracker_traj.shape[-1] == 4, "make sure you are evaluating bbox in [x1, y1, w, h]"
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)
            precision_ret_[sequence_name] = dual_success_error(gt_center, tracker_center, thresholds, n_frame)

        precision_ret[tracker_name] = precision_ret_
    return precision_ret


def convert_bb_to_norm_center(bboxes, gt_wh):
    return convert_bb_to_center(bboxes) / (gt_wh+1e-16)

def eval_norm_precision(gt, eval_trackers, results, given_center=False):
    """
    for evaluation of bbox and rbbox. bbox[x1, y1, w, h], rbbox[cx, cy, w, h, r]
    Args:
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """

    if isinstance(eval_trackers, str):
        eval_trackers = [eval_trackers]

    norm_precision_ret = {}
    for tracker_name in eval_trackers:
        norm_precision_ret_ = {}
        result = results[tracker_name]
        for sequence_name in result:
            gt_traj, gt_vaild = gt[sequence_name]
            tracker_traj = result[sequence_name]
            n_frame = len(gt_traj)

            gt_traj = gt_traj[np.where(gt_vaild == 1)]
            tracker_traj = tracker_traj[np.where(gt_vaild == 1)]
            if given_center:
                assert tracker_traj.shape[-1] == 5, "make sure you are evaluating rbbox in [cx, cy, w, h, r]"
                gt_center_norm = gt_traj[:, :2]/(gt_traj[:, 2:4]+1e-16)
                tracker_center_norm = tracker_traj[:, :2]/(gt_traj[:, 2:4]+1e-16)
            else:
                gt_center_norm = convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])

            thresholds = np.arange(0, 51, 1) / 100
            norm_precision_ret_[sequence_name] = dual_success_error(gt_center_norm, tracker_center_norm, thresholds, n_frame)

        norm_precision_ret[tracker_name] = norm_precision_ret_
    return norm_precision_ret


def success_angle_error(gt_center, result_center, thresholds, n_frame):
    success = np.zeros(len(thresholds))
    #dist = np.ones(len(gt_center)) * (-1)

    gt_center = gt_center / np.linalg.norm(gt_center)
    result_center = result_center / np.linalg.norm(result_center)

    ab = np.sum(np.multiply(gt_center, result_center), axis=1)
    a = np.sqrt(np.sum(np.power(gt_center, 2), axis=1))
    b = np.sqrt(np.sum(np.power(result_center, 2), axis=1))
    cos_ab = ab / np.multiply(a, b) 

    cos_ab[np.where(cos_ab > 1)] = 1
    angle_ab = np.arccos(cos_ab) / np.pi * 180
    for i in range(len(thresholds)):
        success[i] = np.sum(angle_ab <= thresholds[i]) / float(n_frame)
    return success

def convert_uv_to_xyz(uv, img_w, img_h):
    lon, lat = uv2lonlat(uv[:, 0], uv[:, 1], img_w, img_h)
    #x, y, z = lonlat2xyz(lon, lat)
    return lonlat2xyz(lon, lat)

def covert_lonlat_to_xyz(lonlat):
    lonlat = lonlat / 180 * np.pi
    return lonlat2xyz(lonlat[:, 0], lonlat[:, 1])

def uv2lonlat(u, v, img_w, img_h):
    lon = ((u + 0.5) / img_w - 0.5) * 2 * np.pi
    lat = -((v + 0.5) / img_h - 0.5) * np.pi
    return lon, lat

def lonlat2xyz(lon, lat):
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(-lat)
    z = np.cos(lat) * np.cos(lon)
    return np.column_stack([x, y, z])

def eval_angle_precision(gt, eval_trackers, results, given_center=False, give_sphere=False, img_w=3840, img_h=1920):
    """
    given_center: for bbox[x1, y1, w, h], should be false. 
    give_sphere: when evaluate bfov and rbfov, should be true
    Args:
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """

    if isinstance(eval_trackers, str):
        eval_trackers = [eval_trackers]

    precision_ret = {}
    for tracker_name in eval_trackers:
        precision_ret_ = {}
        result = results[tracker_name]
        for sequence_name in result:
            gt_traj, gt_vaild = gt[sequence_name]
            tracker_traj = result[sequence_name]
            n_frame = len(gt_traj)

            gt_traj = gt_traj[np.where(gt_vaild == 1)]
            tracker_traj = tracker_traj[np.where(gt_vaild == 1)]

            if given_center:
                gt_center = gt_traj[:, :2]
                tracker_center = tracker_traj[:, :2]
            else:
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)

            if give_sphere:
                gt_center = covert_lonlat_to_xyz(gt_center)
                tracker_center = covert_lonlat_to_xyz(tracker_center)
            else:
                gt_center = convert_uv_to_xyz(gt_center, img_w, img_h)
                tracker_center = convert_uv_to_xyz(tracker_center, img_w, img_h)

            thresholds = np.arange(0, 101, 1)/10
            precision_ret_[sequence_name] = success_angle_error(gt_center, tracker_center, thresholds, n_frame)
        precision_ret[tracker_name] = precision_ret_
    return precision_ret


def show_result(success_ret, precision_ret=None, norm_precision_ret=None, angle_precision_ret=None,
                 show_video_level=False, helight_threshold=0.6, given_sphere=False, postfix=""):
    """pretty print result
    Args:
        result: returned dict from function eval
    """
    # sort tracker
    tracker_auc = {}
    for tracker_name in success_ret.keys():
        auc = np.mean(list(success_ret[tracker_name].values()))
        tracker_auc[tracker_name] = auc
    tracker_auc_ = sorted(tracker_auc.items(),
                            key=lambda x:x[1],
                            reverse=True)[:40]
    tracker_names = [x[0] for x in tracker_auc_]
    from rich.console import Console
    from rich.table import Table
    table = Table(title="360VOT Metrics in "+ postfix)
    if given_sphere:
        table.add_column("Tracker", justify="right", no_wrap=True)
        table.add_column("S_sphere", justify="center", style="cyan")
        table.add_column("P_angle", justify="center", style="magenta")
    else:
        table.add_column("Tracker", justify="right", no_wrap=True)
        table.add_column("S_dual", justify="center", style="cyan")
        table.add_column("P_dual", justify="center", style="magenta")
        table.add_column("N_P_dual", justify="center", style="cyan")
        table.add_column("P_angle", justify="center", style="magenta")

    for tracker_name in tracker_names:
        # success = np.mean(list(success_ret[tracker_name].values()))
        success = tracker_auc[tracker_name]
        if precision_ret is not None:
            precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20] # sequence, precesion_under_defferent thr
        else:
            precision = 0
        if norm_precision_ret is not None:
            #print(np.mean(list(norm_precision_ret[tracker_name].values()), axis=0))
            norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()))
        else:
            norm_precision = 0

        if angle_precision_ret is not None:
            angle_procision = np.mean(list(angle_precision_ret[tracker_name].values()),
                    axis=0)[30]
        else:
            angle_procision = 0

        if given_sphere:
            table.add_row(tracker_name, "{:0.3f}".format(success), "{:0.3f}".format(angle_procision))
        else:
            table.add_row(tracker_name, "{:0.3f}".format(success), \
                        "{:0.3f}".format(precision),"{:0.3f}".format(norm_precision),
                        "{:0.3f}".format(angle_procision))

    console = Console()
    console.print(table)
    
    if show_video_level:
        maximum = len(success_ret) if len(success_ret) < 10 else 10
        table2 = Table(title="360VOT Metrics Per Sequence in "+ postfix,padding=(0,0))
        tables = [] 
        table2.add_column("Seq", justify="center")
        for tracker_name in tracker_names[:maximum]:
            table2.add_column(tracker_name, justify="center")

        for i in range(maximum):
            tb = Table(padding=(0,0), show_edge=False, show_header=False)
            tb.add_column("S", justify="center")
            tb.add_column("P", justify="center")
            if given_sphere:
                tb.add_row("S_sph", "P_ang")
            else:
                tb.add_row("S_dua", "P_ang")
            tables.append(tb)
        table2.add_row("", *tables)

        videos = list(success_ret[tracker_name].keys())
        for video in videos:
            tables=[]
            for tracker_name in tracker_names[:maximum]:
                success = np.mean(success_ret[tracker_name][video])
                precision = np.mean(angle_precision_ret[tracker_name][video])
                success_str = "{:.3f}".format(success)
                precision_str = "{:.3f}".format(precision)
                if success > helight_threshold:
                    success_str = "[bold blue]"+success_str
                if precision > helight_threshold:
                    precision_str = "[bold red]"+precision_str
                tb = Table(padding=(0,0), show_edge=False, show_header=False)
                tb.add_column("S", justify="center", style="cyan")
                tb.add_column("P", justify="center", style="magenta")
                tb.add_row(success_str, precision_str)
                tables.append(tb)
            
            table2.add_row(video, *tables)
        console.print(table2)
