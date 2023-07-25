import os
import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None, norm_precision_ret=None, angle_precision_ret=None, 
                           bold_name=[], axis=[0, 1], path=None, vis=True):
    # success plot
    #print("draw_success_precision")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if path is not None:
        print("saving ", os.path.join(path, "on {}".format(attr)))

    if attr == 'ALL':
        title = f'Success plots of OPE on {name}'
    else:
        title = f'Success plots of OPE - {attr}'
        #title = f"$\mathbf{Success plots of OPE - {attr}} $"
    plt.title(title, fontweight='bold')
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name in bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[idx%20], linestyle=LINE_STYLE[idx%20],label=label, linewidth=2)
    #ax.legend(loc='lower left', labelspacing=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    # ax.set_aspect(.6)
    if path is not None:
        plt.savefig(os.path.join(path, "Success rate on {}.jpg".format(attr)))
    if vis:
        plt.show()

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.grid(b=True)
        ax.set_aspect(1)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(f'Precision plots of OPE on {name}', fontweight='bold')
        else:
            plt.title(f'Precision plots of OPE - {attr}', fontweight='bold')
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name in bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx%20], linestyle=LINE_STYLE[idx%20],label=label, linewidth=2)
        #ax.legend(loc='lower right', labelspacing=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(0, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if path is not None:
            plt.savefig(os.path.join(path, "Precision rate on {}.jpg".format(attr)))

        if vis:
            plt.show()

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.grid(b=True)
        ax.set_aspect(1)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(f'Normalized Precision plots of OPE on {name}', fontweight='bold')
        else:
            plt.title(f'Normalized Precision plots of OPE - {attr}', fontweight='bold')
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value) #np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name in bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx%20], linestyle=LINE_STYLE[idx%20],label=label, linewidth=2)
        #ax.legend(loc='lower right', labelspacing=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(0, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if path is not None:
            plt.savefig(os.path.join(path, "Norm precision rate on {}.jpg".format(attr)))
        if vis:
            plt.show()

    if angle_precision_ret:
        # angle precision plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.grid(b=True)
        ax.set_aspect(1)
        plt.xlabel('Angle error threshold (degree)')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(f'Angle Precision plots of OPE on {name}', fontweight='bold')
        else:
            plt.title(f'Angle Precision plots of OPE - {attr}', fontweight='bold')
        plt.axis([0, 30]+axis)
        precision = {}
        thresholds = np.arange(0, 101, 1) / 10
        for tracker_name in angle_precision_ret.keys():
            value = [v for k, v in angle_precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[30]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name in bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in angle_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx%20], linestyle=LINE_STYLE[idx%20],label=label, linewidth=2)
        #ax.legend(loc='lower right', labelspacing=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 1))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if path is not None:
            plt.savefig(os.path.join(path, "Angle precision rate on {}.jpg".format(attr)))
        if vis:
            plt.show()
