import argparse
import json
import shutil
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
#from models.yolo import *
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix, fitness, ap_per_class_pred_unique
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

def create_eval_key(all_info, key, classes):
    all_info[key] = dict()
    all_info[key]['jdict'] = []
    all_info[key]['stats'] = []
    all_info[key]['ap'] = []
    all_info[key]['ap_class'] = []
    all_info[key]['wandb_images'] = []
    all_info[key]['classes'] = classes
    all_info[key]['classes_set'] = set(classes)

def load_model(weights, hyp, device='cuda', nc=80):
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), hyp=hyp).cuda()
    #exclude = ['anchor'] if (hyp.get('anchors')) else []  # exclude keys
    exclude = []
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    return model


def training_fitness(all_info):
    """
    Default: all_info['val_names']['mp'], all_info['val_names']['mr'], all_info['val_names']['map50'], all_info['val_names']['map']
    """
    return (all_info['val_names']['mp'], all_info['val_names']['mr'], all_info['val_names']['map50'], all_info['val_names']['map'])

def process_recall(zsd_recall_correct, zsd_recall_classes, nt):
    zsd_recall_correct, zsd_recall_classes = torch.stack(zsd_recall_correct), torch.Tensor(zsd_recall_classes).type(torch.LongTensor)
    classes_split = {}
    for i in zsd_recall_classes.unique():
        classes_split[int(i)] = []
    for idx, c in enumerate(zsd_recall_classes):
        classes_split[int(c)].append(zsd_recall_correct[idx])
    classes_split = {k: torch.stack(v).sum(dim=0) / nt[k] for k, v in classes_split.items()}
    return classes_split

@torch.no_grad()
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         text_embedding_path=None,
         opt=None):
    print(batch_size)
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if opt.hyp:
            with open(opt.hyp, 'w') as f:
                hyp = yaml.safe_load(f)
            model = load_model(weights, hyp, map_location=device)
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if text_embedding_path:
        det = model.model[-1]
        det.text_embeddings = torch.load(text_embedding_path)
        det.update_nc_no()
    print(model.model[-1].text_embeddings.shape)
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    #nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = 1 if single_cls else model.model[-1].nc
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True, hyp={'do_zsd': opt.zsd},
                                       prefix=colorstr(f'{task}: '), annot_folder=opt.annot_folder)[0]
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    if single_cls:
        names = {0: 'object'}
    else:
        if opt.zsd:
            names = {i: data['all_names'][data['val_names'][i]] for i in range(len(data['val_names']))}
        else:
            names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    if opt.zsd:
        if any('_temp' in k for k in data.keys()) and (model.hyp['sim_func'] == 0) and opt.eval_splits != None:
            print('temp: ' + str(model.model[-1].sim_func.temp))
            prev_temp = model.model[-1].sim_func.temp
            new_temp = torch.nn.Parameter(torch.zeros(size=(model.model[-1].text_embeddings.shape[0], ), device=device) + prev_temp.repeat(model.model[-1].text_embeddings.shape[0]))
            for i in opt.eval_splits:
                k = i + '_temp'
                if k in data.keys():
                    new_temp[data[i]] = new_temp[data[i]] * data[k]
            model.model[-1].sim_func.temp = new_temp
            print(new_temp)
        if model.hyp['sim_func'] == 1:
            print('contrast: ' + str(model.model[-1].sim_func.cosine_scalar.contrast))
            print('shift: ' + str(model.model[-1].sim_func.cosine_scalar.shift))
        if opt.favor:
            det = model.model[-1]
            det.favor = torch.zeros(size=(len(data['val_names']), ))
            for i in data['unseen_names']:
                det.favor[i] = opt.favor
    all_info = {}
    create_eval_key(all_info, 'val_names', classes=data['val_names'])
    for i in opt.eval_splits:
        create_eval_key(all_info, i, classes=data[i])
    if opt.eval_by_splits:
        model.model[-1].sim_func.set_eval_groups([data[i] for i in opt.eval_splits])
        
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    if opt.zsd:
        loss = torch.zeros(7, device=device)
    else:
        loss = torch.zeros(3, device=device)
    zsd_recall_correct, zsd_recall_classes = [], []
    iour = torch.Tensor([0.4, 0.5, 0.6]).to(device)
    if opt.visualization_demo:
        if os.path.exists(save_dir / 'visualization_demo_labels'):
            shutil.rmtree(save_dir / 'visualization_demo_labels')
        if os.path.exists(save_dir / 'visualization_demo_pred'):
            shutil.rmtree(save_dir / 'visualization_demo_pred')
        os.mkdir(save_dir / 'visualization_demo_labels')
        os.mkdir(save_dir / 'visualization_demo_pred')
        stats_per_img = []
    if os.path.exists(save_dir / 'batch_plots_labels'):
        shutil.rmtree(save_dir / 'batch_plots_labels')
    if os.path.exists(save_dir / 'batch_plots_pred'):
        shutil.rmtree(save_dir / 'batch_plots_pred')
    os.mkdir(save_dir / 'batch_plots_labels')
    os.mkdir(save_dir / 'batch_plots_pred')
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if single_cls:
            targets[:, 1] = 0.0
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        with torch.no_grad():
            out, train_out = model(img, augment=augment)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            if opt.zsd:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:7]  # box, obj, cls, img, text, self_img, self_text
            else:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
        # Run NMS
        targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        if single_cls:
            out[:, :, 6:] = 0.0
            out[:, :, 5] = 1.0
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls or opt.agnostic_nms, zsd=opt.zsd and (not opt.no_zsd_post), obj_conf_thresh=opt.obj_conf_thresh, max_det=opt.max_det, eval_splits=[data[k] for k in opt.eval_splits], nms_then_zsd=opt.nms_then_zsd)
        t1 += time_synchronized() - t
        
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:6]
            nl = len(labels)
            tcls = labels[:, 0] if nl else torch.Tensor([])  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    for k, v in all_info.items():
                        cls_idx = torch.Tensor([i for i in range(len(tcls)) if tcls[i] in v['classes_set']]).type(torch.LongTensor)
                        v['stats'].append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls.cpu()))
                continue
        
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(all_info['val_names']['wandb_images']) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[int(cls)], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    all_info['val_names']['wandb_images'].append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    all_info['val_names']['jdict'].append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                total = 0
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    total += ti.shape[0]
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
                        
                        recall_detected = {}
                        for j in (ious > iour[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            overlap = ious[j] > iour
                            if (recall_detected.get(d.item()) == None) or (recall_detected.get(d.item()).sum() < overlap.sum()):
                                recall_detected[d.item()] = overlap
                            #if len(recall_detected) == nl:  # all targets already located in image
                            #    break
                        for v in recall_detected.values():
                            zsd_recall_correct.append(v)
                            zsd_recall_classes.append(cls)
            # Append statistics (correct, conf, pcls, tcls)
            all_info['val_names']['stats'].append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls.cpu()))
            if opt.visualization_demo:
                p = paths[si].split('/')[-1]
                vis_path = save_dir / 'visualization_demo_pred' / p
                label_path = save_dir / 'visualization_demo_labels' / p
                #print(targets[targets[:, 0] == si])
                #print(output_to_target(out[si:si + 1]).shape)
                Thread(target=plot_images, args=(img[si:si + 1],
                                                 torch.cat([torch.zeros(size=(nl, 1)).cuda(),
                                                            targets[targets[:, 0] == si, 1:6]], dim=1), paths[si: si + 1], 
                                                 label_path, names, 640, 1, opt.plot_conf), daemon=True).start()
                Thread(target=plot_images, args=(img[si:si + 1], output_to_target(out[si:si + 1]), paths[si: si + 1], vis_path, names, 640, 1, opt.plot_conf), daemon=True).start()
                img_stats = {'filename': p}
                precision, recall, ap, f1, ap_class = ap_per_class_pred_unique(correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), tcls.cpu().numpy(), plot=False, names=names)
                #torch.save((precision, recall, ap, f1, ap_class), 'debugging.pt')
                img_stats['mp'], img_stats['mr'], img_stats['map50'], img_stats['map'] = precision.mean(), recall.mean(), ap[:, 0].mean(), ap.mean()
                img_stats['fitness'] = correct.cpu().numpy().mean()
                img_stats['pred'] = pred.cpu().numpy()
                img_stats['labels'] = targets[targets[:, 0] == si, 1:6].cpu().numpy()
                stats_per_img.append(img_stats)
        # Plot images
        if (plots and batch_i < 3) or opt.visualization_demo:
            f = save_dir / f'batch_plots_labels/test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets[:, :6], paths, f, names, 640, 16, opt.plot_conf), daemon=True).start()
            f = save_dir / f'batch_plots_pred/test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names, 640, 16, opt.plot_conf), daemon=True).start()
        #for testing
        #if batch_i > 10:
        #    break
    # Compute statistics
    if opt.visualization_demo:
        torch.save(stats_per_img, save_dir / 'stats_per_img.pt')
    val_info = all_info['val_names']
    val_info['stats'] = [np.concatenate(x, 0) for x in zip(*val_info['stats'])]  # to numpy
    if len(val_info['stats']) and val_info['stats'][0].any():
        val_info['p'], val_info['r'], val_info['ap'], val_info['f1'], val_info['ap_class'] = ap_per_class(*val_info['stats'], plot=plots, save_dir=save_dir, names=names)
        val_info['ap50'], val_info['ap'] = val_info['ap'][:, 0], val_info['ap'].mean(1)  # AP@0.5, AP@0.5:0.95
        nt = np.bincount(val_info['stats'][3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        print('NO STATS')
    
    #remap classes
    mapping = {int(val_info['classes'][i]): i for i in range(len(val_info['ap_class']))}
    for v in all_info.values():
        v['classes_m'] = [mapping[i] for i in v['classes'] if i in mapping.keys()]
    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    recall_info = process_recall(zsd_recall_correct, zsd_recall_classes, nt)
    #torch.save(recall_info, save_dir / 'recall_info.pt')
    for k, v in all_info.items():
        v['mp'], v['mr'], v['map50'], v['map'] = val_info['p'][v['classes_m']].mean(), val_info['r'][v['classes_m']].mean(), val_info['ap50'][v['classes_m']].mean(), val_info['ap'][v['classes_m']].mean()
        print(pf % (k, seen, nt.sum(), v['mp'], v['mr'], v['map50'], v['map']))

        # Print results per class
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(v['stats']):
            for i, c in zip(v['classes'], v['classes_m']):
                print(pf % (str(i) + ' ' + data['all_names'][i], seen, nt[c], val_info['p'][c], val_info['r'][c], val_info['ap50'][c], val_info['ap'][c]))
            #create double whitespace lines
        print('\n')
    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    
    for k, v in all_info.items():
        info = torch.stack([recall_info.get(k) if k in recall_info.keys() else torch.zeros_like(iour) for k in v['classes_m']]).sum(dim=0) / len(v['classes_m'])
        print(f'Recall for {k}: {info}')
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if all_info['val_names']['wandb_images']:
        wandb_logger.log({"Bounding Box Debugger/Images": all_info['val_names']['wandb_images']})
    
    # Save JSON
    if save_json and len(all_info['val_names']['jdict']):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../datasets/coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(all_info['val_names']['jdict'], f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
    torch.save(opt, save_dir / 'opts.pt')
    torch.save(model, save_dir / 'model.pt')
    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + all_info['val_names']['map']
    for i, c in enumerate(all_info['val_names']['ap_class']):
        maps[c] = all_info['val_names']['ap'][i]
    if any('_temp' in k for k in data.keys()) and opt.eval_by_splits and opt.zsd and model.hyp['sim_func'] == 0:
        model.model[-1].sim_func.temp = prev_temp
    print(list(val_info['ap50']))
    return (*training_fitness(all_info), *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--annot-folder', type=str, default='labels', help='optional path to relative label folder')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--zsd', action='store_true', help='zsd evaluation')
    parser.add_argument('--text-embedding-path', default=None, type=str, help='option to override val_text_embeddings with given path to embeddings')
    parser.add_argument('--favor', default=None, type=float)
    parser.add_argument('--obj-conf-thresh', default=0.1, type=float, help='obj conf thresh for zsd type eval')
    parser.add_argument('--max-det', default=300, type=int, help='maximum number of detections per image')
    parser.add_argument('--eval-splits', default=[], type=str, nargs='+', help='keys for data class splits, eg: "seen_classes"')
    parser.add_argument('--hyp', default=None, type=str, help='path to hyperparameters, also will refresh model type')
    parser.add_argument('--plot-conf', default=0.25, type=float)
    parser.add_argument('--eval-by-splits', default=False, action='store_true')
    parser.add_argument('--agnostic-nms', default=False, action='store_true')
    parser.add_argument('--visualization-demo', default=False, action='store_true', help='creates file with stats per image and saves all detections')
    parser.add_argument('--no-zsd-post', default=False, action='store_true', help='performs regular postprocessing when performing zsd')
    parser.add_argument('--nms-then-zsd', default=False, action='store_true', help = 'performs zsd post-processing after nms')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights[0] if isinstance(opt.weights, list) else opt.weights, #fix later
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             text_embedding_path=opt.text_embedding_path,
             opt=opt
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
