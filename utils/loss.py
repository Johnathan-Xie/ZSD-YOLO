# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from models.yolo import *


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25, from_logits=False):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.from_logits = from_logits
       
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        if not self.from_logits:
            pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
    
def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm.float(), b_norm.transpose(0, 1).float())
    return sim_mt

class CosineDistanceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.cos_func = torch.nn.CosineSimilarity(eps=eps)
        
    def call(self, pred, target):
        return torch.sum(1 - (self.cos_func(pred, target)))
    
class VariableDistanceLoss(nn.Module):
    def __init__(self, diff_exp=1.0):
        super().__init__()
        self.diff_exp = diff_exp
    
    def __call__(self, pred, target):
        return torch.sum(torch.abs(pred - target) ** diff_exp)

class ZSDCrossEntropy(nn.Module):
    '''
    Utilizes SoftmaxSim
    '''
    def __init__(self, hyp, det):
        super().__init__()
        self.diff_exp = hyp['diff_exp']
        self.img_distill_weight = hyp['img_distill_weight']
        self.text_distill_weight = hyp['text_distill_weight']
        self.sim_func = det.sim_func
        self.self_img_loss_scalar = hyp['self_img_loss_scalar']
        self.self_text_loss_scalar = hyp['self_text_loss_scalar']
        if hyp['normalize'] > 0:
            self.normalizer = torch.load(hyp['normalizer_path']) ** hyp['normalize']
            self.normalizer = self.normalizer / self.normalizer.mean()
        self.bg = det.bg
       
    def __call__(self, cls_pred, img_targets=None, classes=None, text_embeddings=None):
        assert (img_targets is not None) or ((classes is not None) and (text_embeddings is not None))
        img_loss = 0.0
        text_loss = 0.0
        self_label_img_loss = 0.0
        self_label_text_loss = 0.0
        idx = classes != -1
        #Incorrect computation for img distill weight
        if (img_targets is not None) and (self.img_distill_weight > 0):
            if len(cls_pred[idx]):
                img_loss += torch.sum(torch.abs(cls_pred[idx] - img_targets[idx]) ** self.diff_exp)
            if self.self_img_loss_scalar and len(cls_pred[torch.logical_not(idx)]):
                self_label_img_loss = torch.sum(torch.abs(cls_pred[torch.logical_not(idx)] - img_targets[torch.logical_not(idx)]) ** self.diff_exp) * self.self_img_loss_scalar
                img_loss += self_label_img_loss
                self_label_img_loss = self_label_img_loss / cls_pred[torch.logical_not(idx)].numel()
            img_loss = img_loss / cls_pred.numel()
            #print(cls_pred.shape[0])
        if ((classes is not None)
            and (text_embeddings is not None)
            and (self.sim_func is not None)
            and (self.text_distill_weight > 0)):
            if self.self_text_loss_scalar:
                sim = self.sim_func(cls_pred, torch.cat([text_embeddings, self.bg.unsqueeze(0)], dim=0))
            else:
                sim = self.sim_func(cls_pred, text_embeddings)
            classes = classes.type(torch.cuda.LongTensor)
            classes[classes < 0] = text_embeddings.shape[0] - 1
            if len(classes[idx]) > 0:
                text_loss += torch.nn.functional.nll_loss(torch.log(sim[idx]), classes[idx])
            if self.self_text_loss_scalar and len(classes[torch.logical_not(idx)]):
                self_label_text_loss = torch.nn.functional.nll_loss(
                    torch.log(sim[torch.logical_not(idx)]), classes[torch.logical_not(idx)]
                ) * self.self_text_loss_scalar
                text_loss += self_label_text_loss
        return img_loss * self.img_distill_weight + text_loss * self.text_distill_weight, img_loss, text_loss, self_label_img_loss, self_label_text_loss

class ZSDBinaryCrossEntropy:
    '''
    Utilizes SigmoidSim
    '''
    def __init__(self, hyp, det, bce_func):
        super().__init__()
        device = det.m[0].weight.device
        self.diff_exp = hyp['diff_exp']
        self.img_distill_weight = hyp['img_distill_weight']
        self.text_distill_weight = hyp['text_distill_weight']
        self.sim_func = det.sim_func
        self.bce_func = bce_func
        self.normalizer = torch.load(hyp['normalizer_path']).to(device) ** hyp['normalize']
        self.normalizer = self.normalizer / self.normalizer.mean()
        self.self_img_loss_scalar = hyp['self_img_loss_scalar']
        self.self_text_loss_scalar = hyp['self_text_loss_scalar']
    def __call__(self, cls_pred, img_targets=None, classes=None, text_embeddings=None):
        assert (img_targets is not None) or ((classes is not None) and (text_embeddings is not None))
        img_loss = 0.0
        text_loss = 0.0
        self_label_img_loss = 0.0
        self_label_text_loss = 0.0
        idx = classes != -1
       
        if (img_targets is not None) and (self.img_distill_weight > 0):
            if len(cls_pred[idx]):
                img_loss += torch.mean(torch.abs(cls_pred[idx] - img_targets[idx]) ** self.diff_exp) * self.img_distill_weight
            if self.self_img_loss_scalar and len(cls_pred[torch.logical_not(idx)]):
                self_label_img_loss = torch.mean(torch.abs(cls_pred[torch.logical_not(idx)] - img_targets[torch.logical_not(idx)]) ** self.diff_exp) * self.img_distill_weight * self.self_img_loss_scalar
                img_loss += self_label_img_loss
       
        if all([i is not None for i in [classes, text_embeddings, self.sim_func, self.bce_func]]):
            sim = self.sim_func(cls_pred, text_embeddings)
            classes = classes.type(torch.cuda.LongTensor)
           
            if len(classes[torch.logical_not(idx)]):
                self_text_targets = torch.zeros_like(sim[torch.logical_not(idx)])
                self_text_loss = self.bce_func(sim[torch.logical_not(idx)], self_text_targets).mean()
                text_loss += self_text_loss * self.self_text_loss_scalar
                self_label_text_loss += self_text_loss * self.self_text_loss_scalar
           
            if len(classes[idx]) > 0:
                text_targets = torch.zeros_like(sim[idx])
                for i in range(len(text_targets)):
                    text_targets[i][classes[idx][i]] = 1.0
                text_loss += (self.bce_func(sim[idx], text_targets) * self.normalizer).mean()
           
        return img_loss + text_loss, img_loss, text_loss, self_label_img_loss, self_label_text_loss
    
class ComputeZSDLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeZSDLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.model = model
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        #self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
       
        # Focal loss
        #g = h['fl_gamma']  # focal loss gamma
        #if g > 0:
        #    BCEobj, BCEcls = FocalLoss(BCEobj, g), FocalLoss(BCEcls, g)
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # ZSDDetect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEobj, self.gr, self.hyp, self.autobalance = BCEobj, model.gr, h, autobalance
        if isinstance(det.sim_func, SigmoidSim):
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
            self.cls_loss = ZSDBinaryCrossEntropy(h, det, BCEcls)
        elif isinstance(det.sim_func, SoftmaxSim):
            self.cls_loss = ZSDCrossEntropy(h, det)
            if h['learnable_background']:
                self.cls_loss.bg = det.bg
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, limg, ltext, lself_img, lself_text, lbox, lobj = (
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        lcls_items = [lcls, limg, ltext, lself_img, lself_text]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    lcls_out = self.cls_loss(ps[:, 5:], tcls[i][:, 1:], tcls[i][:, 0],
                                        self.model.model[-1].text_embeddings)
                    for j in range(len(lcls_items)):
                        lcls_items[j] += lcls_out[j]
                   
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, limg, ltext, lself_img, lself_text, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        #targets, embeddings = targets[:, :6], targets[:, 6:]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, embeddings = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets[:, :6].repeat(na, 1, 1), ai[:, :, None], targets[:, 6:].repeat(na, 1, 1)), 2)  # append anchor indices
       
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = torch.cat((targets[:, :, :7] * gain, targets[:, :, 7:]), dim=2)
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors'
            tcls.append(torch.cat([c.unsqueeze(-1), t[:, 7:]], dim=1))  # class
        return tcls, tbox, indices, anch
'''
    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, limg, ltext, lself_img, lself_text, lbox, lobj = (
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        lcls_items = [lcls, limg, ltext, lself_img, lself_text]
        tcls, full_indices, tbox, non_self_indices, anchors = self.build_targets(p, targets)  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = full_indices[i]  # image, anchor, gridy, gridx
            nb, na, ngj, ngi = non_self_indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                nps = pi[nb, na, ngj, ngi]
                # Regression
                pxy = nps[:, :2].sigmoid() * 2. - 0.5
                pwh = (nps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[nb, na, ngj, ngi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    lcls_out = self.cls_loss(ps[:, 5:], tcls[i][:, 1:], tcls[i][:, 0],
                                        self.model.model[-1].text_embeddings)
                    for j in range(len(lcls_items)):
                        lcls_items[j] += lcls_out[j]
                   
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, limg, ltext, lself_img, lself_text, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        #targets, embeddings = targets[:, :6], targets[:, 6:]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, embeddings = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets[:, :6].repeat(na, 1, 1), ai[:, :, None], targets[:, 6:].repeat(na, 1, 1)), 2)  # append anchor indices
       
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
       
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = torch.cat((targets[:, :, :7] * gain, targets[:, :, 7:]), dim=2)
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors'
            tcls.append(torch.cat([c.unsqueeze(-1), t[:, 7:]], dim=1))  # class
        # other negative values or other indicators may be used to include in localization losses
        used_idxs = [torch.Tensor([j for j in range(len(i)) if i[j][0] != -1]).type(torch.LongTensor) for i in tcls]
        return (tcls, indices,
                [tbox[i][used_idxs[i]] for i in range(len(tbox))],
                [tuple(j[used_idxs[i]] for j in indices[i]) for i in range(len(indices))],
                [anch[i][used_idxs[i]] for i in range(len(anch))])
'''

