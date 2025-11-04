# YOLOv5 ������ by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

import os

from utils.metrics import bbox_iou, probiou
from utils.torch_utils import is_parallel

from utils.general import xywh2xyxy, xyxyxyxy2xywhr, xywhr2xyxyxyxy, box_iou_liu, process_target_liu
from models.yolo import OUT_LAYER
from .tal import TaskAlignedAssigner, RotatedTaskAlignedAssigner,dist2bbox, dist2rbox, bbox2dist,rbox2dist
from models.utils.loss import RTDETRDetectionLoss
from utils.general import check_version
from tensor.tensor import normalize_dim

from scipy.optimize import linear_sum_assignment

from models.HA23 import A23s2abuv

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
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
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

def get_global_target_indices(target_gt_idx, fg_mask, mask_gt):
    # 将 target_gt_idx + fg_mask 映射为 targets[ntb,6] 中的全局索引（ntb = 总 GT 数量）
    # 参数：
    #     target_gt_idx: Tensor[B, ntotal] — 每个 anchor 匹配的（图像内）GT 索引
    #     fg_mask:       Tensor[B, ntotal] — 前景掩码
    #     mask_gt:       Tensor[B, n_gt, 1] — 指示每张图有多少 GT（1 有效，0 padding）
    # 返回：
    #     fg_global_idx: Tensor[N_fg] — 每个前景 anchor 匹配的 targets 全局索引（在 targets[ntb,6] 中的位置）
    B, ntotal = target_gt_idx.shape
    # 计算每张图的 GT 数量
    gt_per_img = mask_gt.squeeze(-1).sum(1).long()  # shape: [B]
    gt_offsets = torch.cumsum(torch.cat([gt_per_img.new_zeros(1), gt_per_img[:-1]]), dim=0)  # shape: [B]
    # 展开 & 过滤前景
    target_gt_idx_flat = target_gt_idx.view(-1)             # [B * ntotal]
    fg_mask_flat = fg_mask.view(-1)                         # [B * ntotal]
    batch_ids = torch.arange(B, device=target_gt_idx.device).view(-1,1).expand(B, ntotal).reshape(-1)  # [B * ntotal]
    # 过滤出前景的局部索引和 batch 索引
    fg_gt_idx = target_gt_idx_flat[fg_mask_flat]            # [N_fg]
    fg_batch_ids = batch_ids[fg_mask_flat]                  # [N_fg]
    # 计算全局 GT 索引：局部索引 + 偏移
    fg_global_idx = fg_gt_idx + gt_offsets[fg_batch_ids]    # [N_fg]

    return fg_global_idx

def get_global_target_indices_rot(target_gt_idx, fg_mask, mask_gt):
    # Convert per-batch GT indices from assigner into flat indices for original targets tensor.
    batch_id = torch.arange(target_gt_idx.shape[0], device=target_gt_idx.device).view(-1, 1).repeat(1, target_gt_idx.shape[1])  # shape [B, ntotal]
    flat_mask = fg_mask.view(-1)  # [B*ntotal]
    flat_gt_idx = target_gt_idx.view(-1)[flat_mask]  # [num_fg]
    flat_batch_id = batch_id.view(-1)[flat_mask]     # [num_fg]
    
    # Count how many GTs per image in mask_gt to build offsets
    num_gt_per_image = mask_gt.squeeze(-1).sum(1).to(torch.int)  # [B]
    gt_offsets = torch.cumsum(torch.cat([num_gt_per_image.new_zeros(1), num_gt_per_image[:-1]]), 0)  # [B]
    
    return flat_gt_idx + gt_offsets[flat_batch_id]  # [num_fg]

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, cen_tobj=False, use_ioa=False,vec_names=None,neg_mode=1, np_max = 100):
        super(ComputeLoss, self).__init__()
        self.h = model.hyp  # hyperparameters
        self.sort_obj_iou = self.h.get('sort_obj_iou',0)
        self.cen_tobj = self.h.get('cen_tobj',cen_tobj)
        self.use_ioa = use_ioa
        self.neg_mode = neg_mode
        self.np_max = np_max
        device = next(model.parameters()).device  # get model device
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=self.h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = self.h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        if hasattr(model,'module') or hasattr(model,'model'):
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
            det = model.module.get_module_byname('Detect') if is_parallel(model) else model.get_module_byname('Detect')  # Detect() module
        else:
            det = None
        if det is not None:
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
            self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, self.h, autobalance
        if vec_names is not None:
            self.vec_names = vec_names.to(device=device)
        
        # self.mode = 'origin'
        self.m = None
        mname = ''
        if hasattr(model,'module') or hasattr(model,'model'):
            for mname in OUT_LAYER.keys():
                m = model.module.get_module_byname(mname) if is_parallel(model) else model.get_module_byname(mname)
                if m is not None:
                    self.m = m
                    break

        if mname in ['DetectDFL', 'OBB', 'YoloText', 'OBBText', 'DetectDFL_xn', 'OBB_xn']:
            tal_topk = 10
            self.nc = m.nc
            self.na = 1
            self.nl = m.nl
            self.strides = m.stride
            self.dfl_loss = m.compute_ecloss_dim3
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.reg_max = m.reg_max
            if mname == 'DetectDFL' or mname=='YoloText' or mname=='DetectDFL_xn':
                # self.mode = 'dfl'
                self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
                self.__call = self.call_text
                self.bbox_loss = BboxLoss(self.reg_max).to(device)
            elif mname == 'OBB' or mname == 'OBBText' or mname == 'OBB_xn':
                # self.mode = 'obb'
                self.assigner = RotatedTaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
                self.__call = self.call_obb_a23
                self.bbox_loss = RotatedBboxLoss(self.reg_max).to(device)
                if mname == 'OBBText':
                    if neg_mode: #neg_mode==1 / 2
                        from transformers import CLIPProcessor, CLIPModel
	                    # 加载预训练的 CLIP 模型和处理器
                        # home_path = os.path.expanduser("~")
	                    # clip_path = os.path.join(home_path,'.cache/huggingface/hub/models--openai--clip-vit-base-patch32')
                        clip_path = 'openai/clip-vit-base-patch32'
                        self.clip = CLIPModel.from_pretrained(clip_path).to(device)
                        self.clip_proc = CLIPProcessor.from_pretrained(clip_path)
                        if neg_mode==1: #clip word none
                            inputs = self.clip_proc(text='none', return_tensors="pt", padding=True).to(device)
                            with torch.no_grad():
                                self.none_vec = self.clip.get_text_features(**inputs)  # 形状: [nc, 512]
                                self.none_vec = normalize_dim(self.none_vec, dim=-1)

            self.proj = m.proj #torch.arange(self.reg_max, dtype=torch.float, device=device)
        elif mname in ['Detect', 'DetectROT']:
            for k in 'na', 'nc', 'nl', 'anchors':
                setattr(self, k, getattr(det, k))
            self.__call = self.call_yolov5
        elif mname == 'RTDETRDecoder':
            tal_topk = 10
            self.nc = m.nc
            self.na = 1
            self.nl = m.nl
            # self.dfl_loss = m.compute_ecloss_dim3
            self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.__call = self.call_text
            # self.bbox_loss = BboxLoss(self.reg_max).to(device)
        elif mname == '':
            self.__call = self.call_detr
        else:
            raise RuntimeError("模型找不到输出层[{OUT_LAYER}], 或者ComputeLoss还没适配当前模型输出层")
        
        # 初始化 SmoothL1Loss（不自动求平均，逐元素计算）
        self.sl1loss_a23 = nn.SmoothL1Loss()


    def __call__(self, *args, **kwds):
        return self.__call(*args, **kwds)
    
    def call_yolov5(self, p, targets, paths = None, master_path=''):   # predictions, targets, model
        device = targets.device
        bs = p[0].shape[0]
        assert paths is None or bs==len(paths)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        #tcls[nl][n,1(cls)]
        #tbox[nl][n,4(xywh)]
        #indices[nl][4(b,a,gj,gi)][n]
        #anchors[nl][n,2(aw,ah)]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            #pi[b,a,gj,gi,4(rect)+1(obj)+cls]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            #[n]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            #tobj[b,a,gj,gi]
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                #[b,a,gj,gi,1(obj)+4(rect[xywh])+cls]-->ps[n,4(rect[xywh])+1(obj)+cls]
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy[n,2]
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                # pwh[n,2]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #pbox[n,4]
                #iou = bbox_iou(pbox, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze(-1)  # iou(prediction, target)
                # iou[n]
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                if self.cen_tobj:
                    pbox_cen = torch.cat((torch.full_like(pxy, 0.5, device=device), anchors[i]), 1)  # predicted box
                    iou_cen = bbox_iou(pbox_cen, tbox[i], CIoU=True).squeeze(-1)  # iou(prediction, target)
                    score_iou = iou_cen.detach().clamp(0).type(tobj.dtype)
                else:
                    score_iou = iou.detach().clamp(0).type(tobj.dtype)
                # score_iou[n]
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                # 所有目标对应的网格anchors都给1，其他都在初始化时刷0

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    if master_path=='':
                        # ps[:, 5:]的shape是[n, cls]  cn==0  全部填0
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        # t[n, cls]
                        # tcls[i]的shape是[n]  cp==1
                        # range(n)是一个shape为[n]的数组，下标从0~n-1
                        t[range(n), tcls[i]] = self.cp
                        ps_cls = ps[:, 5:]
                        lcls += self.BCEcls(ps_cls,t)  # BCE MSE
                        #det1_cls = t-ps[:, 5:]
                        #lcls2 += self.BCEcls(ps2[:, 5:], det1_cls)  # BCE
                        #master_masks = torch.ones([bs],device=tobj.device).to(torch.bool)
                    else:
                        # 生成布尔过滤器数组，True 表示包含 master_path 的路径，False 表示不包含
                        master_masks = torch.tensor([master_path in path for path in paths],device=tobj.device)
                        assert master_masks.shape[0]==bs
                        assert master_masks.sum().item()<=bs
                        if master_masks.sum().item()<bs:
                            # 取反布尔数组
                            #master_masks_reversed = np.logical_not(master_masks.cpu().numpy())
                            # 根据取反后的布尔数组过滤 paths，得到包含 include_str 的路径
                            app_paths = np.array(paths)[~master_masks.cpu().numpy()]
                            assert len(app_paths)>0
                            #print(app_paths)

                        master_obj_masks = master_masks[b] #master_obj_masks[n]
                        assert master_obj_masks.shape[0]==n
                        assert master_obj_masks.sum().item()<=n
                        if master_obj_masks.sum() > 0:
                            t = torch.full_like(ps[master_obj_masks, 7:], self.cn, device=device)  # targets
                            #t[n,class]==self.cn==0
                            t[range(master_obj_masks.sum().item()), tcls[i][master_obj_masks]] = self.cp
                            lcls += self.BCEcls(ps[master_obj_masks, 7:], t)  # BCE
                            if lcls.isnan():
                                print()

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            pobj = pi[..., 4]
            if self.h.get('pn_obj', 1) == 1:
                obji = self.BCEobj(pobj, tobj)
            else:#self.h['max_obj']默认256，dota1.5这种数据集里面最大目标数量可能超过256爆仓
                # 计算 tobj > 0 的损失
                # 创建一个等于0的掩码，只有一定比例的元素为True
                mask = torch.rand_like(tobj) < self.h.get('pn_obj', 1)  # mask_ratio 是你要保留的比例
                mask_obj,mask_noobj = tobj > 0, (tobj == 0) & mask
                objp_scale = self.h.get('objp_scale', 0.05) #((mask_obj.sum() + mask_noobj.sum()) / tobj.numel()).detach()
                objp_loss = self.BCEobj(pobj[mask_obj], tobj[mask_obj])
                objn_loss = self.BCEobj(pobj[mask_noobj], tobj[mask_noobj])
                obji = objp_scale * objp_loss + objn_loss

            lobj += obji * self.balance[i]  # obj loss
            #det1 = tobj - pi[..., 4]
            #lobj2 = self.MseLoss(pi2[..., 4], det1)
            if self.autobalance: #多个不同尺度直接的学习因子平衡
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def call_dfl(self, preds, targets, imgsz, ioa, samples, **kwds):#preds[nl=3][B,C,H,W]  targets[ntb,6=1(b)+1(cls)+4(xywh)]  ioa[ntb]
        assert ioa.shape[0]==targets.shape[0]
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # pred_distri, pred_scores, _ = preds
        feats = preds[1] if isinstance(preds, tuple) else preds #feats[3][B,80=16*4(ltrb)+cls,H,W]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], feats[0].shape[1], -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )#feats[3][B,80=16*4(ltrb)+cls,H,W] -> pred_distri[b,64=16*4(ltrb),ntotal] + pred_scores[b,cls,ntotal]  ntotal=20*20+40*40+80*80

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #pred_scores[b,cls,ntotal]->pred_scores[b,ntotal,cls]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #pred_distri[b,64=16*4(ltrb),ntotal]->pred_distri[b,ntotal,64=16*4(ltrb)]
        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        loss = torch.zeros(3, device=device)  # lbox, lcls, ldfl        
        anchor_points, stride_tensor = self.make_anchors(imgsz, self.strides, dtype, device, 0.5)# self.mdetect.anchor_points, self.mdetect.stride_tensor #anchor_points[ntotal,2]  stride_tensor[ntotal,1]
        imgsz = torch.tensor(imgsz, dtype=dtype, device=device)  # image size (h,w)

        # Targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) #targets[nt,6=1(b)+1(cls)+4(xywh)]
        targets_b,sort_indices = self.preprocess_dfl(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]], device=device) #->targets_b[b,nt,5=1(cls)+4(xyxy)]
        targets = targets[sort_indices]
        gt_labels, gt_bboxes = targets_b.split((1, 4), 2)  # gt_labels[b,nt,1(cls)], gt_bboxes[b,nt,4(xyxy)]
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) #mask_gt[b,nt,1] pad 0 at end

        # Pboxes
        pred_bboxes = dist2bbox(self.bbox_decode(pred_distri), anchor_points, xywh=False) #anchor_points[ntotal,2] + pred_distri[b,ntotal,16(ng)*4(ltrb)] -> pred_bboxes[b, ntotal, 4(xyxy)]
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(), #pred_scores[b,ntotal,cls=16)]
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), #pred_bboxes[b, ntotal, 4(xyxy)]
            anchor_points * stride_tensor, #anchor_points[ntotal,2]*stride_tensor[ntotal,1]->[ntotal,2]  anchor_coords*stride->pixel coords
            gt_labels, #gt_labels[b,nt,1(cls)]
            gt_bboxes, #mask_gt[b,nt,4(xyxy)]
            mask_gt, #mask_gt[b,nt,1]  pad 0 at end of mask_gt
        ) #->target_labels[b,ntotal] target_bboxes[b,ntotal,4(xyxy)] target_scores[b,ntotal,cls] fg_mask[b,ntotal]  target_gt_idx[b,ntotal]

        # Cls loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype)) #pred_scores[b,ntotal,cls]~target_scores[b,ntotal,cls]->bce_loss[b,ntotal,cls]
        bsid = get_global_target_indices(target_gt_idx,fg_mask,mask_gt) #bsid[mnt]
        assert torch.max(bsid)+1 <= targets.shape[0]
        if self.use_ioa:
            assert targets.shape[0]==ioa.shape[0]
            ioa_mask = ioa[bsid] #ioa[ntb->bsid[mnt]]->ioa_mask[mnt]
            ioa_weights = torch.ones_like(fg_mask,dtype=torch.float,device=fg_mask.device) #ioa_weights[b,ntotal]
            ioa_weights[fg_mask] *= ioa_mask #target_scores[b,ntotal]->[mnt]
            bce_loss *= ioa_weights.unsqueeze(-1)  #->bce_loss[b,ntotal,cls]
            target_scores_sum = max((target_scores*ioa_weights.unsqueeze(-1)).sum(), 1) #target_scores_sum
        else:
            target_scores_sum = max(target_scores.sum(), 1) #target_scores_sum
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = bce_loss.sum() / target_scores_sum #loss

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor #target_bboxes[b,ntotal,4(xyxy)] / stride_tensor[ntotal,1] -> target_bboxes[b,ntotal,4(xyxy)] grid coords
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # target_ltrb = self.bbox2dist(anchor_points, target_bboxes)
            # loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
            # loss[2] = (self.dfl_loss(pred_distri, target_ltrb, fg_mask) * weight).sum() / target_scores_sum
            # loss[0], loss[2] = self.bbox_loss(
            #     pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            # )#pred_distri[B,ntotal,64=16*4(ltrb)] pred_bboxes[B,ntotal,4(ltrb)]  anchor_points[ntotal,2] target_bboxes[b,ntotal,4] target_scores[b,ntotal,cls] fg_mask[b,ntotal]
        # torch.cuda.empty_cache()
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp.get('dfl', 0.01)  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def call_text(self, preds_all, targets, imgsz, ioa, samples, **kwds):#targets[ntb,6=1(b)+1(cls)+4(xywh)]  ioa[ntb]
        ntb = targets.shape[0]
        assert ioa.shape[0]==ntb
        #if isinstance(preds_all[0],list):#cls|box + pred_text
        if isinstance(preds_all,tuple) and isinstance(preds_all[0],list):#txt loss
            assert len(preds_all)==2
            if isinstance(preds_all[0],list):#train case: cls|box + pred_text
                preds,pred_text = preds_all
                # preds[nl=3][B,16*4(box)+nc,H,W] pred_text[b,ntotal,Tmax*n_embd]
            else:#val case
                if isinstance(preds_all[1],list):
                    out, preds = preds_all
                    assert len(preds)==3
                    nc = preds[0].shape[1] - self.reg_max * 4
                    nc_box_size = nc + 4
                    pred_text = out[:,:,nc_box_size:] ##pred_text[b,ntotal,nembd]
                    if pred_text.shape[-1]==0:
                        pred_text = None
                else:
                    pred_text = None
        else: #dfl loss
            if isinstance(preds_all[1],list):
                out, preds = preds_all
                assert len(preds)==3
                nc = preds[0].shape[1] - self.reg_max * 4
                nc_box_size = nc + 4
                pred_text = out[:,:,nc_box_size:] ##pred_text[b,ntotal,nembd]
                if pred_text.shape[-1]==0:
                    pred_text = None
            else:
                preds,pred_text = preds_all,None
        
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # pred_distri, pred_scores, _ = preds
        feats = preds[1] if isinstance(preds, tuple) else preds #feats[3][B,80=16*4(ltrb)+cls,H,W]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], feats[0].shape[1], -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )#feats[3][B,80=16*4(ltrb)+cls,H,W] -> pred_distri[b,64=16*4(ltrb),ntotal] + pred_scores[b,cls,ntotal]  ntotal=20*20+40*40+80*80

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #pred_scores[b,cls,ntotal]->pred_scores[b,ntotal,cls]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #pred_distri[b,64=16*4(ltrb),ntotal]->pred_distri[b,ntotal,64=16*4(ltrb)]
        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        loss = torch.zeros(3 + (pred_text is not None), device=device)  # lbox, lcls, ldfl, l_text
        anchor_points, stride_tensor = self.make_anchors(imgsz, self.strides, dtype, device, 0.5)# self.mdetect.anchor_points, self.mdetect.stride_tensor #anchor_points[ntotal,2]  stride_tensor[ntotal,1]
        imgsz = torch.tensor(imgsz, dtype=dtype, device=device)  # image size (h,w)

        # Targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) #targets[nt,6=1(b)+1(cls)+4(xywh)]
        targets_b,sort_indices = self.preprocess_dfl(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]], device=device) #->targets_b[b,nt,5=1(cls)+4(xyxy)]
        if sort_indices.shape[0]>0:
            targets = targets[sort_indices] #targets[nt,6=1(b)+1(cls)+4(xywh)]
            gt_labels, gt_bboxes = targets_b.split((1, 4), 2)  # gt_labels[b,nt,1(cls)], gt_bboxes[b,nt,4(xyxy)]
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) #mask_gt[b,nt,1] pad 0 at end
        else:
            targets = torch.zeros((0,targets.shape[-1])) #targets[nt,6=1(b)+1(cls)+4(xywh)]
            gt_labels, gt_bboxes = targets_b.split((1, 4), 2)  # gt_labels[b,nt,1(cls)], gt_bboxes[b,nt,4(xyxy)]
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) #mask_gt[b,nt,1] pad 0 at end

        if 0:#ver
            for i in range(batch_size):
                tcls_from_targets = targets[targets[:, 0] == i][:, 1].to(torch.int) #tcls_from_targets[nt]
                # valid_mask = mask_gt[i].view(-1).bool()
                valid_mask = mask_gt[i].squeeze(-1).bool()
                tcls_from_mask_gt = gt_labels[i][valid_mask].squeeze(-1)
                if not torch.equal(tcls_from_targets, tcls_from_mask_gt):
                    print(f"[!] Batch {i} mismatch:")
                    print("From targets:", tcls_from_targets)
                    print("From mask_gt:", tcls_from_mask_gt)

        # Pboxes
        pred_bboxes = dist2bbox(self.bbox_decode(pred_distri), anchor_points, xywh=False) #anchor_points[ntotal,2] + pred_distri[b,ntotal,16(ng)*4(ltrb)] -> pred_bboxes[b, ntotal, 4(xyxy)]
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        if hasattr(self,'vec_names'):
            nvec = self.vec_names.shape[0]
            assert pred_scores.shape[-1]==self.nc
            assert torch.max(gt_labels) + 1 <= nvec #gt_labels[b,nt,1(cls)]
            if self.nc != nvec:
                assert self.nc==1
                gt_labels[:,:,:] = 0

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(), #pred_scores[b,ntotal,cls=16)]
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), #pred_bboxes[b, ntotal, 4(xyxy)]
            anchor_points * stride_tensor, #anchor_points[ntotal,2]*stride_tensor[ntotal,1]->[ntotal,2]  anchor_coords*stride->pixel coords
            gt_labels, #gt_labels[b,nt,1(cls)]
            gt_bboxes, #mask_gt[b,nt,4(xyxy)]
            mask_gt, #mask_gt[b,nt,1]  pad 0 at end of mask_gt
        ) #->target_labels[b,ntotal] target_bboxes[b,ntotal,4(xyxy)] target_scores[b,ntotal,cls] fg_mask[b,ntotal]  target_gt_idx[b,ntotal]

        # Cls loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype)) #pred_scores[b,ntotal,cls]~target_scores[b,ntotal,cls]->bce_loss[b,ntotal,cls]
        bsid = get_global_target_indices(target_gt_idx,fg_mask,mask_gt) #bsid[mnt]
        assert bsid.shape[0]==0 or torch.max(bsid)+1 <= targets.shape[0]
        if self.use_ioa:
            assert targets.shape[0]==ioa.shape[0]
            ioa_mask = ioa[bsid] #ioa[ntb->bsid[mnt]]->ioa_mask[mnt]
            ioa_weights = torch.ones_like(fg_mask,dtype=torch.float,device=fg_mask.device) #ioa_weights[b,ntotal]
            ioa_weights[fg_mask] *= ioa_mask #target_scores[b,ntotal]->[mnt]
            bce_loss *= ioa_weights.unsqueeze(-1)  #->bce_loss[b,ntotal,cls]
            target_scores_sum = max((target_scores*ioa_weights.unsqueeze(-1)).sum(), 1) #target_scores_sum
        else:
            target_scores_sum = max(target_scores.sum(), 1) #target_scores_sum
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = bce_loss.sum() / target_scores_sum #loss

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor #target_bboxes[b,ntotal,4(xyxy)] / stride_tensor[ntotal,1] -> target_bboxes[b,ntotal,4(xyxy)] grid coords
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # target_ltrb = self.bbox2dist(anchor_points, target_bboxes)
            # loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
            # loss[2] = (self.dfl_loss(pred_distri, target_ltrb, fg_mask) * weight).sum() / target_scores_sum
            # loss[0], loss[2] = self.bbox_loss(
            #     pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            # )#pred_distri[B,ntotal,64=16*4(ltrb)] pred_bboxes[B,ntotal,4(ltrb)]  anchor_points[ntotal,2] target_bboxes[b,ntotal,4] target_scores[b,ntotal,cls] fg_mask[b,ntotal]
            #
            #pred_text[fg_mask][nt,TMax*n_embd]
            # if we are given some desired targets also calculate the loss
            #YoloText
            if pred_text is not None:#pred_text[b,ntotal,TMax*n_embd]
                imgids = target_gt_idx[fg_mask] #[mnt]
                mnt = imgids.shape[0]
                assert samples['name'].shape[0]==ntb
                assert samples['cls'].shape[0]==ntb
                #
                x = self.m.pred_text(pred_text[fg_mask]) #pred_text[b,ntotal,TMax*n_embd]->[(fg_mask)mnt,TMax*n_embd]->[mnt,(att-blocks)TMax*n_embd]->x[mnt,TMax,n_embd]
                # assert x.shape[0]>=ntb
                assert x.shape[0]==mnt
                #
                if not hasattr(self.m,'vocab_size'):
                    yword = normalize_dim(x[:,0],dim=-1)#x[mnt,TMax,n_embd]->[mnt,n_embd]->yword[mnt,n_embd=512]
                    #
                    tcls = targets[bsid][:,1].to(torch.int) #[ntb,6=1(b)+1(cls)+4(xywh)]->[mnt,6=1(b)+1(cls)+4(xywh)]->tcls[mnt]
                    assert tcls.shape==(mnt,)
                    gt_text = self.vec_names[tcls] #vec_names[nc->tcls[mnt],n_embd=512] -> gt_text[mnt,n_embd=512]
                    assert gt_text.shape==(mnt,512)
                    assert yword.dtype==gt_text.dtype
                    assert yword.shape==gt_text.shape
                    
                    # 点积计算（逐行）
                    dot = torch.sum(yword * gt_text, dim=-1)  # dot[nt]
                    # 验证结果范围
                    assert torch.all(dot >= -1) and torch.all(dot <= 1), "Dot product out of [-1, 1] range"
                    loss[3] = (1.0 - dot).mean()
                    #
                    if self.vec_names is not None:#vec_names[nc,n_embd=512]
                        gtcls = samples['cls'].to(bsid.device)[sort_indices][bsid] #cls[ntb->bsid[mnt]]->gtcls[mnt]
                        assert gtcls.shape==(mnt,)
                        
                        mcid = target_labels[fg_mask] #target_labels[b,ntotal]-> mcid[mnt]
                        assert tcls.shape==mcid.shape
                        if nvec == self.nc:
                            assert torch.equal(mcid, tcls)
                            assert torch.equal(gtcls, tcls)
                        
                        assert samples['vec'].shape[0]==ntb
                        vec = samples['vec'].to(bsid.device)[sort_indices][bsid] #vec[ntb->mnt,n_embd=512]
                        torch.equal(vec,gt_text) #delta[mnt,n_embd=512]
                    
                else:
                    logits = self.m.text_head(x.half()) #x[mnt, TMax, n_embd]-->logits[mnt,TMax, vocab_size] onehot词向量
                    #samples['name'][ntb,TMax]
                    gtids = torch.from_numpy(samples['name']).to(target_gt_idx.device)[mntid] #samples['name'][nt,TMax]->gtids[mnt,TMax]
                    minTMax = min(gtids.shape[1],logits.shape[1])
                    if gtids.shape[1] > minTMax:
                        gtids = gtids[:,:minTMax].contiguous() #gtids[mnt,TMax]->gtids[mnt,minTMax]
                        # 获取每行中最后一个不为 eos_id 的位置
                        for i in range(gtids.shape[0]):
                            # 对每行进行处理，找到最后一个不为 eos_id 的位置
                            non_eos_ids = (gtids[i] != self.m.eos_token_id+1).nonzero(as_tuple=True)[0]
                            if non_eos_ids.numel() > 0:  # 确保存在非 eos_id+1 的元素
                                last_non_eos_idx = non_eos_ids[-1].item()
                                assert last_non_eos_idx>=0 and last_non_eos_idx<minTMax
                                if last_non_eos_idx == minTMax-1:
                                    gtids[i, last_non_eos_idx] = self.m.eos_token_id
                                assert gtids[i, last_non_eos_idx] == self.m.eos_token_id
                    if logits.shape[1] > minTMax:
                        logits = logits[:,:minTMax,:].contiguous()
                    assert(gtids.shape[:2]==logits.shape[:2]) #logits[mnt,TMax,voc_size] ~ gtids[mnt,TMax]
                    if 0:
                        loss[3] = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                    gtids.view(-1),
                                                    ignore_index=self.m.eos_token_id+1,  # 忽略填充标记
                                                    reduction='mean'  # 默认是 mean，也可以不写
                                                )
                    else:
                        loss_flat = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                    gtids.view(-1),
                                                    ignore_index=self.m.eos_token_id+1,  # 忽略填充标记
                                                    reduction='none'  # 默认是 mean，也可以不写
                                                ) #loss_flat[mnt * TMax]
                        # 在 dim=2 上求最大值，得到 ws[b, ntotal]
                        ws, _ = torch.max(target_scores, dim=2)  # ws[b, ntotal]
                        # 使用 fg_mask 过滤 ws[b, ntotal]，得到 ws[nmt]
                        ws_filtered = ws[fg_mask]  # ws_filtered[nmt]
                        ws_expanded = ws_filtered.unsqueeze(1).expand(-1, self.m.TMax).reshape(-1)  # [mnt * TMax]
                        # 加权平均->weighted_loss
                        loss[3] = torch.sum(loss_flat * ws_expanded) / torch.sum(ws_expanded)
                    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), samples['name'].view(-1), ignore_index=-1) #samples['name'][nt,TMax]
                    loss[3] *= self.hyp.text #text gain
                    #
                    _, predicted_ids = logits.max(dim=-1) # logits[mnt,TMax,voc_size]->predicted_ids[mnt,TMax]
                    # 计算准确率
                    mask = gtids!=self.m.eos_token_id+1
                    correct = (predicted_ids[mask] == gtids[mask])  # shape: [mnt, TMax]
                    self.accuracy = correct.sum().float() / correct.numel()  # 总正确数 / 总元素数
                    # print(accuracy)
        # torch.cuda.empty_cache()
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp.get('dfl', 0.01)  # dfl gain
        if pred_text is not None:
            loss[3] *= self.hyp.get('txt', 0.1)  # txt gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, txt)

    def call_obb_a23(self, pabuv, preds_all, targets, imgsz, ioa, samples, **kwds):
        ntb = targets.shape[0]
        #targets[ntb,10=1+1+8(xyxyxyxy)] ioa[ntb]
        assert ioa.shape[0]==ntb
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # pred_distri, pred_scores, pred_bboxes= preds

        # if isinstance(preds_all,tuple) and isinstance(preds_all[0],tuple):
        #     preds,pred_text = preds_all
        # else:
        #     preds,pred_text = preds_all,None
        #     # preds[nl=3][B,16*4(box)+nc,H,W] pred_text[b,ntotal,Tmax*n_embd]
        
        if isinstance(preds_all,tuple) and isinstance(preds_all[0],tuple):#txt loss
            assert len(preds_all)==2
            if isinstance(preds_all[0],tuple):#train case: cls|box + pred_text
                preds,pred_text = preds_all
                # preds[nl=3][B,16*4(box)+nc,H,W] pred_text[b,ntotal,Tmax*n_embd]
            else:#val case
                if isinstance(preds_all[1],list):
                    out, preds = preds_all
                    assert len(preds)==3
                    nc = preds[0].shape[1] - self.reg_max * 4
                    nc_box_size = nc + 5
                    pred_text = out[:,:,nc_box_size:] #pred_text[b,ntotal,nembd]
                    if pred_text.shape[-1]==0:
                        pred_text = None
                else:
                    pred_text = None
        else: #dfl loss
            if isinstance(preds_all[1],tuple):
                out, preds = preds_all
                assert len(preds[0])==3
                nc = preds[0][0].shape[1] - self.reg_max * 4
                nc_box_size = nc + 5
                pred_text = out[:,:,nc_box_size:] #pred_text[b,ntotal,nembd]
                if pred_text.shape[-1]==0:
                    pred_text = None
            else:
                preds,pred_text = preds_all,None
                # preds[nl=3][B,16*4(box)+nc,H,W] pred_text[b,ntotal,Tmax*n_embd]

        # if isinstance(preds_all[0],list):#cls|box + pred_text
        #     preds,pred_text = preds_all
        #     # preds[nl=3][B,16*4(box)+nc,H,W] pred_text[b,ntotal,Tmax*n_embd]
        # else:#val case
        #     out, preds = preds_all
        #     assert len(preds)==3
        #     nc = preds[0].shape[1] - self.reg_max * 4
        #     nc_box_size = nc + 4
        #     pred_text = out[:,:,nc_box_size:] #-Tmax*n_embd
            
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1] #feats[nl=3][B,C=16*4+nc,H,W]  pred_angle[b,1,ntotal]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], feats[0].shape[1], -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        ) #feats[nl=3][B,C=16*4+nc,H,W]->feats[B,C=16*4(box)+nc,ntotal]->pred_distri[B,16*4(box),ntotal] + pred_scores[B,nc,ntotal]
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #pred_scores[B,nc,ntotal]->pred_scores[B,ntotal,nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #pred_distri[B,16*4(box),ntotal]->pred_distri[B,ntotal,16*4(box)]
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()   #pred_angle[b,1,ntotal]->pred_angle[b,ntotal,1]

        device = pred_scores.device
        dtype = pred_scores.dtype
        assert pred_scores.shape[0]==batch_size
        loss = torch.zeros(4+(pred_text is not None), device=device)  # lbox, lcls, ldfl        
        anchor_points, stride_tensor = self.make_anchors(imgsz, self.strides, dtype, device, 0.5)# self.mdetect.anchor_points, self.mdetect.stride_tensor #anchor_points[ntotal,2]  stride_tensor[ntotal,1]
        imgsz = torch.tensor(imgsz, dtype=dtype, device=device)  # image size (h,w)

        # Targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) #targets[nt,7=1(b)+1(cls)+5(xywhr)]
        if len(targets) > 0:
            targets = torch.cat([targets[:, :2], xyxyxyxy2xywhr(targets[:, 2:])], dim=-1)#targets[nt,7=1(b)+1(cls)+5(xywhr)]
        else:
            targets = torch.zeros([0, 7], device=device)
        # targets_wh = targets[:, 4:6].prod(-1).argsort(-1)
        # targets = targets[targets_wh.long()]#targets[nt,7=1(b)+1(cls)+5(xywhr)]
        rw, rh = targets[:, 4] * imgsz[1].item(), targets[:, 5] * imgsz[0].item()
        rwhmask = (rw >= 2) & (rh >= 2)
        targets = targets[rwhmask]  # filter rboxes of tiny size to stabilize training
        #
        targets_b,sort_indices = self.preprocess_obb(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]], device=device) #->targets_b[b,nt,6=1(cls)+5(xywhr)]
        if len(sort_indices) > 0:
            targets = targets[sort_indices]
            gt_labels, gt_bboxes = targets_b.split((1, 5), 2)  # gt_labels[b,nt,1(cls)], gt_bboxes[b,nt,5(xywhr)]
            gt_hboxes,gt_angles = gt_bboxes.split((4, 1), 2)
            mask_gt = gt_hboxes.sum(2, keepdim=True).gt_(0.0) #mask_gt[b,nt,1] pad 0 at end

            if 0:#ver
                for i in range(batch_size):
                    tcls_from_targets = targets[targets[:, 0] == i][:, 1].to(torch.int) #tcls_from_targets[nt]
                    # valid_mask = mask_gt[i].view(-1).bool()
                    valid_mask = mask_gt[i].squeeze(-1).bool()
                    tcls_from_mask_gt = gt_labels[i][valid_mask].squeeze(-1).to(torch.int)
                    assert tcls_from_targets.shape == tcls_from_mask_gt.shape
                    if not torch.equal(tcls_from_targets, tcls_from_mask_gt):
                        print(f"[!] Batch {i} mismatch:")
                        print("From targets:", tcls_from_targets)
                        print("From mask_gt:", tcls_from_mask_gt)

            # Pboxes
            pred_bboxes = dist2rbox(self.bbox_decode(pred_distri), pred_angle, anchor_points, dim=-1) #anchor_points[ntotal,2] + pred_distri[b,ntotal,16(ng)*4(ltrb)] -> pred_bboxes[b, ntotal, 5(xywhr)]
            pred_bboxes = torch.cat((pred_bboxes, pred_angle), dim=-1)
            # pred_bboxes = self.bbox_decode(anchor_points, pred_distri) #anchor_points[ntotal,2] + pred_distri[b,ntotal,16(ng)*4(ltrb)] -> pred_bboxes[b, ntotal, 5(xywhr)]
            # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
            # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

            if hasattr(self,'vec_names'):
                nvec = self.vec_names.shape[0]
                assert pred_scores.shape[-1]==self.nc
                assert torch.max(gt_labels) + 1 <= nvec #gt_labels[b,nt,1(cls)]
                if self.nc != nvec:
                    assert self.nc==1
                    gt_labels[:,:,:] = 0

            bboxes_for_assigner = pred_bboxes.clone().detach() #->bboxes_for_assigner[B,ntotal,5(xywhr)]
            bboxes_for_assigner[..., :4] *= stride_tensor
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
                pred_scores.detach().sigmoid(), #pred_scores[b,ntotal,cls=16)]
                bboxes_for_assigner.type(gt_bboxes.dtype), #pred_bboxes[b, ntotal, 4(xyxy)]
                anchor_points * stride_tensor, #anchor_points[ntotal,2]*stride_tensor[ntotal,1]->[ntotal,2]  anchor_coords*stride->pixel coords
                gt_labels, #gt_labels[b,nt,1(cls)]
                gt_bboxes, #gt_bboxes[b,nt,5(xywhr)]
                mask_gt, #mask_gt[b,nt,1]  pad 0 at end of mask_gt
                stride_tensor #stride_tensor[ntotal,1]
            ) #->target_labels[b,ntotal] target_bboxes[b,ntotal,4(xyxy)] target_scores[b,ntotal,cls] fg_mask[b,ntotal]  target_gt_idx[b,ntotal]

            # Cls loss
            bce_loss = self.bce(pred_scores, target_scores.to(dtype)) #pred_scores[b,ntotal,cls]~target_scores[b,ntotal,cls]->bce_loss[b,ntotal,cls]
            bsid = get_global_target_indices_rot(target_gt_idx,fg_mask,mask_gt) #target_gt_idx[b,ntotal],fg_mask[b,ntotal] -> bsid[mnt]
            assert bsid.shape[0]==0 or torch.max(bsid)+1 <= targets.shape[0]
            if self.use_ioa:
                ioa = ioa[rwhmask]
                assert targets.shape[0]==ioa.shape[0]
                ioa_mask = ioa[bsid] #ioa[ntb->bsid[mnt]]->ioa_mask[mnt]
                ioa_weights = torch.ones_like(fg_mask,dtype=torch.float,device=fg_mask.device) #ioa_weights[b,ntotal]
                ioa_weights[fg_mask] *= ioa_mask #target_scores[b,ntotal]->[mnt]
                bce_loss *= ioa_weights.unsqueeze(-1)  #->bce_loss[b,ntotal,cls]
                target_scores_sum = max((target_scores*ioa_weights.unsqueeze(-1)).sum(), 1) #target_scores_sum
            else:
                target_scores_sum = max(target_scores.sum(), 1) #target_scores_sum
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = bce_loss.sum() / target_scores_sum #loss

            # Bbox loss
            if fg_mask.sum():
                target_bboxes[..., :4] /= stride_tensor #target_bboxes[b,ntotal,4(xyxy)] / stride_tensor[ntotal,1] -> target_bboxes[b,ntotal,4(xyxy)] grid coords
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                )
                # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
                # iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
                # target_ltrb = self.bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]))
                # loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
                # loss_dfl = self.dfl_loss(pred_distri, target_ltrb, fg_mask) * weight
                # loss[2] = loss_dfl.sum() / target_scores_sum
                # # loss[0], loss[2] = self.bbox_loss(
                #     pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                # )#pred_distri[B,ntotal,64=16*4(ltrb)] pred_bboxes[B,ntotal,4(ltrb)]  anchor_points[ntotal,2] target_bboxes[b,ntotal,4] target_scores[b,ntotal,cls] fg_mask[b,ntotal]
                #YoloText
                if pred_text is not None:#pred_text[b,ntotal,TMax*n_embd]
                    imgids = target_gt_idx[fg_mask] #[mnt]
                    mnt = imgids.shape[0]
                    assert samples['name'].shape[0]==ntb
                    assert samples['cls'].shape[0]==ntb
                    #
                    if 'imgs' in kwds and self.neg_mode==2: #CLIP image
                        pred_text_neg = pred_text[~fg_mask] #pred_text[b,ntotal,nembd=512]->pred_text_neg[n_neg,n_embd]
                        n_neg = pred_text_neg.shape[0]
                        topk = 256  # 你想筛选出的行数，topk < n_neg
                        indices = torch.randperm(n_neg)[:topk]  #indices[topk] 随机打乱取前topk个

                        #pred_bboxes[b,ntotal,5(xywhr)]->pred_bboxes_abs[[n_neg,5(xywhr)]]->pred_bboxes_abs[topk,5(xywhr)]
                        pred_bboxes_abs = pred_bboxes[~fg_mask][indices].clone().detach() #pred_bboxes_abs[topk,5(xywhr)]

                        abs_bias = anchor_points[None].repeat(batch_size, 1, 1)[~fg_mask][indices]
                        abs_gain = stride_tensor[None].repeat(batch_size, 1, 4)[~fg_mask][indices] #stride_tensor[ntotal,1]->[b,ntotal,4]->abs_gain[topk,4]
                        
                        pred_bboxes_abs[...,:2] += abs_bias #pred_bboxes_abs[topk,5(xywhr)]
                        pred_bboxes_abs[...,:4] *= abs_gain #pred_bboxes_abs[topk,5(xywhr)]

                        distill_xy4 = xywhr2xyxyxyxy(pred_bboxes_abs) # pred_bboxes_abs[topk, 5] --> distill_xy4[topk, 4(pts), 2(xy)]
                        distill_ltrb = torch.cat([distill_xy4.amin(-2, keepdim=True), distill_xy4.amax(-2, keepdim=True)], dim=-2).view(topk, 4).round()
                        #[topk, 1, 2(minxy)],[topk, 1, 2(maxxy)]-->[topk, 2, 2(minxy)]-->distill_ltrb[topk, 4(ltrb)]
                        distill_ltrb[..., 0].clamp_(0, imgsz[1]) #distill_ltrb[topk,4(ltrb)]
                        distill_ltrb[..., 2].clamp_(0, imgsz[1])
                        distill_ltrb[..., 1].clamp_(0, imgsz[0])
                        distill_ltrb[..., 3].clamp_(0, imgsz[0])
                        #
                        neg_coords = (~fg_mask).nonzero(as_tuple=False) # neg_coords[n_neg, 2]
                        sampled_indices = neg_coords[indices]           # sampled_indices[topk, 2]
                        b_id = sampled_indices[:, 0]                    # b_id[topk]
                        ntotal_indices = sampled_indices[:, 1]          # ntotal_indices[topk]
                        #
                        #b_id[topk] distill_ltrb[topk,4(ltrb)]
                        mask_filter = (distill_ltrb[..., 2] > distill_ltrb[..., 0]) & (distill_ltrb[..., 3] > distill_ltrb[..., 1]) #mask_filter[topk]
                        # assert all(mask_filter)
                        if mask_filter.sum() > 0:
                            b_id, distill_ltrb  = b_id[mask_filter], distill_ltrb[mask_filter] #b_id[topk], distill_ltrb[topk,4(ltrb)]
                            distill_ltrb = distill_ltrb.long()
                            distill_img = [kwds['imgs'][b, :, y1:y2, x1:x2] for b, (x1, y1, x2, y2) in zip(b_id, distill_ltrb)]
                            #
                            # 生成图向量
                            inputs = self.clip_proc(images=distill_img, #distill_img[topk][C,h,w]
                                                    return_tensors="pt", 
                                                    do_rescale=False, 
                                                    padding=True,
                                                    input_data_format='channels_first').to(device)
                            with torch.no_grad():
                                clip_crop_vec = self.clip.get_image_features(**inputs) # clip_crop_vec[topk, nembd=512]
                            clip_crop_vec = normalize_dim(clip_crop_vec, dim = -1) # clip_crop_vec[topk, nembd=512]

                            pred_neg_sampled = pred_text_neg[indices][mask_filter] #pred_neg_sampled[topk,n_embd]
                            pred_neg_sampled = self.m.pred_text(pred_neg_sampled).squeeze(1) #pred_neg_sampled[topk,n_embd]->y[topk,1,n_embd]->y[topk,n_embd]
                            pred_neg_text_vec = normalize_dim(pred_neg_sampled, dim = -1) # pred_neg_sampled[topk,n_embd]->pred_neg_text_vec[topk, n_embd=512]
                            dot = torch.sum(pred_neg_text_vec * clip_crop_vec, dim=-1) #dot[topk]
                            assert torch.all(dot >= -1) and torch.all(dot <= 1), "Dot product out of [-1, 1] range"
                            loss[3] = (1.0 - dot).mean()
                        else: #->distill_img[b*topk][C,h,w]
                            print('mask_filter.sum()不应该为0, 训练可能存在问题')
                        # top 300
                    elif self.neg_mode==1: #CLIP word none
                        pred_text_neg = pred_text[~fg_mask] #pred_text[b,ntotal,nembd=512]->pred_text_neg[n_neg,n_embd]
                        n_neg = pred_text_neg.shape[0]
                        topk = 4096  # 你想筛选出的行数，topk < n_neg
                        indices = torch.randperm(n_neg)[:topk]  #indices[topk] 随机打乱取前topk个
                        x = self.m.pred_text(pred_text_neg[indices])
                        yword_neg = normalize_dim(x[:,0],dim=-1)
                        # 点积计算（逐行）
                        dot = torch.sum(yword_neg * self.none_vec, dim=-1)  # dot[nt]
                        # 验证结果范围
                        assert torch.all(dot >= -1) and torch.all(dot <= 1), "Dot product out of [-1, 1] range"
                        loss[3] = (1.0 - dot).mean()
                    else: #neg_mode==0 #no neg loss
                        loss[3] = 0

                    if not hasattr(self.m,'vocab_size'):
                        x = self.m.pred_text(pred_text[fg_mask]) #pred_text[b,ntotal,TMax*n_embd]->[(fg_mask)mnt,TMax*n_embd]->[mnt,(att-blocks)TMax*n_embd]->x[mnt,TMax,n_embd]
                        # assert x.shape[0]>=ntb
                        assert x.shape[0]==mnt
                        yword = normalize_dim(x[:,0],dim=-1)#x[mnt,TMax,n_embd]->[mnt,n_embd]->yword[mnt,n_embd=512]
                        #
                        tcls = targets[bsid][:,1].to(torch.int) #[ntb,6=1(b)+1(cls)+5(xywhr)]->[mnt,6=1(b)+1(cls)+5(xywhr)]->tcls[mnt]
                        
                        assert tcls.shape==(mnt,)
                        gt_text = self.vec_names[tcls] #vec_names[nc->tcls[mnt],n_embd=512] -> gt_text[mnt,n_embd=512]
                        assert gt_text.shape==(mnt,512)
                        assert yword.dtype==gt_text.dtype
                        assert yword.shape==gt_text.shape
                        
                        # 点积计算（逐行）
                        dot = torch.sum(yword * gt_text, dim=-1)  # dot[nt]
                        # 验证结果范围
                        assert torch.all(dot >= -1) and torch.all(dot <= 1), "Dot product out of [-1, 1] range"
                        loss[3] += (1.0 - dot).mean()
                        #
                        if self.vec_names is not None:#vec_names[nc,n_embd=512]
                            gtcls = samples['cls'].to(bsid.device)[rwhmask][sort_indices][bsid] #cls[ntb->bsid[mnt]]->gtcls[mnt]
                            assert gtcls.shape==(mnt,)

                            mcid = target_labels[fg_mask] #target_labels[b,ntotal]-> tcls[mnt]
                            assert tcls.shape==mcid.shape #mcid[mnt]
                            if nvec == self.nc:
                                assert torch.equal(mcid, tcls)
                                assert torch.equal(gtcls, tcls)
                            
                            assert samples['vec'].shape[0]==ntb
                            vec = samples['vec'].to(bsid.device)[rwhmask][sort_indices][bsid] #vec[ntb->mnt,n_embd=512]
                            assert vec.shape==gt_text.shape
                            torch.equal(vec,gt_text) #delta[mnt,n_embd=512]
            else:
                loss[0] += (pred_angle * 0).sum()
        
        gtA23s = samples['A23'] #gtA23s[B,2,3]
        if gtA23s is not None:
            gtabuv = A23s2abuv(imgsz,gtA23s) #gtabuv[bs,4(abuv)]
            assert gtabuv.shape==pabuv.shape
            assert gtabuv.device==pabuv.device
            # 创建权重张量：[n, 2, 3]，前两列权重 1，最后一列权重 scale_t
            loss[3] += self.sl1loss_a23(pabuv[:,:2],gtabuv[:,:2])
            loss[3] += self.sl1loss_a23(pabuv[:,2:],gtabuv[:,2:]) * self.hyp.get('a23_t',0.01)

        # torch.cuda.empty_cache()
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp.get('dfl', 0.01)  # dfl gain
        loss[3] *= self.hyp.get('a23',1.0)

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, txt)
    
    def call_detr(self,preds, target, imgsz, ioa, samples, **kwds):
        objs,pred_boxes,pred_clss = preds #objs[np_max,1], pred_boxes[np_max,4],pred_clss[np_max,nc]
        device = objs.device
        B = objs.shape[0]
        assert pred_boxes.shape[0]==B
        assert pred_clss.shape[0]==B
        # Process target if provided
        target, target_mask = process_target_liu(target,B) #target[bnt,6=1(b)+1(cls)+4(xywh)]->target[b,nmax,5=1(cls)+4(xywh)] for hbb, target_mask[B, nt_max]
        assert target.shape[:2]==target_mask.shape
        if target_mask.shape[1]>self.np_max:
            target, target_mask = target[:,:self.np_max,:], target_mask[:,:self.np_max]
        gt_boxes = target[..., 2:6]  # gt_boxes[B, nt_max, 4]
        gt_cls = target[..., 1].long()  # gt_cls[B, nt_max]
        nt_max = gt_boxes.size(1)
        assert target_mask.shape[-1]==nt_max

        loss = torch.zeros(3, device=device)  # lbox, lcls, ldfl

        for i in range(B):
            # step 1: 获取当前 batch 的预测和 GT
            pred_box = pred_boxes[i] # pred_box[np_max, 4]
            pred_cls = pred_clss[i]  # pred_cls[np_max, nc]
            gt_mask = target_mask[i] # gt_mask[nt_max]
            gt_boxes_i = gt_boxes[i][gt_mask]  # gt_boxes_i[nt, 4]
            gt_cls_i = gt_cls[i][gt_mask]      # gt_cls_i[nt]
            nt = len(gt_boxes_i)
            if nt > 0:
                # step 2: 计算匹配 cost（如 IOU + 类别差）
                iou_matrix = box_iou_liu(pred_box, gt_boxes_i)  # iou_matrix[np_max, nt]
                cls_matrix = -pred_cls.softmax(-1)[:, gt_cls_i]  # cls_matrix[np_max, nt], 类别越匹配越接近0
                assert iou_matrix.shape == cls_matrix.shape
                assert iou_matrix.shape[1] == nt
                assert iou_matrix.shape[0] >= nt

                cost = -iou_matrix + cls_matrix  # cost[np_max, nt] 越小越好
                cost = cost.detach().cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost) #row_idx[nt] col_idx[nt]

                # step 3: 标记匹配上的索引
                matched_pred_idx = torch.tensor(row_idx, dtype=torch.long, device=device) #matched_pred_idx[nt] 0~np_max-1
                matched_gt_idx = torch.tensor(col_idx, dtype=torch.long, device=device)   #matched_gt_idx[nt] 0~nt-1

                # step 4: 正样本 obj=1
                obj_target = torch.zeros(self.np_max, device=device) #obj_target[np_max]
                obj_target[matched_pred_idx] = 1.0
                # 计算 objectness loss
                loss_obj = F.binary_cross_entropy_with_logits(
                    objs[i].squeeze(-1),  # [np_max]
                    obj_target,  # obj_target[np_max]
                    reduction='mean'
                )

                # step 5: 计算 box 和 cls loss
                matched_pred_boxes = pred_box[matched_pred_idx] # matched_pred_boxes[nt, 4]
                matched_gt_boxes = gt_boxes_i[matched_gt_idx]   # matched_gt_boxes[nt, 4]
                loss_box = F.smooth_l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='mean')

                matched_cls = pred_cls[matched_pred_idx]  # pred_cls[np_max, nc]->matched_cls[nt, nc]
                matched_gt_cls = gt_cls_i[matched_gt_idx] # gt_cls_i[nt]        ->matched_gt_cls[nt]
                loss_cls = F.nll_loss(matched_cls, matched_gt_cls, reduction='mean')

                loss[0] += loss_box
                loss[1] += loss_obj
                loss[2] += loss_cls
        loss /= B

        if self.hyp is not None:
            loss[0] *= self.hyp.get('box',1.0) # box gain
            loss[1] *= self.hyp.get('obj',1.0) # obj gain
            loss[2] *= self.hyp.get('cls',1.0) # cls gain
        
        return loss.sum() * B, loss.detach()  # loss(box, cls, dfl, txt)

    def build_targets(self, p, targets):
        # p[list][b,a,gy,gx,4+1+cls]
        # targets[nt,6=(1(b)+1(cls)+4(box)]
        # targets = targets[max(targets[:, 4],targets[:, 5]).max(1)[0] > 0.01]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # gain[7] = {1,1,1,1,1,1,1}
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # ai[na,nt]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # targets[na,nt,7=(1(b)+1(cls)+4(box)+1(anchor))]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # off[5,2]

        for i in range(self.nl):
            shape = p[i].shape
            # anchors[3,na,2]
            anchors = self.anchors[i]
            # anchors[na,2]
            # p[i][b,a,gy,gx,4+1+cls]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # gain[7=(1(b)+1(cls)+4(box)+1(anchor))] = {1,1,gx,gy,gx,gy,1}

            # Match targets to anchors
            t = targets * gain
            # t[na,nt,7=(1(b)+1(cls)+4(box)+1(anchor))]
            if nt:
                # Matches
                # t[:, :, 4:6]的shape是    t      [na,nt,2]
                # anchors[:, None]的shape是anchors[na, ?,2]  得到?=nt
                #                                [na,nt,2]
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # r[na,nt,2] / anchors[na,?,2] = r[na,nt,2]

                # r1 = torch.max(r, 1. / r)的shape是r1[na,nt,2]
                # r1.max(2)意思是[r和1/r]在编号(2)的维度找最大值[0]，求最大值后会把维度2去掉,shape变为[na,nt]
                # 得到r1.max(2)[0]是shape为[na,nt]的最大值
                # 得到r1.max(2)[1]是shape为[na,nt]的最大值对应的整数编号{0,1}
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # 意思是[r和1/r]在编号(2)的维度找最大值[0],求最大值后会把维度2去掉，j的shape为[na,nt]的bool数组
                # r, 1. / r两者最大值都比较小，说明接近1，说明和相应anchor比较匹配

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # j[na, nt]
                # t[na, nt, 7 = (1(b) + 1(cls) + 4(box) + 1(anchor))]
                t = t[j]  # filter
                # t[n_match_obj,7=(1(b)+1(cls)+4(box)+1(anchor))]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # gxy[n_match_obj,2=(xy)]
                gxi = gain[[2, 3]] - gxy  # inverse
                # gxi[n_match_obj,2=(xy)]
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T #j[n_match_obj], k[n_match_obj]
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T #l[n_match_obj], m[n_match_obj]
                # j,k,l,m的shape全是[n_match_obj]
                j = torch.stack((torch.ones_like(j), j, k, l, m)) #j[5,n_match_obj]
                # j[5,n_match_obj] [第1行全1/目标中心x接近下限神经元j/目标中心y接近下限神经元k/目标中心x接近上限神经元l/目标中心y接近上限神经元m]
                # t.repeat((5, 1, 1))的shape是[5,n_match_obj,7 = (1(b) + 1(cls) + 4(box) + 1(anchor))]
                t = t.repeat((5, 1, 1))[j]
                # t[n_match_obj*3,7]  目标第1行全1，[目标x要么接近下限x，要么接近上限x],[目标y要么接近下限y，要么接近上限y],因此只有可能是选择3倍

                # gxy[n_match_obj, 2 = (xy)]
                # torch.zeros_like(gxy)[None]的shape是[?, n_match_obj, 2 = (xy)]
                # off[5,2]
                # off[:, None]的shape是[5, ?, 2]
                # [?1, n_match_obj, 2]和
                # [ 5,          ?2, 2]
                # 要能在一起计算，只有?1=5, ?2=n_match_obj  最终得到的shape是
                # [ 5, n_match_obj, 2]
                #得到(torch.zeros_like(gxy)[None] + off[:, None])的shape是[5, n_match_obj, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                #[5, n_match_obj, 2]经过j[5,n_match_obj]过滤得到offsets[n_match_obj * 3, 2]
                #相当于把n_match_obj*5的组合全部遍历了一遍,选择其中n_match_obj*3的就近的网格偏移
            else:
                t = targets[0] #targets[na=3,nt=0,7=(1(b)+1(cls)+4(box)+1(anchor))]->t[nt=0,7]
                offsets = 0

            # Define
            # t[:, :2]的shape是[3*n_match_obj,2]
            b, c = t[:, :2].long().T  # image, class
            # b[3*n_match_obj]  c[3*n_match_obj]
            # t[:, 2:4]的shape是[3*n_match_obj,2]
            gxy = t[:, 2:4]  # grid xy
            # gxy[3*n_match_obj,2]   得到3*n_match_obj个gxy网格(xy坐标复制了3倍)
            gwh = t[:, 4:6]  # grid wh
            # gwh[3*n_match_obj,2]   得到3*n_match_obj个gwh网格(wh坐标复制了3倍)
            gij = (gxy - offsets).long()
            # gij[3*n_match_obj,2]   得到3*n_match_obj个就近网格
            gi, gj = gij.T  # grid xy indices
            # gi[3*n_match_obj]  gj[3*n_match_obj]

            # Append
            a = t[:, 6].long()  # anchor indices
            # a[3*n_match_obj]
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # gxy - gij才是网络真正要拟合的相对偏移！
            # anchors[3,2]   a[3*n_match_obj]
            # anchors[a] shape是[3*n_match_obj,2]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def preprocess_dfl(self, targets, batch_size, scale_tensor, device):
        # Preprocesses the target counts and matches with the input batch size to output a tensor.
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=device)
            sort_indices = torch.zeros((0))
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
            global_indices = []  # 存储排序后的原始下标
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
                    #
                    bboxes = targets[matches, 2:]  #targets[nt,6=1(b)+1(cls)+4(xywh)]->bboxes[nt2,4(xywh)]
                    areas = bboxes[:, 2] * bboxes[:, 3]  # w * h
                    sort_idx = areas.argsort(descending=False)  # 根据面积降序排列

                    # 排序后的 bboxes 和 cls
                    bboxes_sorted = bboxes[sort_idx] #bboxes_sorted[nt,4(xywh)]
                    cls_sorted = targets[matches, 1:2][sort_idx] #cls_sorted[nt,1]

                    # 写入输出
                    bboxes_sorted[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([cls_sorted, bboxes_sorted], dim=-1) #cls_sorted[nt,1(cls)],bboxes_sorted[nt,4(xywh)]->[nt,5=1(cls)+4(xywh)]

                    # 记录排序后的原始下标
                    matched_idx = matches.nonzero(as_tuple=False).squeeze(1)  # 原始 targets 中的索引
                    sorted_original_idx = matched_idx[sort_idx]
                    global_indices.append(sorted_original_idx)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5])
            sort_indices = torch.cat(global_indices, dim=0)  # [nt]，满足：targets_sorted = targets[sort_indices]
        return out,sort_indices
    # def preprocess_dfl(self, targets, batch_size, scale_tensor, device):
    #     #Preprocesses the target counts and matches with the input batch size to output a tensor.
    #     nl, ne = targets.shape
    #     if nl == 0:
    #         out = torch.zeros(batch_size, 0, ne - 1, device=device)
    #     else:
    #         i = targets[:, 0]  # image index
    #         _, counts = i.unique(return_counts=True)
    #         out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
    #         for j in range(batch_size):
    #             matches = i == j
    #             n = matches.sum()
    #             if n:
    #                 out[j, :n] = targets[matches, 1:]
    #         out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    #     return out

    def preprocess_obb(self, targets, batch_size, scale_tensor, device):
        #Preprocesses the target[nt,7=1(b)+1(cls)+5(xywhr)] counts and matches with the input batch size to output a tensor.
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=device)
            sort_indices = torch.zeros((0))
        else:
            # targets_wh = targets[:, 4:6].prod(-1).argsort(-1)
            # targets = targets[targets_wh.long()]
            i = targets[:, 0]  #i[nt] image index
            _, counts = i.unique(return_counts=True) #counts[b]
            out = torch.zeros(batch_size, counts.max(), 6, device=device) #out[b,nmax,6=1(cls)+5(xywhr)]
            # new_targets = []  # 存放排序后的 targets
            global_indices = []  # 存储排序后的原始下标
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n: #object number of this image b
                    bboxes = targets[matches, 2:]  #targets[nt,7]->bboxes[nt2,5(xywhr)]
                    areas = bboxes[:, 2] * bboxes[:, 3]  # w * h
                    sort_idx = areas.argsort(descending=False)  # 根据面积降序排列

                    # 排序后的 bboxes 和 cls
                    bboxes_sorted = bboxes[sort_idx]
                    cls_sorted = targets[matches, 1:2][sort_idx]

                    # 写入输出
                    bboxes_sorted[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([cls_sorted, bboxes_sorted], dim=-1)

                    # 记录排序后的原始下标
                    matched_idx = matches.nonzero(as_tuple=False).squeeze(1)  # 原始 targets 中的索引
                    sorted_original_idx = matched_idx[sort_idx]
                    global_indices.append(sorted_original_idx)

                    # 保存排序后的 targets，保留原始结构
                    # t_sorted = targets[matches][sort_idx]
                    # new_targets.append(t_sorted)

            # 拼接新的 targets 排序后输出
            # targets_sorted = torch.cat(new_targets, dim=0)  # [nt, 7]，保持按图像分组顺序排布
            sort_indices = torch.cat(global_indices, dim=0)  # [nt]，满足：targets_sorted = targets[sort_indices]
        return out,sort_indices #out[b,nt,6=1(cls)+5(xywhr)]

    
    def make_anchors(self, hw, strides, dtype, device, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        # dtype, device = feats[0].dtype, feats[0].device
        ho, wo = hw
        for i, stride in enumerate(strides):
            # _, _, h, w = feats[i].shape
            h = ho // stride
            w = wo // stride
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            else:
                sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    
    def bbox_decode(self, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return pred_dist
    

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
    
class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
    
class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    