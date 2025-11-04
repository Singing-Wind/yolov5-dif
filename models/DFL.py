import torch
import torch.nn.functional as F

def dfl_pred(logits, keys): #logits[B,C],keys[C] -> pred_val[B]
    # 对logits进行softmax得到概率分布
    probs = F.softmax(logits, dim=1)  # logits[B, C]->probs[B, C]
    # 目标分布在 idl, idr 两个位置有非零概率
    # p_idl = probs[torch.arange(B), idl] # probs[[B], idl[B]]->p_idl[B]
    # p_idr = probs[torch.arange(B), idr] # probs[[B], idr[B]]->p_idr[B]
    # 根据概率分布计算预测值
    pred_val = (probs * keys.unsqueeze(0)).sum(dim=1)  #probs[B][C] * keys[C]->[B][C] .sum(1)->pred_val[B]
    return pred_val #pred_val[B]

def find_left_right(gt, keys):
    # gt:   [B] (浮点数, 可能分布在 keys[0] ~ keys[-1] 之间)
    # keys: [C] 已从小到大排序好的关键值(长度=C)
    # 返回: idl, idr (形状均为 [B])，满足 keys[idl] <= gt < keys[idr]。
    #       若 gt 超出 keys 区间，也可做 clamp 或其他处理方式。
    B = gt.shape[0]
    C = keys.shape[0]

    # 1) searchsorted 会返回: 在 keys 中可以插入 gt 的位置 pos，
    #    使得 keys[pos-1] <= gt < keys[pos] (right=False).
    #    pos 的取值范围是 [0, C], 可能 =0 (gt<keys[0]) 或 =C (gt>=keys[-1])
    pos = torch.searchsorted(keys, gt, right=False)  # [B], int64

    # 2) 我们要把 pos clamp 到 [1, C-1]，这样 idl=pos-1, idr=pos 就不会越界
    pos = torch.clamp(pos, 1, C-1)

    # 3) 得到 idl, idr
    idr = pos
    idl = pos - 1

    return idl, idr

def dfl_loss(logits, gt, keys, key_flag=0):
    # logits[B, C] 模型对某个连续值(如xc)的C个bin输出的logits
    # gt[B] 真实值(连续), 值域[0,1]
    # keys[C] 从小到大分布的关键值点,一般为linspace(0,1,C)

    # 返回: CE_loss, L1_loss
    B, C = logits.shape
    device = logits.device

    # 根据概率分布logits[B, C]和keys[C]计算预测值pred_val[B]
    pred_val = dfl_pred(logits, keys) #logits[B, C],keys[C]->pred_val[B]
    # L1 loss
    L1smooth_loss = F.smooth_l1_loss(pred_val, gt) #pred_val[B] ~ gt[B]
    
    # 找到idl和idr
    # gt对应的bin索引(左侧bin)
    
    if key_flag==0:#only for [0,1]
      idl = torch.clamp((gt*(C-1)).floor().long(), 0, C-2) # 确保 idr 不溢出
      idr = idl + 1 #idl[B] idr[B]
    else:#any key
      idl, idr = find_left_right(gt, keys) #gt[B]+keys[C] -> idl[B] idr[B]   
    left_keys = keys[idl]   # [B]
    right_keys = keys[idr]  # [B]
    torch.all(left_keys<=gt) and torch.all(gt<=right_keys)

    # 计算权重
    w_idl = (right_keys - gt) / (right_keys - left_keys) #w_idl[B]
    w_idr = (gt - left_keys) / (right_keys - left_keys) #w_idr[B]

    # CE = - [ w_idl * log(p_idl) + w_idr * log(p_idr) ]
    # 注意必须用log_softmax对logits再取，需要对logits取log_softmax以稳定
    log_probs = F.log_softmax(logits, dim=1) #logits[B, C]->log_probs[B, C]
    log_p_idl = log_probs[torch.arange(B), idl] #log_probs[[B],idl[B]]->log_p_idl[B]
    log_p_idr = log_probs[torch.arange(B), idr] #log_probs[[B],idr[B]]->log_p_idr[B]

    # CE loss
    CE_loss = -(w_idl * log_p_idl + w_idr * log_p_idr).mean() #[B].mean()->CE_loss

    return CE_loss, L1smooth_loss, F.l1_loss(pred_val, gt)

def dfl_loss_modified(logits: torch.Tensor,
                      gt: torch.Tensor,
                      keys: torch.Tensor,
                      key_flag=0,
                      K: int = 2
                      ):
    # 与原始 dfl_loss 类似，但不再只使用左右两个关键值，而是使用以“真实值所在 bin”为中心的 2K+1 个关键值进行加权的交叉熵。
    # logits: [B, C]，模型对某个连续值(如 x_c)离散化为 C 个 bin 后的 logits
    # gt:     [B]，真实值(连续)，取值范围 [0,1]
    # keys:   [C]，从小到大分布的关键值点，一般为 linspace(0,1,C)
    # K:      选取以“真实值所在 bin”为中心，左右各取 K 个，共 2K+1 个邻近 bin   
    # return: 
    #   CE_loss   (2K+1 个关键值加权得到的交叉熵损失)
    #   L1_loss   (smooth L1 损失)
    #   L1_loss_2 (普通的 L1 损失，和上面 smooth L1 可二选一)

    B, C = logits.shape
    device = logits.device

    # ========== 1. 计算预测值 pred_val，并计算 L1 损失 ==========
    pred_val = dfl_pred(logits, keys)  # logits[B, C], keys[C] -> pred_val[B]
    L1smooth_loss = F.smooth_l1_loss(pred_val, gt)  # smooth L1
    L1_loss_2 = F.l1_loss(pred_val, gt)             # 普通 L1

    # ========== 2. 计算 2K+1 个关键值的加权交叉熵损失 ==========
    # 2.1 对 logits 做 log_softmax
    log_probs = F.log_softmax(logits, dim=1)  # [B, C]

    # 2.2 找到“真实值对应的 bin 索引” gt_bin
    if key_flag==0:
        idl = torch.clamp((gt*(C - 1)).floor().long(), min=0, max=C-2)  # 确保不溢出
        idr = idl + 1  # 右侧 bin 索引 [B]
    else:
        idl, idr = find_left_right(gt, keys) #gt[B]+keys[C] -> idl[B] idr[B]
    left_keys = keys[idl]   # [B]
    right_keys = keys[idr]  # [B]
    torch.all(left_keys<=gt) and torch.all(gt<=right_keys)

    # 2.3 为每个样本取以 idl 为中心、左右各 K 个 bin，共 2K+1 个
    #     构造 idxs: [B, 2K+1]
    #     以 idl 为中心
    offsets = torch.arange(-K, K+1, device=device).unsqueeze(0)  # [1, 2K+1]
    idxs = idl.unsqueeze(-1) + offsets                  # [B, 2K+1]
    idxs = torch.clamp(idxs, 0, C-1)                             # 防止越界

    # 2.4 从 log_probs 中取出对应的 log(p_i)
    #     log_probs: [B, C]
    #     idxs:      [B, 2K+1]
    #     log_p_selected: [B, 2K+1]
    log_p_selected = log_probs.gather(dim=1, index=idxs)

    # 2.5 计算距离 + 距离的平方的倒数(权重)
    #     selected_keys: [B, 2K+1]
    #     dist:          [B, 2K+1]
    #     w:             [B, 2K+1]
    #     最终要做归一化，保证 w 之和为 1
    eps = 1e-12
    selected_keys = keys[idxs]  # [B, 2K+1]
    dist = (selected_keys - gt.unsqueeze(-1)).abs()  # [B, 2K+1]
    w = 1.0 / (dist**2 + eps)  # 距离平方的倒数作为权重
    w_sum = w.sum(dim=1, keepdim=True)  # [B, 1]
    w_normalized = w / w_sum           # [B, 2K+1]

    # 2.6 计算最终的加权交叉熵
    #     CE_sample: [B]
    CE_sample = -(w_normalized * log_p_selected).sum(dim=1)
    CE_loss = CE_sample.mean()

    return CE_loss, L1smooth_loss, L1_loss_2