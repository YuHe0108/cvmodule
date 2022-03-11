# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
import torch
import torch.nn as nn

from Detection.yolov5.utils.metrics import bbox_iou
from Detection.yolov5.utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyper-parameters

        # 使用BCE作为损失函数，而不是交叉熵，因为一个框可能不止一个目标,  'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # pos_weight参数是正样本损失的权重参数。
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        '''
        由于样本类别不均衡，指定正负样本的类别损失权重系数。
        Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        对标签做平滑,eps=0就代表不做标签平滑,那么默认cp=1,cn=0, 后续对正类别赋值cp，负类别赋值cn
        '''
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # 是否使用 Focal loss, 解决样本类别不均衡的问题
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        ''' 
        模型的最后一层, Detect() module
        每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4, 0.4对应P5。
        如果是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, .02]，可对应 1-5 个输出层 P3-P7 的情况。
        '''
        det = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        '''
        赋值各种参数, gr是用来设置IoU的值在 object-ness loss中做标签的系数, 
        使用代码如下：
            tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
            train.py源码中model.gr=1，也就是说完全使用标签框与预测框的CIoU值来作为该预测框的 object-ness 标签。
        '''
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        '''
        从build_targets函数中构建目标标签，获取标签中的tcls, tbox, indices, anchors
            tcls = [[cls1,cls2,...],[cls1,cls2,...],[cls1,cls2,...]] 每个anchor对应的类别
            tcls.shape = [nl, N]
            tbox = [[[gx1,gy1,gw1,gh1],[gx2,gy2,gw2,gh2],...], # 中心点的偏移量、宽高
    
            indices = [[image indices1,anchor indices1, gridj1, gridi1],
                       [image indices2,anchor indices2, gridj2, gridi2],
                       ...]] # anchor所属batch图像idx、用到了哪个anchor、中心点坐标距左上角的距离
            anchors = [[aw1,ah1],[aw2,ah2],...]		  
        '''

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            p.shape = [nl,bs,na,nx,ny,no]
            nl 为 预测层数，一般为3
            na 为 每层预测层的anchor数，一般为3
            nx,ny 为 grid的 w和 h
            no 为 输出数，为 5 + nc (5:x,y,w,h,obj, nc:分类数)
            
            a:      所有anchor的索引
            b:      标签所属image的索引
            gridy:  标签所在grid的y，在0到ny-1之间
            gridy:  标签所在grid的x，在0到nx-1之间
            
            pi.shape = [bs,na,nx,ny,no]
            tobj.shape = [bs,na,nx,ny]
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                '''
                ps为batch中第b个图像第a个anchor的第gj行第gi列的output
                ps.shape = [N,5+nc], N = a[0].shape，即符合anchor大小的所有标签数
                '''
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                ''' # Regression
                xy的预测范围为-0.5~1.5
                wh的预测范围是0~4倍anchor的w和h，
                原理在代码后讲述。
                '''
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # 通过gr用来设置IoU的值在object-ness loss中做标签的比重, Object-ness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    '''
                    ps[:, 5:].shape = [N,nc],用 self.cn 来填充型为[N,nc]得Tensor。
                    self.cn 通过smooth_BCE平滑标签得到的，使得负样本不再是0，而是0.5 * eps
                    self.cp 通过smooth_BCE平滑标签得到的，使得正样本不再是1，而是1.0 - 0.5 * eps
                    '''
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            '''
            pi[..., 4]所存储的是预测的obj
            self.balance[i]为第i层输出层所占的权重
            将每层的损失乘上权重计算得到obj损失
            '''
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
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        tcls, tbox, indices, anch = [], [], [], []
        ''' 
        Build targets for compute_loss(), input targets(image_idx, class, x, y, w, h)
        na = 3, 表示每个预测层anchors的个数, nt为一个batch中所有标签的数量
        '''
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        '''
        gain是为了最终将坐标所属 grid 坐标限制在坐标系内，不要超出范围,
        其中 7 是为了对应: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny, 其中nx和ny为当前输出层的grid大小。
        '''
        gain = torch.ones(7, device=targets.device)  # normalized to grid-space gain
        # 将 target 复制三份，每份在最后一维配比一个anchor的编号：0,1,2，当前标签所属的anchor索引
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na, nt, 1] (None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],...],
                          [[1],[1],[1],...],
                          [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7]
        targets = [[[image1,class1,x1,y1,w1,h1,0],
                    [image2,class2,x2,y2,w2,h2,0],
                    ...],
                    [[image1,class1,x1,y1,w1,h1,1],
                     [image2,class2,x2,y2,w2,h2,1],
                    ...],
                    [[image1,class1,x1,y1,w1,h1,2],
                     [image2,class2,x2,y2,w2,h2,2],
                    ...]]
        这么做是为了纪录每个label对应的anchor。
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # 目标中心偏移量设置
        '''
        五个偏移：不动、右、下、左、上，坐标原点为图像的左上角，x轴在右边(列)
        [0, 0]代表中间,
        [1, 0] * g = [0.5, 0]代表往左偏移半个grid， 
        [0, 1]*0.5 = [0, 0.5]代表往上偏移半个grid，与后面代码的j,k对应
        [-1, 0] * g = [-0.5, 0]代代表往右偏移半个grid， 
        [0, -1]*0.5 = [0, -0.5]代表往下偏移半个grid，与后面代码的l,m对应
        '''
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):  # num layers
            '''
            原本yaml中加载的anchors.shape = [3 ,6],但 在 yolo.py 的 Detect 中已经通过代码
                a = torch.tensor(anchors).float().view(self.nl, -1, 2)
                self.register_buffer('anchors', a) 
            将anchors进行了reshape。
                self.anchors.shape = [3,3,2]
                anchors.shape = [3,2]
                
            p.shape     = [nl,bs,na,nx,ny,no]
            p[i].shape  = [bs,na,nx,ny,no]
            gain        = [1,1,nx,ny,nx,ny,1]
            '''
            anchors = self.anchors[i]  # 三个不同anchor的尺寸, 此时anchor的大小对应于特征图
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # 为什么是[3, 2, 3, 2], 因为长宽相反, xyxy gain

            # target中的值由于缩放到了0~1之间，因此要与当前特征尺寸相乘, 表明当前特征尺寸下，边框的大小。Match targets to anchors
            t = targets * gain
            if nt:  # 如果存在 num_target
                ''' 
                计算 gt 和 anchor 的匹配程度，将预测的边框 / anchor 的长宽比值, 相当于对预测值进行长宽不同尺度的缩放， Matches
                    t[:, :, 4:6].shape = [na,nt,2] = [3,nt,2], 存放的是标签的w和h
                    anchor[:,None]  = [3,1,2]
                    r.shape         = [3,nt,2], 存放的是标签和当前层anchor的长宽比
                '''
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                '''
                判断 r 和 1/r 与 hyp['anchor_t'] 的大小关系，将一些与gt相差较大的 anchor 过滤掉
                torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2]
                再max(2)求出同一标签中 宽比 和 长比 较大的一个，shape = [2,3,nt],之所以第一个维度变成2，
                因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                    torch.return_types.max(values=tensor([...]), indices=tensor([...]))
                所以还需要加上索引0获取values，
                    torch.max(r, 1. / r).max(2)[0].shape = [3,nt],
                将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                    j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                    j.shape = [3,nt]
                '''
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                '''
                t.shape = [na,nt,7] = [3,nt,7]
                j.shape = [3, nt]
                假设j中有NTrue个True值，则
                    t[j].shape = [NTrue,7]
                返回的是 na*nt 的标签中，所有属于当前层anchor的标签。
                '''
                t = t[j]  # 将满足条件的 anchor(每个点3个) 筛选出来 -  filter

                '''
                使用 Offsets 扩充 targets 的数量，将比较targets附近的4个点，选取最近的2个点作为新targets中心，
                新targets的w、h使用与原targets一致，只是中心点坐标的不同。
                    t.shape = [NTrue,7] 
                    7:  image(当前的坐标信息归于batch中的哪一张图),class,x,y,h,w,ai(第几个anchor)
                    gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的记录
                    gxi.shape = [NTrue,2] 存放的是w-x,h-y,相当于测量坐标到坐标系右边框和下边框的距离
                '''
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                '''
                因为grid单位为1，共 nx * ny个gird
                    gxy % 1相当于求得标签在第gxy.long()个grid中以grid左上角为原点的相对坐标，
                    gxi % 1相当于求得标签在第gxy.long()个grid中以grid右下角为原点的相对坐标，
                下面这两行代码作用在于
                筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签          
                    j.shape = [NTrue], j = [bool,bool,...]
                    k.shape = [NTrue], k = [bool,bool,...]
                    l.shape = [NTrue], l = [bool,bool,...]
                    m.shape = [NTrue], m = [bool,bool,...]
                '''
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                '''
                j.shape = [5, NTrue]  第一列表示不偏移，保留所有的直线筛选下的anchor
                t.repeat之后shape为[5, NTrue, 7], 
                通过索引 j 后 t.shape = [NOff,7], NOff表示NTrue + (j,k,l,m中True的总数量)
                
                torch.zeros_like(gxy)[None].shape = [1, NTrue, 2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理。
                '''
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 为什么是5：表示了五个方向的偏移
                t = t.repeat((5, 1, 1))[j]  # 通过中心点偏移，增加anchor, 筛选后 t 的数量是原来t的3倍。
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 表示了每个anchor是否经过了偏移
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 将所有 targets 中心点坐标进行偏移, 并取整
            gi, gj = gij.T  # 每个anchor的横纵坐标: grid xy indices

            # Append
            '''
            a: 所有anchor的索引 shape = [NOff]
            b: 标签所属image的索引 shape = [NOff]
            gj.clamp_(0, gain[3] - 1)将标签所在grid的y限定在0到ny-1之间
            gi.clamp_(0, gain[2] - 1)将标签所在grid的x限定在0到nx-1之间
            indices = [image, anchor, gridy, gridx] 最终shape = [nl, 4, NOff]
            tbox：存放的是 <中心点标签> 在所在grid内的相对坐标，∈[0,1] 最终shape = [nl, NOff]
            anch：存放的是anchors 最终shape = [nl,NOff,2]
            tcls：存放的是标签的分类 最终shape = [nl,NOff]
            '''
            a = t[:, 6].long()  # 每个 anchor 归属于哪个候选框(三个不同尺寸的候选框): anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # 用到了哪些anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch
