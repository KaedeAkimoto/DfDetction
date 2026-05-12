import torch, torchvision
from torch import nn




class AlphaIouLoss(nn.Module):
    def __init__(self, iou_type='alpha_iou', alpha=2.0, scale=1.0, eps=1e-7):
        super().__init__()
        self.iou_type = iou_type
        self.alpha = alpha
        self.eps = eps
        self.scale = scale

    def forward(self, pred_boxes, target_boxes,):
        """
        计算α-IoU损失
        :param pred_boxes: 预测框 [N, 4] (x1, y1, x2, y2)
        :param target_boxes: 真实框 [N, 4]
        :param alpha: 幂指数，控制梯度强度
        :param eps: 数值稳定性阈值
        :return: 标量损失值
        """
        # 计算成对IoU矩阵
        iou_matrix = torchvision.ops.box_iou(pred_boxes, target_boxes)
        
        # 假设一对一匹配，取对角线元素
        if iou_matrix.shape == iou_matrix.shape:
            iou = iou_matrix.diag()
        else:
            # 多对一情况下可选择最大值或其他策略
            iou, _ = iou_matrix.max(dim=1)
        
        # 防止0^alpha异常
        iou = iou.clamp(min=self.eps)
        
        # 应用幂变换
        loss = 1.0 - torch.pow(iou, self.alpha)
        return loss.mean() * self.scale

