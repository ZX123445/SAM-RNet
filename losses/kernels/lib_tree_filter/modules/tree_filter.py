import torch
from torch import nn
import torch.distributed as dist

from ..functions.mst import mst
from ..functions.bfs import bfs
from ..functions.refine import refine

class MinimumSpanningTree(nn.Module): #构建给定特征图的最小生成树
    def __init__(self, distance_func): #distance_fune：距离函数
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func

    """
    定义一个静态方法， 构建特征图fm的索引矩阵， 表示图中节点之间的连接关系
    """
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width

        #构建行和列的索引树， 并将它们合并并扩展到批次大小， 得到最终的索引矩阵
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2),
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm): #计算特征图相邻节点之间的权重
        batch = fm.shape[0]
        #计算水平和垂直方向上相邻节点之间的距离
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])

        #将距离重塑并合并， 再+1 作为最终的权重
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1) + 1
        return weight

    def _build_label_weight(self, fm): #计算带有标签的特征图中相邻节点之间的加权距离
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        diff_weight = torch.cat([weight_row, weight_col], dim=1)

        weight_row = (fm[:, :, :-1, :] + fm[:, :, 1:, :]).sum(1)
        weight_col = (fm[:, :, :, :-1] + fm[:, :, :, 1:]).sum(1)
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        labeled_weight = torch.cat([weight_row, weight_col], dim=1)

        weight = diff_weight * labeled_weight
        return weight

    def forward(self, guide_in, label=None): #guide_in和可选择的标签
        with torch.no_grad():#不计算梯度的条件下，构建索引矩阵和特征权重
            index = self._build_matrix_index(guide_in)
            weight = self._build_feature_weight(guide_in)
            if label is not None: #计算标签的权重
                label_weight = self._build_label_weight(label)
                label_idx = (label_weight > 0)
                weight[label_idx] = torch.sigmoid(weight[label_idx])

            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3]) #最小生成树
        return tree


class TreeFilter2D(nn.Module):#
    def __init__(self, groups=1, sigma=0.02, distance_func=None, enable_log=False):
        super(TreeFilter2D, self).__init__()
        self.groups = groups
        self.enable_log = enable_log
        if distance_func is None:
            self.distance_func = self.norm2_distance
        else:
            self.distance_func = distance_func

        self.sigma = sigma #标准差

    @staticmethod
    def norm2_distance(fm_ref, fm_tar):  #L2距离
        diff = fm_ref - fm_tar
        weight = (diff * diff).sum(dim=1)
        return weight

    @staticmethod
    def batch_index_opr(data, index): #根据索引从数据中提取对应元素
        with torch.no_grad():
            channel = data.shape[1]
            index = index.unsqueeze(1).expand(-1, channel, -1).long()
        data = torch.gather(data, 2, index)
        return data



    def build_edge_weight(self, fm, sorted_index, sorted_parent, low_tree):
        #根据输入的特征图（fm）、排序索引（sorted_index）、
        #排序父节点（sorted_parent）和是否考虑低树结构（low_tree），计算边缘权重。
        batch   = fm.shape[0]
        channel = fm.shape[1]
        vertex  = fm.shape[2] * fm.shape[3]

        fm = fm.reshape([batch, channel, -1])

        #根据 sorted_index 和 sorted_parent 对特征图进行索引操作
        fm_source = self.batch_index_opr(fm, sorted_index)
        fm_target = self.batch_index_opr(fm_source, sorted_parent)
        fm_source = fm_source.reshape([-1, channel // self.groups, vertex])
        fm_target = fm_target.reshape([-1, channel // self.groups, vertex])

        #计算源特征图和目标特征图之间的距离， 得到边缘权重
        edge_weight = self.distance_func(fm_source, fm_target)

        if low_tree: #根据low_tree的值, 对边缘权重进行缩放
            edge_weight = torch.exp(-edge_weight / self.sigma)   # default  voc2012: 0.02  citys: 0.002
        else:
            edge_weight = torch.exp(-edge_weight)
        return edge_weight


    #将输入的特征图feature_in根据分组数量self.groups分组，并处理树结构顺序
    def split_group(self, feature_in, *tree_orders): #*tree_orders：一个或多个树结构顺序， 用于分组后的特征图
        feature_in = feature_in.reshape(feature_in.shape[0] * self.groups,
                                        feature_in.shape[1] // self.groups,
                                        -1)

        returns = [feature_in.contiguous()]

        for order in tree_orders:  #对每个树结构顺序order, 进行扩展和重塑
            order = order.unsqueeze(1).expand(order.shape[0], self.groups, *order.shape[1:])
            order = order.reshape(-1, *order.shape[2:])
            returns.append(order.contiguous())

        return tuple(returns)


    def print_info(self, edge_weight): #打印权重的统计信息（均值， 标准差， 最大值， 最小值）
        edge_weight = edge_weight.clone()
        info = torch.stack([edge_weight.mean(), edge_weight.std(), edge_weight.max(), edge_weight.min()])

        if self.training and dist.is_initialized():
            #如果在训练模式下且分布式环境已初始化，使用dist_all_reduce进行跨节点聚合
            dist.all_reduce(info / dist.get_world_size())
            info_str = (float(x) for x in info)
            if dist.get_rank() == 0:
                print('Mean:{0:.4f}, Std:{1:.4f}, Max:{2:.4f}, Min:{3:.4f}'.format(*info_str))
        else:
            info_str = [float(x) for x in info]
            print('Mean:{0:.4f}, Std:{1:.4f}, Max:{2:.4f}, Min:{3:.4f}'.format(*info_str))



    def forward(self, feature_in, embed_in, tree, low_tree=True):
        ori_shape = feature_in.shape   #输入特征图的原始形状
        sorted_index, sorted_parent, sorted_child = bfs(tree, 4) #

        edge_weight = self.build_edge_weight(embed_in, sorted_index, sorted_parent, low_tree)

        self.enable_log = False
        with torch.no_grad():
            if self.enable_log:
                self.print_info(edge_weight)

        feature_in, sorted_index, sorted_parent, sorted_child = \
            self.split_group(feature_in, sorted_index, sorted_parent, sorted_child)

        feature_out = refine(feature_in, edge_weight, sorted_index,
                             sorted_parent, sorted_child, low_tree)

        feature_out = feature_out.reshape(ori_shape)
        return feature_out