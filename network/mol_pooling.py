# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: mol_pooling.py
@time: 6/10/21 4:37 PM
@desc:
"""
import torch
import torch.nn as nn


class AtomPooling(nn.Module):
    def __init__(self, kernel_size: int = 4, stride: int = 4):
        super(AtomPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_op = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0, return_indices=True)

    def forward(self,
                pos: "(bs, n, 3)",
                node_fea: "(bs, n, channel)",
                res_idx: "(bs, n)",
                node_mask: "(bs, n)"):
        """
        Return:
            vertices_pool: (bs, pool_atom_num, 3),
            feature_map_pool: (bs, pool_atom_num, channel_num)
        """
        channel_num = node_fea.size()[2]
        fea_T = node_fea.transpose(1, 2)  # [bs, c, a_n]
        p_fea_T, index = self.pool_op(fea_T)  # [bs, c, a_n_pooled], [bs, c, a_n_pooled]
        """
        torch.gather(input, dim, index, out=None) → Tensor
        
            Gathers values along an axis specified by dim.
            For a 3-D tensor the output is specified by:
                out[i][j][k] = input[index[i][j][k]] [j][k]  # dim=0
                out[i][j][k] = input[i] [index[i][j][k]] [k]  # dim=1
                out[i][j][k] = input[i][j] [index[i][j][k]]  # dim=2

            Parameters:	
                input (Tensor) – The source tensor
                dim (int) – The axis along which to index
                index (LongTensor) – The indices of elements to gather
                out (Tensor, optional) – Destination tensor

            Example:
                >>> t = torch.Tensor([[1,2],[3,4]])
                >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
                1  1
                4  3
                [torch.FloatTensor of size 2x2]
        """
        pos_rep_T = pos.unsqueeze(2).repeat(1, 1, channel_num, 1).transpose(1, 2)  # [bs, c, n, 3]
        p_pos_T = torch.gather(pos_rep_T, dim=2, index=index.unsqueeze(-1). repeat(1, 1, 1, 3))  # [bs, c, n, 3]
        # center_pos_T = torch.sum(p_pos_T, dim=1) / channel_num  # [bs, n, 3]

        ridx_rep_T = res_idx.unsqueeze(2).repeat(1, 1, channel_num).transpose(1, 2)  # [bs, c, n]
        p_ridx_T = torch.gather(ridx_rep_T, dim=2, index=index)
        round_ridx_T = torch.div(torch.sum(p_ridx_T, dim=1), channel_num)  # [bs, n]
        round_ridx_T = round_ridx_T.round()

        # mask_rep_T = node_mask.unsqueeze(2).repeat(1, 1, channel_num).transpose(1, 2)  # [bs, c, n]
        # p_mask_T = torch.gather(mask_rep_T, dim=2, index=index)
        # It's impossible to select a 0 elements in MaxPooling op
        p_mask = self.pool_op(node_mask.unsqueeze(1))[0].squeeze()  # [bs, n]

        # if self.channel_first:
        #     return center_pos_T, p_fea_T, round_ridx_T, p_mask
        # else:
        #     return center_pos_T, p_fea_T.transpose(1, 2), round_ridx_T, p_mask

        return p_pos_T, p_fea_T, round_ridx_T, p_mask


class AbstractPooling(nn.Module):
    def __init__(self, kernel_size: int = 4, stride: int = 4):
        super(AbstractPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_op = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0, return_indices=True)

    def forward(self,
                pos: "(bs, n, 3)",
                node_fea: "(bs, channel, n, dim)",
                res_idx: "(bs, n)",
                node_mask: "(bs, n)"):
        """
        Return:
            vertices_pool: (bs, pool_atom_num, 3),
            feature_map_pool: (bs, pool_atom_num, channel_num)
        """
        channel_num = node_fea.size()[1]
        # TODO: deal with dim in pooling or in convolution
        p_fea_T, index = self.pool_op(node_fea.squeeze())  # [bs, c, n_pooled, 1], [bs, c, n_pooled]
        p_fea_T = p_fea_T.unsqueeze(-1)
        """
        torch.gather(input, dim, index, out=None) → Tensor

            Gathers values along an axis specified by dim.
            For a 3-D tensor the output is specified by:
                out[i][j][k] = input[index[i][j][k]] [j][k]  # dim=0
                out[i][j][k] = input[i] [index[i][j][k]] [k]  # dim=1
                out[i][j][k] = input[i][j] [index[i][j][k]]  # dim=2

            Parameters:	
                input (Tensor) – The source tensor
                dim (int) – The axis along which to index
                index (LongTensor) – The indices of elements to gather
                out (Tensor, optional) – Destination tensor

            Example:
                >>> t = torch.Tensor([[1,2],[3,4]])
                >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
                1  1
                4  3
                [torch.FloatTensor of size 2x2]
        """
        pos_rep_T = pos.unsqueeze(1).repeat(1, channel_num, 1, 1)  # [bs, c, n, 3]
        p_pos_T = torch.gather(pos_rep_T, dim=2, index=index.unsqueeze(-1).repeat(1, 1, 1, 3))  # [bs, c, n, 3]
        # center_pos_T = torch.sum(p_pos_T, dim=1) / self.kernel_size  # [bs, n, 3]

        ridx_rep_T = res_idx.unsqueeze(1).repeat(1, channel_num, 1)  # [bs, c, n]
        p_ridx_T = torch.gather(ridx_rep_T, dim=2, index=index)
        main_ridx_T = torch.div(torch.sum(p_ridx_T, dim=1), channel_num)
        main_ridx_T.ceil_()

        # mask_rep_T = node_mask.unsqueeze(2).repeat(1, 1, channel_num).transpose(1, 2)  # [bs, c, n]
        # p_mask_T = torch.gather(mask_rep_T, dim=2, index=index)
        # It's impossible to select a 0 elements in MaxPooling op
        p_mask = self.pool_op(node_mask.unsqueeze(1))[0].squeeze()  # [bs, n]

        return p_pos_T, p_fea_T, main_ridx_T, p_mask

