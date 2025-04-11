# gradformer.py

import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Parameter, LeakyReLU, BatchNorm1d

from model.GateGCN import GatedGCNLayer
from model.grad_conv import GPSConv
from utils.process import process_hop
from torch_geometric.nn import global_add_pool, GINConv, GINEConv, global_mean_pool, global_max_pool


class Gradformer(torch.nn.Module):
    def __init__(self, args, node_dim: int, edge_dim: int, num_tasks: int, mpnn: str, pool: str):
        super().__init__()
        self.gamma = args.gamma
        self.slope = args.slope
        self.mpnn = mpnn
        self.pool = pool
        self.pe = args.pe_norm
        self.proj = args.projection
        self.node_method = args.node_method
        self.edge_method = args.edge_method
        self.task = args.task  # 添加任务类型
        self.node_add = Linear(node_dim, args.channels)
        self.pe_add = Linear(args.pe_origin_dim, args.channels)
        self.pe_lin = Linear(args.pe_origin_dim, args.pe_dim)
        self.node_lin = Linear(node_dim, args.channels - args.pe_dim)
        self.no_pe = Linear(node_dim, args.channels)
        self.node_emb = Embedding(node_dim, args.channels - args.pe_dim)
        self.atom_enc = AtomEncoder(args.channels - args.pe_dim - 1)
        self.bond_enc = BondEncoder(args.channels)
        edge_attr_dim = edge_dim  # 使用传入的 edge_dim 参数
        embedding_dim = args.channels  # 与其他层一致，使用 args.channels
        self.edge_emb = Linear(edge_attr_dim, embedding_dim)
        self.pe_norm = BatchNorm1d(args.pe_origin_dim)
        self.hop = Parameter(torch.full((args.nhead, 1, 1), float(args.n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(args.num_layers):
                nn = Sequential(
                    Linear(args.channels, args.channels),
                    ReLU(),
                    Linear(args.channels, args.channels),
                )
                conv = GPSConv(args.channels, GINConv(nn), heads=args.nhead, dropout=args.dropout,
                               attn_dropout=args.attn_dropout, drop_prob=args.drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(args.num_layers):
                nn = Sequential(
                    Linear(args.channels, args.channels),
                    ReLU(),
                    Linear(args.channels, args.channels),
                )
                conv = GPSConv(args.channels, GINEConv(nn), heads=args.nhead, dropout=args.dropout,
                               attn_dropout=args.attn_dropout, drop_prob=args.drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            Lap_pe = True if "hiv" in args.dataset else False
            for _ in range(args.num_layers):
                conv = GPSConv(args.channels, GatedGCNLayer(args.channels, args.channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=args.nhead, dropout=args.dropout, attn_dropout=args.attn_dropout,
                               drop_prob=args.drop_prob)
                self.convs.append(conv)

        # 为地理位置任务修改MLP
        if self.task == 'geo_regression':
            # 输出经纬度和延迟
            self.mlp_geo = Sequential(
                Linear(args.channels, args.channels // 2),
                ReLU(),
                Linear(args.channels // 2, args.channels // 4),
                ReLU(),
                Linear(args.channels // 4, 2),  # 输出经纬度
            )

            # 延迟预测分支
            self.mlp_delay = Sequential(
                Linear(args.channels, args.channels // 2),
                ReLU(),
                Linear(args.channels // 2, args.channels // 4),
                ReLU(),
                Linear(args.channels // 4, 1),  # 输出延迟
            )
        else:
            # 标准MLP，用于其他任务
            self.mlp = Sequential(
                Linear(args.channels, args.channels // 2),
                ReLU(),
                Linear(args.channels // 2, args.channels // 4),
                ReLU(),
                Linear(args.channels // 4, num_tasks),
            )

        self.lin = Linear(args.channels, num_tasks)

    def forward(self, x, pe, edge_index, edge_attr, batch, sph):
        if self.pe:
            pe = self.pe_norm(pe)

        # process node
        if self.node_method == 'add':
            x = self.node_add(x) + self.pe_add(pe)
        if self.node_method == 'linear':
            x = torch.cat((self.node_lin(x), self.pe_lin(pe)), 1)
        if self.node_method == 'embedding':
            x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(pe)), 1)
        if self.node_method == 'ogb':
            x = torch.cat((self.atom_enc(x), self.pe_lin(pe)), 1)
        if self.node_method == 'no_pe':
            x = self.no_pe(x)

        # process edge
        if self.edge_method == 'ogb':
            edge_attr = self.bond_enc(edge_attr)
        if self.edge_method == 'embedding':
            edge_attr = self.edge_emb(edge_attr)

        # get the sph
        sph = process_hop(sph, self.gamma, self.hop, self.slope)

        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr)

        # 对于地理定位任务，我们不需要全局池化
        if self.task == 'geo_regression':
            # 直接返回节点表示，不进行池化
            # 将节点表示传递给MLP，分别预测经纬度和延迟
            geo_out = self.mlp_geo(x)
            delay_out = self.mlp_delay(x)
            return geo_out, delay_out
        else:
            # pooling for graph-level tasks
            if self.pool == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pool == 'add':
                x = global_add_pool(x, batch)
            elif self.pool == 'max':
                x = global_max_pool(x, batch)

            # MLP
            if self.proj == 'mlp':
                x = self.mlp(x)
            else:
                x = self.lin(x)

            return x