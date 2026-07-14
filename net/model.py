import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from equiformer_pytorch import Equiformer
from net.GIN import GNN_node
from net.MLP import MLP
from net.Trigonometry import Transition, TriangleProteinToCompound_v3, TriangleSelfAttentionRowWise


# =================================================================
# 核心组件 1: 跨模态瓶颈注意力 (MBT / DBT)
# =================================================================

class BottleneckAttention(nn.Module):
    """
    基于解耦思想的跨模态融合模块
    提取共有特征 Shared 和模态独有特征 Specific
    """

    def __init__(self, input_dim, proj_dim=512, num_bottlenecks=8, heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_bottlenecks = num_bottlenecks

        # 1. 模态投影层
        self.proj_2d = nn.Linear(input_dim, proj_dim)
        self.proj_3d = nn.Linear(input_dim, proj_dim)

        # 2. 共有特征提取器：瓶颈 tokens
        self.shared_bottlenecks = nn.Parameter(
            torch.randn(1, num_bottlenecks, proj_dim)
        )

        self.attn_2d_to_shared = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=heads,
            batch_first=True
        )
        self.attn_3d_to_shared = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=heads,
            batch_first=True
        )
        self.attn_shared_self = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=heads,
            batch_first=True
        )

        # 3. 独有特征提取器
        self.specific_2d_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

        self.specific_3d_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

        self.norm_shared = nn.LayerNorm(proj_dim)
        self.norm_spec2d = nn.LayerNorm(proj_dim)
        self.norm_spec3d = nn.LayerNorm(proj_dim)

        # shared: num_bottlenecks * proj_dim
        # spec2d: proj_dim
        # spec3d: proj_dim
        fusion_in_dim = num_bottlenecks * proj_dim + 2 * proj_dim
        fusion_out_dim = num_bottlenecks * proj_dim

        self.fusion_fc = nn.Linear(fusion_in_dim, fusion_out_dim)

    def forward(self, feat_2d, feat_3d):
        feat_2d = torch.nan_to_num(feat_2d, nan=0.0, posinf=0.0, neginf=0.0)
        feat_3d = torch.nan_to_num(feat_3d, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size = feat_2d.shape[0]
        scaling = self.proj_dim ** 0.5

        # 1. 投影到统一潜在空间
        x_2d = F.gelu(self.proj_2d(feat_2d))
        x_3d = F.gelu(self.proj_3d(feat_3d))

        # 2. 共有特征提取
        b = self.shared_bottlenecks.expand(batch_size, -1, -1)

        b_from_2d, _ = self.attn_2d_to_shared(
            b,
            x_2d.unsqueeze(1) / scaling,
            x_2d.unsqueeze(1)
        )

        b_from_3d, _ = self.attn_3d_to_shared(
            b,
            x_3d.unsqueeze(1) / scaling,
            x_3d.unsqueeze(1)
        )

        b = self.norm_shared(
            b
            + torch.nan_to_num(b_from_2d, nan=0.0, posinf=0.0, neginf=0.0)
            + torch.nan_to_num(b_from_3d, nan=0.0, posinf=0.0, neginf=0.0)
        )

        b_self, _ = self.attn_shared_self(
            b,
            b / scaling,
            b
        )

        shared_feat = self.norm_shared(
            b + torch.nan_to_num(b_self, nan=0.0, posinf=0.0, neginf=0.0)
        )

        shared_feat_flat = shared_feat.reshape(batch_size, -1)
        shared_mean = shared_feat.mean(dim=1)

        # 3. 独有特征提取
        spec_2d = self.norm_spec2d(
            self.specific_2d_proj(x_2d - shared_mean)
        )

        spec_3d = self.norm_spec3d(
            self.specific_3d_proj(x_3d - shared_mean)
        )

        # 4. 最终融合
        concat_feat = torch.cat(
            [shared_feat_flat, spec_2d, spec_3d],
            dim=1
        )

        final_fused_feat = self.fusion_fc(concat_feat)

        return final_fused_feat, shared_mean, spec_2d, spec_3d


# =================================================================
# 核心组件 2: Top-K MoE
# =================================================================

class TopKMoE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        num_experts=4,
        k=2,
        dropout=0.4,
        noisy_gating_std=0.1
    ):
        super(TopKMoE, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = int(num_experts)
        self.k = max(1, min(int(k), self.num_experts))
        self.noisy_gating_std = noisy_gating_std

        # 专家组
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            for _ in range(self.num_experts)
        ])

        # 输入是 kroneck + 3 个物理路由特征
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim + 3, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.num_experts)
        )

        # 温度缩放
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, gate_input):
        gate_logits = self.gating_net(gate_input)

        if self.training and self.noisy_gating_std > 0:
            noise = torch.randn_like(gate_logits) * self.noisy_gating_std
            gate_logits = gate_logits + noise

        gate_probs = F.softmax(gate_logits, dim=1)

        topk_logits, topk_indices = torch.topk(
            gate_logits,
            self.k,
            dim=1
        )

        topk_weights = F.softmax(topk_logits, dim=1)

        all_expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )

        batch_size = x.shape[0]

        gather_indices = topk_indices.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_classes
        )

        selected_outputs = torch.gather(
            all_expert_outputs,
            1,
            gather_indices
        )

        expert_probs = F.softmax(selected_outputs, dim=-1)

        final_logits = torch.sum(
            topk_weights.unsqueeze(-1) * selected_outputs,
            dim=1
        )

        temperature = torch.clamp(self.temperature, min=0.05, max=10.0)
        final_logits = final_logits / temperature

        # Epistemic uncertainty：专家分歧
        if self.k > 1:
            epistemic_unc = torch.var(
                expert_probs,
                dim=1
            ).mean(dim=1, keepdim=True)
        else:
            epistemic_unc = torch.zeros(
                batch_size,
                1,
                device=x.device,
                dtype=x.dtype
            )

        # Aleatoric uncertainty：预测熵
        final_probs = F.softmax(final_logits, dim=-1)
        aleatoric_unc = -torch.sum(
            final_probs * torch.log(final_probs + 1e-9),
            dim=1,
            keepdim=True
        )

        return final_logits, aleatoric_unc, epistemic_unc, gate_probs


# =================================================================
# 主模型: GerNA-Cert
# =================================================================

class GerNA(nn.Module):
    def __init__(
        self,
        params,
        input_dim_rna=640,
        input_dim_mol=55,
        trigonometry=True,
        mol_graph=True,
        coors=True,
        rna_repre=True,
        rna_graph=True,
        coors_3_bead=True,
        uncertainty=False,
        num_classes=2,
        hparams=None,
        use_mbt=True
    ):
        super(GerNA, self).__init__()

        # =========================================================
        # 1. 统一管理超参数
        # =========================================================
        default_hparams = {
            "MBT_Num_Bottlenecks": 8,
            "MBT_Proj_Dim": 512,
            "MBT_Residual_Alpha": 0.5,
            "Residual_Alpha": 0.5,
            "Dropout_Rate": 0.4,
            "num_experts": 4,
            "top_k": 2
        }

        if hparams is not None:
            default_hparams.update(hparams)

        self.hparams = default_hparams

        self.use_mbt = use_mbt
        self.mol_graph = mol_graph
        self.coors = coors
        self.rna_graph = rna_graph
        self.rna_repre = rna_repre
        self.coors_3_bead = coors_3_bead
        self.uncertainty = uncertainty
        self.num_classes = num_classes
        self.trigonometry = trigonometry

        GNN_depth, DMA_depth, hidden_size1, hidden_size2 = params
        self.DMA_depth = DMA_depth

        # =========================================================
        # 2. 3D Equiformer 编码器
        # =========================================================
        if self.coors:
            self.equi_mol = Equiformer(
                num_tokens=10,
                dim=(16, 4, 2),
                dim_head=(10, 10, 10),
                heads=(2, 2, 2),
                num_degrees=3,
                depth=2,
                attend_self=True,
                reversible=False
            )

            self.equi_rna = Equiformer(
                num_tokens=5 if coors_3_bead else 9,
                dim=(16, 4, 2),
                dim_head=(10, 10, 10),
                heads=(2, 2, 2),
                num_degrees=3,
                depth=2,
                attend_self=True,
                reversible=False
            )

        # =========================================================
        # 3. RNA 2D 图编码器
        # =========================================================
        if self.rna_graph:
            self.GCN_rna = GNN_node(
                input_dim_rna,
                GNN_depth,
                hidden_size1,
                edge_attr_option=False,
                gnn_type='gcn'
            )

            self.rna_2d_final = nn.Linear(
                hidden_size1,
                hidden_size2
            )

        # =========================================================
        # 4. Ligand 2D 图编码器 + 2D DMA-GRU
        # =========================================================
        if self.mol_graph:
            self.GCN_mol = GNN_node(
                input_dim_mol,
                GNN_depth,
                hidden_size1,
                gnn_type='gcn'
            )

            self.mol_2d_final = nn.Linear(
                hidden_size1,
                hidden_size2
            )

            self.mc1 = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.mp1 = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hc0 = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hp0 = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hc1 = nn.ModuleList([
                nn.Linear(hidden_size2, 1)
                for _ in range(DMA_depth)
            ])

            self.hp1 = nn.ModuleList([
                nn.Linear(hidden_size2, 1)
                for _ in range(DMA_depth)
            ])

            self.m_to_r_transform = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.r_to_m_transform = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.GRU_dma = nn.GRUCell(
                hidden_size2,
                hidden_size2
            )

        # =========================================================
        # 5. RNA 1D 表征编码器
        # =========================================================
        if self.rna_repre:
            self.mlp_rna = MLP(
                input_dim_rna,
                hidden_size1
            )

            self.rna_1d_final = nn.Linear(
                hidden_size1,
                hidden_size2
            )

        # =========================================================
        # 6. 3D DMA-GRU
        # =========================================================
        if self.coors:
            self.rna_3d_final = nn.Linear(
                16,
                hidden_size2
            )

            self.mol_3d_final = nn.Linear(
                16,
                hidden_size2
            )

            self.mc1_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.mp1_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hc0_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hp0_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.hc1_3d = nn.ModuleList([
                nn.Linear(hidden_size2, 1)
                for _ in range(DMA_depth)
            ])

            self.hp1_3d = nn.ModuleList([
                nn.Linear(hidden_size2, 1)
                for _ in range(DMA_depth)
            ])

            self.m_to_r_transform_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.r_to_m_transform_3d = nn.ModuleList([
                nn.Linear(hidden_size2, hidden_size2)
                for _ in range(DMA_depth)
            ])

            self.GRU_dma_3d = nn.GRUCell(
                hidden_size2,
                hidden_size2
            )

        # =========================================================
        # 7. MBT / DBT 融合层
        # =========================================================
        if self.mol_graph and self.coors:
            kroneck_dim = hidden_size2 * hidden_size2

            self.num_bottlenecks = int(
                self.hparams.get("MBT_Num_Bottlenecks", 8)
            )

            self.proj_dim = int(
                self.hparams.get("MBT_Proj_Dim", 512)
            )

            self.mbt_alpha = float(
                self.hparams.get(
                    "MBT_Residual_Alpha",
                    self.hparams.get("Residual_Alpha", 0.5)
                )
            )

            fusion_out_dim = self.num_bottlenecks * self.proj_dim

            if self.use_mbt:
                self.fusion_module = BottleneckAttention(
                    input_dim=kroneck_dim,
                    proj_dim=self.proj_dim,
                    num_bottlenecks=self.num_bottlenecks
                )
            else:
                self.concat_proj = nn.Linear(
                    2 * kroneck_dim,
                    fusion_out_dim
                )

            self.res_proj = nn.Linear(
                kroneck_dim,
                fusion_out_dim
            )

            in_features = fusion_out_dim

        else:
            self.num_bottlenecks = int(
                self.hparams.get("MBT_Num_Bottlenecks", 8)
            )
            self.proj_dim = int(
                self.hparams.get("MBT_Proj_Dim", 512)
            )
            self.mbt_alpha = float(
                self.hparams.get(
                    "MBT_Residual_Alpha",
                    self.hparams.get("Residual_Alpha", 0.5)
                )
            )

            in_features = hidden_size2 * hidden_size2

        # =========================================================
        # 8. 输出层：MoE 或普通 MLP
        # =========================================================
        if uncertainty:
            self.W_out = TopKMoE(
                input_dim=in_features,
                num_classes=num_classes,
                num_experts=int(self.hparams.get("num_experts", 4)),
                k=int(self.hparams.get("top_k", 2)),
                dropout=float(self.hparams.get("Dropout_Rate", 0.4))
            )
        else:
            self.W_out = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1)
            )

    # =============================================================
    # mask softmax
    # =============================================================
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)

        if len(mask.shape) < len(a.shape):
            a_exp = a_exp * mask.unsqueeze(-1)
        else:
            a_exp = a_exp * mask

        return a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)

    # =============================================================
    # 2D DMA-GRU
    # =============================================================
    def dma_gru_2d(
        self,
        batch_size,
        rna_fea,
        mol_fea,
        rna_mask,
        mol_mask,
        pair_2d
    ):
        m = (
            torch.sum(mol_fea * mol_mask.unsqueeze(-1), 1)
            / (torch.sum(mol_mask, 1, keepdim=True) + 1e-6)
        ) * (
            torch.sum(rna_fea * rna_mask.unsqueeze(-1), 1)
            / (torch.sum(rna_mask, 1, keepdim=True) + 1e-6)
        )

        m_att = None
        r_att = None

        for i in range(self.DMA_depth):
            m_r = torch.matmul(
                pair_2d,
                torch.tanh(self.m_to_r_transform[i](mol_fea))
            )

            r_m = torch.matmul(
                pair_2d.transpose(1, 2),
                torch.tanh(self.r_to_m_transform[i](rna_fea))
            )

            m_att = self.mask_softmax(
                self.hc1[i](
                    torch.tanh(self.hc0[i](mol_fea))
                    * torch.tanh(self.mc1[i](m)).unsqueeze(1)
                    * r_m
                ).squeeze(-1),
                mol_mask
            )

            r_att = self.mask_softmax(
                self.hp1[i](
                    torch.tanh(self.hp0[i](rna_fea))
                    * torch.tanh(self.mp1[i](m)).unsqueeze(1)
                    * m_r
                ).squeeze(-1),
                rna_mask
            )

            m = self.GRU_dma(
                m,
                torch.sum(mol_fea * m_att.unsqueeze(-1), 1)
                * torch.sum(rna_fea * r_att.unsqueeze(-1), 1)
            )

        mf = torch.sum(mol_fea * m_att.unsqueeze(-1), 1)
        rf = torch.sum(rna_fea * r_att.unsqueeze(-1), 1)

        return mf, rf

    # =============================================================
    # 3D DMA-GRU
    # =============================================================
    def dma_gru_3d(
        self,
        batch_size,
        rna_fea,
        mol_fea,
        rna_mask,
        mol_mask,
        pair_3d
    ):
        m = (
            torch.sum(mol_fea * mol_mask.unsqueeze(-1), 1)
            / (torch.sum(mol_mask, 1, keepdim=True) + 1e-6)
        ) * (
            torch.sum(rna_fea * rna_mask.unsqueeze(-1), 1)
            / (torch.sum(rna_mask, 1, keepdim=True) + 1e-6)
        )

        m_att = None
        r_att = None

        for i in range(self.DMA_depth):
            m_r = torch.matmul(
                pair_3d,
                torch.tanh(self.m_to_r_transform_3d[i](mol_fea))
            )

            r_m = torch.matmul(
                pair_3d.transpose(1, 2),
                torch.tanh(self.r_to_m_transform_3d[i](rna_fea))
            )

            m_att = self.mask_softmax(
                self.hc1_3d[i](
                    torch.tanh(self.hc0_3d[i](mol_fea))
                    * torch.tanh(self.mc1_3d[i](m)).unsqueeze(1)
                    * r_m
                ).squeeze(-1),
                mol_mask
            )

            r_att = self.mask_softmax(
                self.hp1_3d[i](
                    torch.tanh(self.hp0_3d[i](rna_fea))
                    * torch.tanh(self.mp1_3d[i](m)).unsqueeze(1)
                    * m_r
                ).squeeze(-1),
                rna_mask
            )

            m = self.GRU_dma_3d(
                m,
                torch.sum(mol_fea * m_att.unsqueeze(-1), 1)
                * torch.sum(rna_fea * r_att.unsqueeze(-1), 1)
            )

        mf = torch.sum(mol_fea * m_att.unsqueeze(-1), 1)
        rf = torch.sum(rna_fea * r_att.unsqueeze(-1), 1)

        return mf, rf

    # =============================================================
    # Affinity Prediction Module
    # =============================================================
    def Affinity_pred_module(
        self,
        batch_size,
        RNA_1d,
        RNA_2d,
        Mol_2d,
        rna_mask_2d,
        mol_mask_2d,
        RNA_3d,
        Mol_3d,
        rna_mask_3d,
        mol_mask_3d,
        pair_2d,
        pair_3d
    ):
        kroneck_2d_feat = None
        kroneck_3d_feat = None

        shared_mean = None
        spec_2d = None
        spec_3d = None

        # =========================================================
        # 1. 2D 分支
        # =========================================================
        if self.mol_graph:
            Mol_2d_final = F.leaky_relu(
                self.mol_2d_final(Mol_2d),
                0.1
            )

            if self.rna_repre and self.rna_graph:
                RNA_2d_input = (
                    0.5 * F.leaky_relu(self.rna_2d_final(RNA_2d), 0.1)
                    + 0.5 * F.leaky_relu(self.rna_1d_final(RNA_1d), 0.1)
                )
            elif self.rna_graph:
                RNA_2d_input = F.leaky_relu(
                    self.rna_2d_final(RNA_2d),
                    0.1
                )
            elif self.rna_repre:
                RNA_2d_input = F.leaky_relu(
                    self.rna_1d_final(RNA_1d),
                    0.1
                )
            else:
                raise ValueError(
                    "mol_graph=True 时，至少需要 rna_graph=True 或 rna_repre=True。"
                )

            mf2, rf2 = self.dma_gru_2d(
                batch_size,
                RNA_2d_input,
                Mol_2d_final,
                rna_mask_2d,
                mol_mask_2d,
                pair_2d
            )

            kroneck_2d_feat = torch.matmul(
                mf2.unsqueeze(-1),
                rf2.unsqueeze(-2)
            ).view(batch_size, -1)

            kroneck_2d_feat = torch.nan_to_num(
                F.leaky_relu(kroneck_2d_feat, 0.1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        # =========================================================
        # 2. 3D 分支
        # =========================================================
        if self.coors:
            RNA_3d_final = F.leaky_relu(
                self.rna_3d_final(RNA_3d),
                0.1
            )

            Mol_3d_final = F.leaky_relu(
                self.mol_3d_final(Mol_3d),
                0.1
            )

            mf3, rf3 = self.dma_gru_3d(
                batch_size,
                RNA_3d_final,
                Mol_3d_final,
                rna_mask_3d,
                mol_mask_3d,
                pair_3d
            )

            kroneck_3d_feat = torch.matmul(
                mf3.unsqueeze(-1),
                rf3.unsqueeze(-2)
            ).view(batch_size, -1)

            kroneck_3d_feat = torch.nan_to_num(
                F.leaky_relu(kroneck_3d_feat, 0.1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        # =========================================================
        # 3. 多模态融合 / 单模态回退
        # =========================================================
        if self.mol_graph and self.coors:
            if self.use_mbt:
                bottleneck_feat, shared_mean, spec_2d, spec_3d = self.fusion_module(
                    kroneck_2d_feat,
                    kroneck_3d_feat
                )
            else:
                raw_concat = torch.cat(
                    [kroneck_2d_feat, kroneck_3d_feat],
                    dim=1
                )

                bottleneck_feat = F.gelu(
                    self.concat_proj(raw_concat)
                )

                shared_mean = torch.zeros(
                    batch_size,
                    self.proj_dim,
                    device=kroneck_2d_feat.device,
                    dtype=kroneck_2d_feat.dtype,
                    requires_grad=True
                )

                spec_2d = torch.zeros(
                    batch_size,
                    self.proj_dim,
                    device=kroneck_2d_feat.device,
                    dtype=kroneck_2d_feat.dtype,
                    requires_grad=True
                )

                spec_3d = torch.zeros(
                    batch_size,
                    self.proj_dim,
                    device=kroneck_2d_feat.device,
                    dtype=kroneck_2d_feat.dtype,
                    requires_grad=True
                )

            res_feat = self.res_proj(
                (kroneck_2d_feat + kroneck_3d_feat) / 2.0
            )

            kroneck = bottleneck_feat + self.mbt_alpha * torch.tanh(res_feat)

            # 2D / 3D 几何路由信号
            p2d_s = torch.nan_to_num(
                pair_2d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            p3d_s = torch.nan_to_num(
                pair_3d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            sparsity = (
                (p2d_s > 0.5).float().mean(dim=(1, 2))
                + (p3d_s > 0.5).float().mean(dim=(1, 2))
            ).unsqueeze(1) / 2.0

            confidence = (
                p2d_s.max(dim=2)[0].max(dim=1)[0]
                + p3d_s.max(dim=2)[0].max(dim=1)[0]
            ).unsqueeze(1) / 2.0

            p3d_aligned = F.adaptive_max_pool2d(
                p3d_s.unsqueeze(1),
                (p2d_s.shape[1], p2d_s.shape[2])
            ).squeeze(1)

            const_gap = torch.mean(
                torch.abs(p2d_s - p3d_aligned),
                dim=(1, 2)
            ).unsqueeze(1)

        elif self.mol_graph:
            kroneck = kroneck_2d_feat

            contact = torch.nan_to_num(
                pair_2d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            sparsity = (contact > 0.5).float().mean(
                dim=(1, 2)
            ).unsqueeze(1)

            confidence = contact.max(dim=2)[0].max(
                dim=1
            )[0].unsqueeze(1)

            const_gap = torch.zeros_like(sparsity)

        elif self.coors:
            kroneck = kroneck_3d_feat

            contact = torch.nan_to_num(
                pair_3d,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            sparsity = (contact > 0.5).float().mean(
                dim=(1, 2)
            ).unsqueeze(1)

            confidence = contact.max(dim=2)[0].max(
                dim=1
            )[0].unsqueeze(1)

            const_gap = torch.zeros_like(sparsity)

        else:
            raise ValueError(
                "模型至少需要 mol_graph=True 或 coors=True。"
            )

        routing_features = torch.cat(
            [sparsity, confidence, const_gap],
            dim=1
        )

        # =========================================================
        # 4. MoE 或普通输出层
        # =========================================================
        if self.uncertainty:
            gate_input = torch.cat(
                [kroneck, routing_features],
                dim=1
            )

            out_logits, out_alea, out_epis, out_gate = self.W_out(
                kroneck,
                gate_input
            )

            return (
                out_logits,
                out_alea,
                out_epis,
                out_gate,
                shared_mean,
                spec_2d,
                spec_3d
            )

        return torch.sigmoid(self.W_out(kroneck))

    # =============================================================
    # forward
    # 注意：传参顺序是 Mol_Graph 然后 RNA_Graph
    # =============================================================
    def forward(
        self,
        RNA_repre,
        Seq_mask,
        Mol_Graph,
        RNA_Graph,
        RNA_feats,
        RNA_C4_coors,
        RNA_coors,
        RNA_mask,
        Mol_feats,
        Mol_coors,
        Mol_mask,
        Mol_LAS_dis
    ):
        batch_size = RNA_repre.size(0)

        # =========================================================
        # 1. RNA 2D 图特征
        # =========================================================
        if self.rna_graph:
            RNA_2d_fea, rna_mask_2d = to_dense_batch(
                self.GCN_rna(RNA_Graph),
                RNA_Graph.batch
            )
        else:
            RNA_2d_fea = None
            rna_mask_2d = Seq_mask

        # =========================================================
        # 2. Ligand 2D 图特征
        # =========================================================
        if self.mol_graph:
            Mol_2d_fea, mol_mask_2d = to_dense_batch(
                self.GCN_mol(Mol_Graph),
                Mol_Graph.batch
            )
        else:
            Mol_2d_fea = None
            mol_mask_2d = None

        # =========================================================
        # 3. RNA 1D 特征
        # =========================================================
        if self.rna_repre:
            RNA_1d_fea = self.mlp_rna(RNA_repre)
        else:
            RNA_1d_fea = None

        # =========================================================
        # 4. 3D Equiformer 特征
        # =========================================================
        if self.coors:
            Mol_3d_fea = self.equi_mol(
                Mol_feats,
                Mol_coors,
                Mol_mask
            ).type0

            RNA_3d_fea = self.equi_rna(
                RNA_feats,
                RNA_coors,
                RNA_mask
            ).type0
        else:
            Mol_3d_fea = None
            RNA_3d_fea = None

        # =========================================================
        # 5. 2D / 3D pairwise contact maps
        # =========================================================
        if self.mol_graph:
            if RNA_2d_fea is None or Mol_2d_fea is None:
                raise ValueError(
                    "mol_graph=True 时，RNA_2d_fea 和 Mol_2d_fea 不能为空。"
                )

            pair_2d = torch.sigmoid(
                torch.matmul(
                    RNA_2d_fea,
                    Mol_2d_fea.transpose(1, 2)
                )
            )
        else:
            pair_2d = None

        if self.coors:
            if RNA_3d_fea is None or Mol_3d_fea is None:
                raise ValueError(
                    "coors=True 时，RNA_3d_fea 和 Mol_3d_fea 不能为空。"
                )

            pair_3d = torch.sigmoid(
                torch.matmul(
                    RNA_3d_fea,
                    Mol_3d_fea.transpose(1, 2)
                )
            )
        else:
            pair_3d = None

        # =========================================================
        # 6. Affinity prediction
        # =========================================================
        res = self.Affinity_pred_module(
            batch_size=batch_size,
            RNA_1d=RNA_1d_fea,
            RNA_2d=RNA_2d_fea,
            Mol_2d=Mol_2d_fea,
            rna_mask_2d=rna_mask_2d.float() if rna_mask_2d is not None else None,
            mol_mask_2d=mol_mask_2d.float() if mol_mask_2d is not None else None,
            RNA_3d=RNA_3d_fea,
            Mol_3d=Mol_3d_fea,
            rna_mask_3d=RNA_mask.float() if RNA_mask is not None else None,
            mol_mask_3d=Mol_mask.float() if Mol_mask is not None else None,
            pair_2d=pair_2d,
            pair_3d=pair_3d
        )

        # =========================================================
        # 7. 返回值
        # =========================================================
        if self.uncertainty:
            (
                logits,
                alea,
                epis,
                gate_probs,
                shared_mean,
                spec_2d,
                spec_3d
            ) = res

            return (
                logits,
                alea,
                epis,
                gate_probs,
                pair_2d,
                pair_3d,
                shared_mean,
                spec_2d,
                spec_3d
            )

        return res, pair_2d
