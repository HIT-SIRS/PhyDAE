import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from einops import rearrange
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numbers
import math

"""
Residual Manifold Projector (RMP)
"""
class ResidualManifoldProjector(nn.Module):

    def __init__(self, base_dim: int, target_dims: List[int], num_scales: int = 3):
        super().__init__()
        self.base_dim = base_dim
        self.target_dims = target_dims  # 各解码层的目标维度
        self.num_scales = num_scales

        # 残差编码器：对传输残差进行初步编码
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(3, base_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_dim // 4),
            nn.GELU(),
            nn.Conv2d(base_dim // 4, base_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_dim // 2),
            nn.GELU(),
            nn.Conv2d(base_dim // 2, base_dim, kernel_size=3, padding=1, bias=False)
        )

        # 为每个目标维度创建投影层
        self.dim_projectors = nn.ModuleList([
            nn.Conv2d(base_dim, target_dim, kernel_size=1, bias=False)
            for target_dim in target_dims
        ])

        # MDTA模块：Multi-Dconv head transposed attention
        self.mdta_modules = nn.ModuleList([
            self._create_mdta_block(base_dim) for _ in range(num_scales)
        ])

        # GDFN模块：Gated-Dconv feedforward network
        self.gdfn_modules = nn.ModuleList([
            self._create_gdfn_block(base_dim) for _ in range(num_scales)
        ])

        # 多尺度卷积层
        self.conv1x1 = nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False)
        self.conv3x3_layers = nn.ModuleList([
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False)
            for _ in range(num_scales - 1)
        ])

        # 对比学习头：用于任务识别
        self.contrastive_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(base_dim, base_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(base_dim // 2, base_dim // 4, bias=False)
        )

    def _create_mdta_block(self, dim):
        """创建MDTA块"""
        return nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False),
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim),
            nn.GELU()
        )

    def _create_gdfn_block(self, dim):
        """创建GDFN块"""
        return nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim)
        )

    def forward(self, transport_residual):

        # 编码传输残差
        R0 = self.residual_encoder(transport_residual)

        # 生成基础多尺度残差嵌入（对应论文公式15）
        base_embeddings = []

        # R₁ = GDFN(MDTA(Conv1×1(R₀)))
        R1_input = self.conv1x1(R0)
        R1_mdta = self.mdta_modules[0](R1_input)
        R1 = self.gdfn_modules[0](R1_mdta)
        base_embeddings.append(R1)

        # R₂ = Conv3×3(GDFN(MDTA(Conv1×1(R₀))))
        R2_mdta = self.mdta_modules[1](R1_input)
        R2_gdfn = self.gdfn_modules[1](R2_mdta)
        R2 = self.conv3x3_layers[0](R2_gdfn)
        base_embeddings.append(R2)

        # R₃ = Conv3×3(GDFN(MDTA(Conv1×1(R₂))))
        R3_input = self.conv1x1(R2)
        R3_mdta = self.mdta_modules[2](R3_input)
        R3_gdfn = self.gdfn_modules[2](R3_mdta)
        R3 = self.conv3x3_layers[1](R3_gdfn)
        base_embeddings.append(R3)

        # 为每个目标维度投影残差嵌入
        residual_embeddings_dict = {}
        for i, (projector, target_dim) in enumerate(zip(self.dim_projectors, self.target_dims)):
            # 使用不同的基础嵌入进行投影
            base_idx = i % len(base_embeddings)
            projected = projector(base_embeddings[base_idx])
            residual_embeddings_dict[target_dim] = projected

        # 生成对比学习特征（用R₁进行任务识别）
        contrastive_feat = self.contrastive_head(R1)

        return residual_embeddings_dict, contrastive_feat

# Frequency-Aware Degradation Decomposer (FADD)
class FrequencyAwareDegradationDecomposer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 频域模式提取器
        self.freq_extractors = nn.ModuleDict({
            'low_freq': nn.Conv2d(dim, dim // 4, kernel_size=7, padding=3, groups=dim // 4, bias=False),
            'mid_freq': nn.Conv2d(dim, dim // 4, kernel_size=5, padding=2, groups=dim // 4, bias=False),
            'high_freq': nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, groups=dim // 4, bias=False),
            'edge_freq': nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        })

        # 退化类型分类器
        self.degradation_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Flatten(1),
            nn.Linear(dim // 2 * 16, 4, bias=False),  # 4种退化类型
            nn.Softmax(dim=-1)
        )

        # 频域嵌入生成器
        self.freq_embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False)
        )

    def forward(self, residual_features):

        # 提取不同频率成分
        freq_components = []
        for name, extractor in self.freq_extractors.items():
            freq_comp = extractor(residual_features)
            freq_components.append(freq_comp)

        freq_features = torch.cat(freq_components, dim=1)  # 合并频域特征

        # 退化类型识别
        degradation_probs = self.degradation_classifier(residual_features)

        # 生成频域嵌入
        freq_embedding = self.freq_embedder(residual_features)

        return freq_features, degradation_probs, freq_embedding


# Degradation-Aware Restoration Expert (DARE)
class DegradationAwareRestorationExpert(nn.Module):

    def __init__(self, dim: int, expert_type: str, shared_components):
        super().__init__()
        self.dim = dim
        self.expert_type = expert_type
        self.shared = shared_components
        self.eps = 1e-3

        # 残差条件模块
        self.residual_conditioner = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(self._get_valid_num_groups(dim, 16), dim),  # 修复这里
            nn.GELU()
        )

        # 专家特定模块
        if expert_type == 'dehaze':
            self.expert_module = self._build_dehaze_expert()
        elif expert_type == 'denoise':
            self.expert_module = self._build_denoise_expert()
        elif expert_type == 'dedark':
            self.expert_module = self._build_dedark_expert()
        elif expert_type == 'deblur':
            self.expert_module = self._build_deblur_expert()

    def _build_dehaze_expert(self):

        # 波长相关的缩放因子 - 作为类的直接属性
        self.wavelength_scale = nn.Parameter(torch.tensor([1.0, 0.95, 0.9]))  # R, G, B的相对波长影响

        return nn.ModuleDict({
            # 多波段透射率估计网络
            'transmission_estimator': nn.ModuleDict({
                # 波段特定的透射率估计器
                'band_specific': nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim // 4, 3, padding=1, groups=self.dim // 4, bias=False),
                        nn.Conv2d(self.dim // 4, 1, 1, bias=False),
                        nn.Sigmoid()
                    ) for _ in range(3)  # RGB三个波段
                ]),

                # 空间非均匀性建模
                'spatial_refine': nn.Sequential(
                    nn.Conv2d(3, self.dim // 4, 1, bias=False),  # 融合三波段
                    nn.Conv2d(self.dim // 4, self.dim // 4, 3, padding=1, groups=self.dim // 4, bias=False),
                    nn.GELU(),
                    nn.Conv2d(self.dim // 4, self.dim // 4, 3, padding=2, dilation=2, groups=self.dim // 4, bias=False),
                    nn.Conv2d(self.dim // 4, 3, 1, bias=False),  # 输出精细化的三波段透射率
                    nn.Sigmoid()
                ),
                # 注意：wavelength_scale 已移到类属性中
            }),

            # 大气光估计网络（考虑局部和全局）
            'atmospheric_light': nn.ModuleDict({
                # 全局大气光
                'global': nn.Sequential(
                    nn.AdaptiveAvgPool2d(8),
                    nn.Conv2d(self.dim, self.dim // 4, 1, bias=False),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.dim // 4, 3, 1, bias=False),  # RGB三通道
                    nn.Sigmoid()
                ),

                # 局部大气光变化（处理非均匀雾霾）
                'local': nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // 4, 5, padding=2, stride=2, bias=False),
                    nn.GELU(),
                    nn.Conv2d(self.dim // 4, self.dim // 8, 3, padding=1, bias=False),
                    nn.ConvTranspose2d(self.dim // 8, 3, 4, stride=2, padding=1, bias=False),
                    nn.Sigmoid()
                ),

                # 混合权重
                'mix_weight': nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.dim, 1, 1, bias=False),
                    nn.Sigmoid()
                )
            }),

            # 深度/距离引导网络（雾霾浓度与深度相关）
            'depth_guide': nn.Sequential(
                nn.Conv2d(self.dim, self.dim // 2, 3, padding=1, bias=False),
                nn.GroupNorm(self._get_valid_num_groups(self.dim // 2, 8), self.dim // 2),
                nn.GELU(),
                nn.Conv2d(self.dim // 2, 1, 3, padding=1, bias=False),
                nn.Sigmoid()  # 归一化深度图
            ),

            # 细节恢复网络
            'detail_recovery': nn.Sequential(
                nn.Conv2d(self.dim + 3, self.dim, 3, padding=1, groups=1, bias=False),  # +3 for dehazed RGB
                nn.GroupNorm(self._get_valid_num_groups(self.dim, 16), self.dim),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim, 1, bias=False)
            )
        })

    def _get_valid_num_groups(self, num_channels, preferred_groups):
        # 从preferred_groups开始向下搜索，找到第一个能整除num_channels的数
        for g in range(min(preferred_groups, num_channels), 0, -1):
            if num_channels % g == 0:
                return g
        return 1

    def _build_denoise_expert(self):
        # 空间自适应噪声估计
        noise_estimator = nn.ModuleDict({
            # 多尺度噪声分析
            'multi_scale': nn.ModuleList([
                nn.Conv2d(self.dim, self.dim // 4, k, padding=k // 2, groups=self.dim // 4, bias=False)
                for k in [3, 5, 7]
            ]),

            # 噪声图生成
            'noise_map': nn.Sequential(
                nn.Conv2d(self.dim // 4 * 3, self.dim // 2, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(self.dim // 2, 1, 3, padding=1, bias=False),
                nn.Sigmoid()  # 输出噪声强度图 [0,1]
            )
        })

        # 自适应滤波器组 - 修正版本
        adaptive_filters = nn.ModuleDict({
            # 轻度噪声滤波器 (σ=5-15)
            'light': nn.Sequential(
                # 先处理特征，然后融合噪声信息
                nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=False),
                nn.Conv2d(self.dim, self.dim, 1, bias=False),  # 融合通道
                nn.GELU()
            ),

            # 中度噪声滤波器 (σ=15-25)
            'medium': nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 5, padding=2, groups=self.dim, bias=False),
                nn.Conv2d(self.dim, self.dim, 1, bias=False),
                nn.GELU()
            ),

            # 重度噪声滤波器 (σ=25-35)
            'heavy': nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 7, padding=3, groups=self.dim, bias=False),
                nn.Conv2d(self.dim, self.dim, 1, bias=False),
                nn.GELU()
            )
        })

        # 噪声条件融合模块
        noise_condition = nn.Sequential(
            nn.Conv2d(1, self.dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim // 4, self.dim, 1, bias=False),
            nn.Sigmoid()
        )

        # 细节保护分支
        detail_preserve = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=False),
            nn.Conv2d(self.dim, self.dim // 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim // 2, self.dim, 1, bias=False)
        )

        # 输出融合
        output_fusion = nn.Sequential(
            nn.Conv2d(self.dim * 4, self.dim * 2, 1, bias=False),
            nn.GroupNorm(self._get_valid_num_groups(self.dim * 2, 16), self.dim * 2),
            nn.GELU(),
            nn.Conv2d(self.dim * 2, self.dim, 1, bias=False)
        )

        return nn.ModuleDict({
            'noise_estimator': noise_estimator,
            'adaptive_filters': adaptive_filters,
            'noise_condition': noise_condition,
            'detail_preserve': detail_preserve,
            'output_fusion': output_fusion
        })

    def _build_deblur_expert(self):

        # 增强的参数估计：估计kernel_size, σx, σy, angle
        param_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(self.dim, self.dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim // 4, self.dim // 8, 3, padding=1, groups=self.dim // 8, bias=False),
            nn.Flatten(),
            nn.Linear((self.dim // 8) * 16 * 16, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 4, bias=False)  # [kernel_size_factor, σx, σy, angle]
        )

        # 自适应核生成网络（轻量级）
        kernel_generator = nn.ModuleDict({
            # 参数嵌入生成器
            'param_embedder': nn.Sequential(
                nn.Linear(4, self.dim // 4, bias=False),
                nn.GELU(),
                nn.Linear(self.dim // 4, self.dim, bias=False),
                nn.Sigmoid()
            ),

            # 各向异性卷积分支（修正版本）
            'anisotropic_branch': nn.ModuleList([
                # X方向处理
                nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, (1, 7), padding=(0, 3), groups=self.dim, bias=False),
                    nn.Conv2d(self.dim, self.dim // 2, 1, bias=False),
                    nn.GELU()
                ),
                # Y方向处理
                nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, (7, 1), padding=(3, 0), groups=self.dim, bias=False),
                    nn.Conv2d(self.dim, self.dim // 2, 1, bias=False),
                    nn.GELU()
                ),
                # 对角线处理（通过组合实现旋转效果）
                nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, 5, padding=2, groups=self.dim, bias=False),
                    nn.Conv2d(self.dim, self.dim // 2, 1, bias=False),
                    nn.GELU()
                )
            ]),

            # 自适应混合
            'mixer': nn.Sequential(
                nn.Conv2d(self.dim // 2 * 3, self.dim, 1, bias=False),
                nn.GroupNorm(self._get_valid_num_groups(self.dim, 16), self.dim),
                nn.GELU()
            )
        })

        # 频域增强分支（针对模糊特性优化）
        freq_enhance = nn.Sequential(
            # 使用不同大小的卷积核捕获不同频率
            nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=False),
            nn.Conv2d(self.dim, self.dim, 5, padding=2, groups=self.dim, bias=False),
            nn.Conv2d(self.dim, self.dim, 1, bias=False),
            nn.GELU()
        )

        # 细节恢复
        detail_recovery = nn.Conv2d(self.dim * 2, self.dim, 1, bias=False)

        return nn.ModuleDict({
            'param_estimator': param_estimator,
            'kernel_generator': kernel_generator,
            'freq_enhance': freq_enhance,
            'detail_recovery': detail_recovery
        })

    def _build_dedark_expert(self):

        return nn.ModuleDict({
            # 自适应伽马估计网络
            'gamma_estimator': nn.ModuleDict({
                # 局部伽马图估计
                'local_gamma': nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // 2, 3, padding=1, stride=2, bias=False),
                    nn.GELU(),
                    nn.Conv2d(self.dim // 2, self.dim // 4, 3, padding=1, bias=False),
                    nn.ConvTranspose2d(self.dim // 4, 1, 4, stride=2, padding=1, bias=False),
                    nn.Sigmoid()  # 输出[0,1]，映射到gamma [1,3]
                ),

                # 全局伽马值
                'global_gamma': nn.Sequential(
                    nn.AdaptiveAvgPool2d(4),
                    nn.Conv2d(self.dim, self.dim // 4, 1, bias=False),
                    nn.GELU(),
                    nn.Flatten(),
                    nn.Linear((self.dim // 4) * 16, 1, bias=False),
                    nn.Sigmoid()
                ),

                # 平滑正则化
                'smooth_filter': nn.Conv2d(1, 1, 5, padding=2, bias=False)
            }),

            # 基于Retinex的照明分解
            'illumination_decompose': nn.ModuleDict({
                # 多尺度照明估计
                'multi_scale': nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim // 4, k, padding=k // 2, groups=self.dim // 4, bias=False),
                        nn.Conv2d(self.dim // 4, 1, 1, bias=False),
                        nn.Sigmoid()
                    ) for k in [3, 5, 7, 9]  # 多尺度
                ]),

                # 照明融合
                'fusion': nn.Sequential(
                    nn.Conv2d(4, self.dim // 4, 1, bias=False),  # 4个尺度
                    nn.GELU(),
                    nn.Conv2d(self.dim // 4, 1, 3, padding=1, bias=False),
                    nn.Sigmoid()
                )
            }),

            # 颜色校正网络（修复）
            'color_correction': nn.ModuleDict({
                # 颜色分布统计 - 修复输出维度
                'color_stats': nn.Sequential(
                    nn.AdaptiveAvgPool2d(8),
                    nn.Conv2d(3, self.dim // 4, 1, bias=False),
                    nn.GELU(),
                    nn.Flatten(),
                    nn.Linear((self.dim // 4) * 64, 9, bias=False)  # 修复：64 = 8*8, 输出9维用于3x3矩阵
                ),

                # 局部颜色调整
                'local_adjust': nn.Sequential(
                    nn.Conv2d(self.dim + 3, self.dim // 2, 1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(self.dim // 2, 3, 3, padding=1, groups=1, bias=False),
                    nn.Tanh()  # 输出调整量
                )
            }),

            # 细节增强网络
            'detail_enhance': nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=False),
                nn.Conv2d(self.dim, self.dim // 2, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(self.dim // 2, self.dim, 1, bias=False)
            )
        })

    def forward(self, x, residual_embedding=None):

        # 残差条件化处理
        if residual_embedding is not None:
            if residual_embedding.shape[1] != x.shape[1]:
                print(f"Warning: residual_embedding dim {residual_embedding.shape[1]} != x dim {x.shape[1]}")
                conditioned_x = x
            else:
                if residual_embedding.shape[-2:] != x.shape[-2:]:
                    residual_embedding = F.interpolate(
                        residual_embedding, size=x.shape[-2:], mode='bilinear', align_corners=False
                    )
                conditioned_x = self.residual_conditioner(torch.cat([x, residual_embedding], dim=1))
        else:
            conditioned_x = x

        # 提取多尺度特征
        ms_features = []
        for extractor in self.shared.multi_scale:
            ms_features.append(extractor(conditioned_x))
        multi_scale_feat = torch.cat(ms_features, dim=1)

        # 专家特定处理
        if self.expert_type == 'dehaze':
            return self._dehaze_forward(conditioned_x, multi_scale_feat)
        elif self.expert_type == 'denoise':
            return self._denoise_forward(conditioned_x, multi_scale_feat)
        elif self.expert_type == 'dedark':
            return self._dedark_forward(conditioned_x, multi_scale_feat)
        elif self.expert_type == 'deblur':
            return self._deblur_forward(conditioned_x, multi_scale_feat)

    def _dehaze_forward(self, x, ms_feat):
        """修复版去雾 - 增强数值稳定性"""
        b, c, h, w = x.shape
        rgb_input = x[:, :3] if c >= 3 else x.repeat(1, 3, 1, 1)[:, :3]

        # 1. 透射率估计（添加裁剪）
        band_transmissions = []
        for i, estimator in enumerate(self.expert_module['transmission_estimator']['band_specific']):
            t_band = estimator(x)
            scale = self.wavelength_scale[i]
            band_transmissions.append(t_band * scale)

        t_coarse = torch.cat(band_transmissions, dim=1)
        t_refined = self.expert_module['transmission_estimator']['spatial_refine'](t_coarse)

        depth_map = self.expert_module['depth_guide'](x)
        beta = 0.5
        t_depth_guided = t_refined * torch.exp(-beta * depth_map)

        # === 增大透射率下限，避免除零 ===
        t_final = torch.clamp(t_depth_guided, min=0.2, max=1.0)  # 原min=0.1

        # 2. 大气光估计（添加裁剪）
        global_A = self.expert_module['atmospheric_light']['global'](x)
        local_A = self.expert_module['atmospheric_light']['local'](x)
        mix_w = self.expert_module['atmospheric_light']['mix_weight'](x)
        A = mix_w * global_A + (1 - mix_w) * local_A
        A = torch.clamp(A, 0.1, 0.9)  # === 限制大气光范围 ===

        # 3. 应用大气散射模型（增强稳定性）
        numerator = rgb_input - A * (1 - t_final)
        denominator = t_final + self.eps

        # === 添加分子裁剪，避免极值 ===
        numerator = torch.clamp(numerator, -2.0, 2.0)
        J = numerator / denominator
        J = torch.clamp(J, 0, 1)

        # === 检测NaN ===
        if torch.isnan(J).any():
            print("Warning: NaN in dehaze forward, returning input")
            J = rgb_input

        # 4. 细节恢复
        combined = torch.cat([x, J], dim=1)
        detail_enhanced = self.expert_module['detail_recovery'](combined)
        output = self.shared.shared_refine(detail_enhanced)

        return output + x * 0.2

    def _denoise_forward(self, x, ms_feat=None):
        # 多尺度噪声分析
        noise_features = []
        for analyzer in self.expert_module['noise_estimator']['multi_scale']:
            noise_features.append(analyzer(x))

        # 生成空间噪声图
        noise_concat = torch.cat(noise_features, dim=1)
        noise_map = self.expert_module['noise_estimator']['noise_map'](noise_concat)  # [B, 1, H, W]

        # 生成噪声条件特征
        noise_cond = self.expert_module['noise_condition'](noise_map)  # [B, dim, H, W]

        # 应用自适应滤波器，使用噪声条件进行调制
        # 对每个滤波器，先处理特征，然后用噪声条件调制
        light_filtered = self.expert_module['adaptive_filters']['light'](x)
        light_filtered = light_filtered * noise_cond  # 噪声调制

        medium_filtered = self.expert_module['adaptive_filters']['medium'](x)
        medium_filtered = medium_filtered * noise_cond

        heavy_filtered = self.expert_module['adaptive_filters']['heavy'](x)
        heavy_filtered = heavy_filtered * noise_cond

        # 根据噪声强度自适应混合
        # noise_map: [0,1], 映射到 σ=[5,35]
        noise_level = noise_map * 30 + 5  # 近似噪声标准差

        # 计算各滤波器权重
        w_light = torch.exp(-((noise_level - 10) ** 2) / (2 * 5 ** 2))
        w_medium = torch.exp(-((noise_level - 20) ** 2) / (2 * 5 ** 2))
        w_heavy = torch.exp(-((noise_level - 30) ** 2) / (2 * 5 ** 2))

        # 归一化权重
        w_sum = w_light + w_medium + w_heavy + 1e-6
        w_light, w_medium, w_heavy = w_light / w_sum, w_medium / w_sum, w_heavy / w_sum

        # 加权混合
        filtered = (w_light * light_filtered +
                    w_medium * medium_filtered +
                    w_heavy * heavy_filtered)

        # 细节保护
        details = self.expert_module['detail_preserve'](x)

        # 最终融合
        output = self.expert_module['output_fusion'](torch.cat([
            filtered, details, light_filtered, heavy_filtered
        ], dim=1))

        # 共享细化
        refined = self.shared.shared_refine(output)

        # 自适应残差连接（噪声越大，残差权重越小）
        residual_weight = 1.0 - noise_map.mean(dim=(2, 3), keepdim=True) * 0.8
        return refined + x * residual_weight

    def _dedark_forward(self, x, ms_feat):
        b, c, h, w = x.shape
        rgb_input = x[:, :3] if c >= 3 else x.repeat(1, 3, 1, 1)[:, :3]

        # 1. 伽马估计（添加范围限制）
        local_gamma = self.expert_module['gamma_estimator']['local_gamma'](x)
        global_gamma = self.expert_module['gamma_estimator']['global_gamma'](x)
        local_gamma = self.expert_module['gamma_estimator']['smooth_filter'](local_gamma)

        # === 限制gamma范围，避免极值 ===
        gamma_map = 1 + 1.5 * (0.7 * local_gamma + 0.3 * global_gamma.view(b, 1, 1, 1))
        gamma_map = torch.clamp(gamma_map, 1.0, 2.5)  # 原[1,3]，现在更保守

        # 2. 照明分解
        illuminations = []
        for scale_estimator in self.expert_module['illumination_decompose']['multi_scale']:
            illum = scale_estimator(x)
            illuminations.append(illum)

        illum_concat = torch.cat(illuminations, dim=1)
        illumination = self.expert_module['illumination_decompose']['fusion'](illum_concat)
        illumination = torch.clamp(illumination, min=0.01)  # 增加下限

        # 3. 伽马校正（增强稳定性）
        adaptive_gamma = gamma_map * (1 + 0.5 * (1 - illumination))
        adaptive_gamma = torch.clamp(adaptive_gamma, 1.0, 3.0)  # === 确保范围安全 ===

        rgb_normalized = torch.clamp(rgb_input, min=1e-6, max=1.0)

        # === 使用安全的pow操作 ===
        gamma_inv = 1.0 / adaptive_gamma
        gamma_inv = torch.clamp(gamma_inv, 0.33, 1.0)  # 限制倒数范围
        enhanced = torch.pow(rgb_normalized, gamma_inv)
        enhanced = torch.clamp(enhanced, 0, 1)

        # === 检测NaN ===
        if torch.isnan(enhanced).any():
            print("Warning: NaN in dedark forward, returning input")
            enhanced = rgb_input

        # 4. 颜色校正（简化以提高稳定性）
        color_matrix_flat = self.expert_module['color_correction']['color_stats'](enhanced)
        color_matrix = color_matrix_flat.view(b, 3, 3)

        # === 使用try-except避免矩阵乘法失败 ===
        try:
            enhanced_flat = enhanced.view(b, 3, -1)
            color_corrected = torch.bmm(color_matrix, enhanced_flat)
            color_corrected = color_corrected.view(b, 3, h, w)
            color_corrected = torch.clamp(color_corrected, 0, 1)
        except Exception as e:
            print(f"Warning: Color correction failed: {e}, skipping")
            color_corrected = enhanced

        # 局部调整
        combined_color = torch.cat([x, color_corrected], dim=1)
        local_adjust = self.expert_module['color_correction']['local_adjust'](combined_color)
        final_rgb = torch.clamp(color_corrected + 0.1 * local_adjust, 0, 1)

        # 5. 特征融合
        if c > 3:
            output = x.clone()
            output[:, :3] = final_rgb
        else:
            output = final_rgb

        detail_enhanced = self.expert_module['detail_enhance'](output)
        output = output + 0.3 * detail_enhanced
        output = torch.clamp(output, 0, 1)  # === 最终裁剪 ===

        # 6. 共享细化
        refined = self.shared.shared_refine(output)
        residual_weight = illumination.mean(dim=(2, 3), keepdim=True) * 0.5

        return refined + x * residual_weight

    def _deblur_forward(self, x, ms_feat=None):
        b, c, h, w = x.shape

        # 估计模糊参数
        params = self.expert_module['param_estimator'](x)
        kernel_factor = torch.sigmoid(params[:, 0:1])  # [0,1] 映射到 [7,21]
        sigma_x = 0.2 + 3.8 * torch.sigmoid(params[:, 1:2])  # [0.2, 4]
        sigma_y = 0.2 + 3.8 * torch.sigmoid(params[:, 2:3])  # [0.2, 4]
        angle = torch.sigmoid(params[:, 3:4])  # [0, 1]

        # 构建参数向量（用于param_embedder）
        param_vector = torch.cat([
            kernel_factor.view(b, 1),
            sigma_x.view(b, 1),
            sigma_y.view(b, 1),
            angle.view(b, 1)
        ], dim=1)  # [B, 4]

        # 参数嵌入
        param_embed_vector = self.expert_module['kernel_generator']['param_embedder'](param_vector)  # [B, dim]

        # 将嵌入向量扩展到空间维度
        param_embed_spatial = param_embed_vector.view(b, self.dim, 1, 1).expand(b, self.dim, h, w)

        # 与输入特征结合
        x_with_params = x * param_embed_spatial  # 使用乘法调制而不是拼接

        # 各向异性去模糊
        branch_outputs = []
        for branch in self.expert_module['kernel_generator']['anisotropic_branch']:
            branch_outputs.append(branch(x_with_params))

        # 根据参数自适应混合
        # 使用σx, σy和angle来调整各分支权重
        anisotropy = (sigma_x - sigma_y).abs().view(b, 1, 1, 1)

        # 混合权重：考虑各向异性程度
        weights = F.softmax(torch.stack([
            1.0 - anisotropy,  # 各向同性权重
            anisotropy * (1 + torch.cos(angle * math.pi).view(b, 1, 1, 1)),  # X方向权重
            anisotropy * (1 + torch.sin(angle * math.pi).view(b, 1, 1, 1))  # Y方向权重
        ], dim=1), dim=1)

        # 加权混合
        deblurred = self.expert_module['kernel_generator']['mixer'](torch.cat(branch_outputs, dim=1))

        # 频域增强
        freq_enhanced = self.expert_module['freq_enhance'](x)

        # 细节恢复
        output = self.expert_module['detail_recovery'](torch.cat([deblurred, freq_enhanced], dim=1))

        # 共享细化
        refined = self.shared.shared_refine(output)

        # 残差连接
        return refined + x * 0.3

# Probabilistic Expert Allocator (PEA)
class ProbabilisticExpertAllocator(nn.Module):

    def __init__(self, dim: int, num_experts: int = 4, freq_dim: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.freq_dim = freq_dim if freq_dim is not None else dim

        # 视觉路由 - 增强判别能力
        self.visual_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_experts, kernel_size=1, bias=False),
            nn.Flatten(1),
            nn.AdaptiveAvgPool1d(num_experts)
        )

        # 频域路由 - 增强判别能力
        self.freq_router = nn.Sequential(
            nn.Linear(self.freq_dim, self.freq_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.freq_dim // 2, num_experts, bias=False)
        )

        # 任务特定的专家偏好矩阵 - 这是关键!
        # 初始化为对角矩阵,鼓励每个专家专注于一个任务
        self.expert_task_affinity = nn.Parameter(
            torch.eye(num_experts) * 2.0 + torch.randn(num_experts, num_experts) * 0.1
        )

        # **修复: 使用可学习的温度参数,初始值设为较小值以增强区分度**
        self.temperature = nn.Parameter(torch.ones(1) * 0.3)

        # 添加温度范围约束
        self.min_temp = 0.1
        self.max_temp = 1.0

    def forward(self, x, freq_emb=None, degradation_probs=None, training=True):

        batch_size = x.size(0)

        # 1. 视觉路由
        visual_logits = self.visual_router(x)

        # 2. 频域路由
        if freq_emb is not None:
            freq_logits = self.freq_router(freq_emb)
            freq_logits = torch.clamp(freq_logits, min=-10, max=10)
        else:
            freq_logits = torch.zeros_like(visual_logits)

        # 3. **关键修复: 基于退化类型的专家亲和度**
        if degradation_probs is not None:
            # degradation_probs: [B, 4] - 退化类型概率
            # expert_task_affinity: [4, 4] - 专家对任务的偏好
            # 结果: [B, 4] - 每个样本对每个专家的偏好
            task_guided_logits = torch.matmul(degradation_probs, self.expert_task_affinity)
            task_guided_logits = torch.clamp(task_guided_logits, min=-5, max=5)
        else:
            task_guided_logits = torch.zeros_like(visual_logits)

        # 4. 融合所有路由信号 - 增加任务引导的权重
        combined_logits = (
                visual_logits +
                0.5 * freq_logits +
                1.5 * task_guided_logits  # 增加任务引导权重
        )
        combined_logits = torch.clamp(combined_logits, min=-20, max=20)

        # 5. **修复: 约束温度范围**
        temp = torch.clamp(self.temperature, self.min_temp, self.max_temp)

        if training:
            # 训练时: 使用Gumbel-Softmax增强探索
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(combined_logits) + 1e-8) + 1e-8)
            noisy_logits = (combined_logits + 0.1 * gumbel_noise) / temp
            weights = F.softmax(noisy_logits, dim=-1)
        else:
            # 推理时: 直接softmax
            weights = F.softmax(combined_logits / temp, dim=-1)

        return weights, combined_logits

    def get_specialization_loss(self, weights, task_ids):

        batch_size = weights.size(0)

        # 为每个任务计算理想的专家分布 (one-hot)
        ideal_distribution = F.one_hot(task_ids, num_classes=self.num_experts).float()

        # 使用KL散度衡量当前分布与理想分布的差异
        # 添加小常数避免log(0)
        weights_safe = weights + 1e-8
        ideal_safe = ideal_distribution + 1e-8

        kl_div = F.kl_div(
            weights_safe.log(),
            ideal_safe,
            reduction='batchmean'
        )

        return kl_div


# Sparse Adaptive Expert Adapter (SAEA)
class SparseAdaptiveExpertAdapter(nn.Module):

    def __init__(self, dim: int, rank: int, num_experts: int = 4,
                 top_k: int = 2, freq_dim: int = None):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过专家数
        self.freq_dim = freq_dim if freq_dim is not None else dim

        # 共享组件
        self.shared_components = SharedPhysicsComponents(dim)

        # 物理专家
        expert_types = ['dehaze', 'denoise', 'dedark', 'deblur']
        self.experts = nn.ModuleList([
            DegradationAwareRestorationExpert(dim, expert_type, self.shared_components)
            for expert_type in expert_types
        ])

        # **修复: 使用改进的路由器**
        self.router = ProbabilisticExpertAllocator(dim, num_experts, freq_dim=self.freq_dim)

        # 频域门控
        self.freq_gate = nn.Sequential(
            nn.Linear(self.freq_dim, num_experts, bias=False),
            nn.Sigmoid()
        )

        # 低秩投影
        self.down_proj = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(rank, dim, kernel_size=1, bias=False)

        # 特征融合
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # 损失记录
        self.moe_loss = 0.0
        self.balance_loss = 0.0
        self.specialization_loss = 0.0

    def forward(self, x, freq_emb=None, degradation_probs=None,
                residual_embeddings_dict=None, task_ids=None):

        b, c, h, w = x.shape

        # 低秩投影
        x_low = self.down_proj(x)

        # **关键: 获取路由权重和logits**
        routing_weights, routing_logits = self.router(
            x, freq_emb, degradation_probs, self.training
        )

        # 频域门控
        if freq_emb is not None:
            freq_weights = self.freq_gate(freq_emb)
            combined_weights = routing_weights * (1 + 0.2 * freq_weights)
            # 重新归一化
            combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            combined_weights = routing_weights

        # **修复: Top-K专家选择 - 使用更激进的策略**
        if self.top_k == 1:
            # 单专家模式: 直接选择概率最高的
            topk_weights, topk_indices = torch.topk(combined_weights, 1, dim=-1)
            topk_weights = torch.ones_like(topk_weights)  # 权重设为1
        else:
            # 多专家模式: 选择Top-K并重新归一化
            topk_weights, topk_indices = torch.topk(combined_weights, self.top_k, dim=-1)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 专家处理
        expert_output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        for i in range(b):
            for k in range(self.top_k):
                expert_idx = topk_indices[i, k].item()
                weight = topk_weights[i, k]

                # 获取残差嵌入
                residual_emb = None
                if residual_embeddings_dict is not None and c in residual_embeddings_dict:
                    residual_emb = residual_embeddings_dict[c][i:i + 1]

                # 专家处理
                output = self.experts[expert_idx](x[i:i + 1], residual_emb)
                expert_output[i:i + 1] += weight.view(1, 1, 1, 1) * output

                # 记录专家使用情况
                expert_usage[expert_idx] += weight.item()

        # 特征融合
        x_up = self.up_proj(x_low)
        output = self.fusion(torch.cat([x_up, expert_output], dim=1))

        # **修复: 改进的负载均衡损失**
        # 目标: 每个专家被使用的次数应该大致相等
        avg_usage = expert_usage.mean()
        balance_loss = ((expert_usage - avg_usage) ** 2).mean() / (avg_usage + 1e-6)
        self.balance_loss = balance_loss * 0.01

        # **新增: 专家专业化损失**
        if task_ids is not None:
            self.specialization_loss = self.router.get_specialization_loss(
                combined_weights, task_ids
            ) * 0.1
        else:
            self.specialization_loss = 0.0

        # 总MoE损失
        self.moe_loss = self.balance_loss + self.specialization_loss

        return output

# Layer Norm and Basic Components
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 复用原始组件
class SharedPhysicsComponents(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False),
                nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
                nn.GroupNorm(4, dim // 4),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]
        ])
        self.shared_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim),
            nn.ReLU(inplace=True)
        )
        self.spatial_process = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=False),
            nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False)
        )


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1,
                                   padding=7 // 2, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# Dual-Stream Decoder Block (DSAB)
class DualStreamAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 rank=None, num_experts=None, top_k=None, freq_dim=None, **kwargs):
        super().__init__()

        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type),
        ])

        self.proj_shared = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        self.shared = Attention(dim, num_heads, bias)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # 使用改进的适配器
        self.da_adapter = SparseAdaptiveExpertAdapter(
            dim=dim,
            rank=rank,
            num_experts=num_experts,
            top_k=top_k,
            freq_dim=freq_dim
        )

    def forward(self, x, freq_emb=None, degradation_probs=None,
                residual_embeddings_dict=None, task_ids=None):
        shortcut = x
        x = self.norms[0](x)

        x_s = self.proj_shared(x)
        x_s = self.shared(x_s)

        # **修复: 传递task_ids**
        x_a = self.da_adapter(
            x, freq_emb, degradation_probs,
            residual_embeddings_dict, task_ids
        )

        x = self.mixer(x_a, x_s) + shortcut
        x = x + self.ffn(self.norms[1](x))

        return x, self.da_adapter.moe_loss


# 其他基础组件
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norms = nn.ModuleList([
            LayerNorm(dim, LayerNorm_type),
            LayerNorm(dim, LayerNorm_type)
        ])
        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x


class EncoderResidualGroup(nn.Module):
    def __init__(self, dim: int, num_heads: List[int], num_blocks: int,
                 ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()
        self.loss = None
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList([
            EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        self.loss = 0
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderResidualGroup(nn.Module):

    def __init__(self, dim: int, num_heads: List[int], num_blocks: int,
                 ffn_expansion: int, LayerNorm_type: str, bias: bool,
                 rank=None, num_experts=None, top_k=None, freq_dim=None, **kwargs):
        super().__init__()
        self.loss = None
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList([
            DualStreamAttentionBlock(
                dim, num_heads, ffn_expansion, bias, LayerNorm_type,
                rank=rank, num_experts=num_experts, top_k=top_k,
                freq_dim=freq_dim, **kwargs
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, freq_emb=None, degradation_probs=None,
                residual_embeddings_dict=None, task_ids=None):
        self.loss = 0
        for layer in self.layers:
            x, loss = layer(x, freq_emb, degradation_probs,
                            residual_embeddings_dict, task_ids)
            self.loss += loss
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class FrequencyEmbedding(nn.Module):
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        with torch.no_grad():
            kernel = torch.tensor([[[[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]]]], dtype=torch.float32) / 9.0
            self.high_conv.weight.data = kernel.repeat(dim, 1, 1, 1).clone()
        self.high_conv.weight.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False)
        )

    def forward(self, x):
        x = F.gelu(self.high_conv(x))
        x = x.mean(dim=(-2, -1))
        x = self.mlp(x)
        return x

# PhyDAE: Physics-Guided Degradation-Adaptive Experts
class PhyDAE(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 levels: int = 4,
                 heads=[1, 2, 4, 8],
                 num_blocks=[4, 6, 6, 8],
                 num_dec_blocks=[2, 4, 4],
                 ffn_expansion_factor=2,
                 num_refinement_blocks=1,
                 LayerNorm_type='WithBias',
                 bias=False,
                 rank=8,
                 num_experts=4,
                 topk=2):
        super(PhyDAE, self).__init__()

        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.dim = dim

        dims = [dim * 2 ** i for i in range(levels)]
        ranks = [rank for i in range(levels - 1)]

        # 计算解码器各层的维度（从高到低）
        decoder_dims = dims[::-1][1:]  # 去掉最高层，因为decoder从第二高层开始

        # 为每个解码层生成对应维度的嵌入
        self.regm = ResidualManifoldProjector(
            base_dim=dim,
            target_dims=decoder_dims
        )

        # 傅里叶残差分析器 - 使用基础维度
        self.freq_analyzer = FrequencyAwareDegradationDecomposer(dim)

        # === 原有架构组件 ===
        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])

        # Encoder
        self.enc = nn.ModuleList([])
        for i in range(levels - 1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i],
                    num_blocks=num_blocks[i],
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=True
                ),
                Downsample(dims[i])
            ]))

        # Latent
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1],
            num_heads=heads[-1],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type,
            bias=True
        )

        # Decoder ===
        dims = dims[::-1]
        ranks = ranks[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]

        self.dec = nn.ModuleList([])
        for i in range(levels - 1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i + 1],
                    num_blocks=num_dec_blocks[i],
                    num_heads=heads[i + 1],
                    ffn_expansion=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type,
                    bias=bias,
                    rank=ranks[i],
                    num_experts=num_experts,
                    top_k=topk,
                    freq_dim=dims[0]  # 使用最高层维度作为freq_dim
                ),
            ]))

        # Refinement
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks,
            num_heads=heads[0],
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type,
            bias=True
        )

        # 输出层
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # 损失跟踪
        self.total_loss = None
        self.contrastive_loss = 0.0
        self.discriminative_loss = 0.0

    def _compute_contrastive_loss(self, contrastive_feat, labels, temperature=0.5):

        batch_size = contrastive_feat.size(0)

        # 归一化特征
        contrastive_feat = F.normalize(contrastive_feat, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(contrastive_feat, contrastive_feat.T)

        # 裁剪相似度值防止数值问题
        similarity_matrix = torch.clamp(similarity_matrix, min=-5, max=5)

        # 应用温度缩放
        similarity_matrix = similarity_matrix / temperature

        # 创建正负样本掩码
        labels_expanded = labels.unsqueeze(0).expand(batch_size, -1)
        positive_mask = (labels_expanded == labels_expanded.T).float()

        # 移除对角线
        positive_mask.fill_diagonal_(0)

        # 计算对比损失
        if positive_mask.sum() > 0:
            # 使用log-sum-exp技巧避免溢出
            max_sim = similarity_matrix.max(dim=1, keepdim=True)[0].detach()
            similarity_matrix = similarity_matrix - max_sim

            exp_similarity = torch.exp(similarity_matrix)

            # 计算正样本和所有样本的和
            positive_sum = (exp_similarity * positive_mask).sum(dim=1)
            all_sum = exp_similarity.sum(dim=1) - torch.diag(exp_similarity)

            # 安全的log计算
            eps = 1e-6
            loss = -torch.log(positive_sum / (all_sum + eps) + eps)

            # 裁剪异常值
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            loss = torch.where(torch.isinf(loss), torch.ones_like(loss) * 10.0, loss)
            loss = torch.clamp(loss, min=0.0, max=10.0)

            return loss.mean()
        else:
            return torch.tensor(0.0, device=contrastive_feat.device)

    def forward(self, x, labels=None):

        original_input = x.clone()

        # === 第一阶段：无条件生成 ===
        feats = self.patch_embed(x)
        enc_feats = []

        # Encoder
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)

        # Latent
        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)

        # 第一阶段输出（中间结果）
        temp_feats = feats
        temp_enc_feats = enc_feats.copy()  # 复制编码器特征
        for i, (upsample, fusion, _) in enumerate(self.dec):
            temp_feats = upsample(temp_feats)
            temp_feats = fusion(torch.cat([temp_feats, temp_enc_feats[-(i + 1)]], dim=1))

        temp_feats = self.refinement(temp_feats)
        intermediate_output = self.output(temp_feats) + original_input

        # === 计算传输残差 ===
        transport_residual = original_input - intermediate_output  # r̂₀ = y - x̂₀

        # === REGM：生成多尺度残差嵌入 ===
        residual_embeddings_dict, contrastive_feat = self.regm(transport_residual)

        # === 频域分析 ===
        residual_feats = self.patch_embed(transport_residual)
        freq_features, degradation_probs, freq_embedding = self.freq_analyzer(residual_feats)

        # === 第二阶段：条件生成（使用残差嵌入） ===
        self.total_loss = 0
        feats = self.patch_embed(x)  # 重新开始
        enc_feats = []

        # 重新编码
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)

        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)

        # 解码器
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))

            # 注入残差嵌入、频域嵌入、退化概率、任务ID
            feats = block(
                feats,
                freq_emb,
                degradation_probs,
                residual_embeddings_dict,
                labels  # 新增: 传递任务标签
            )
            self.total_loss += block.loss

        # 最终输出
        feats = self.refinement(feats)
        final_output = self.output(feats) + original_input

        # === 损失计算 ===
        if labels is not None:
            self.contrastive_loss = self._compute_contrastive_loss(contrastive_feat, labels)
            self.discriminative_loss = self.contrastive_loss
        else:
            self.contrastive_loss = 0.0
            self.discriminative_loss = 0.0

        # 安全的总损失计算
        num_blocks = max(sum(self.num_dec_blocks), 1)
        avg_moe_loss = self.total_loss / num_blocks

        # 裁剪各损失分量
        avg_moe_loss = torch.clamp(avg_moe_loss, min=0.0, max=10.0) if isinstance(avg_moe_loss, torch.Tensor) else min(
            max(avg_moe_loss, 0.0), 10.0)
        da_loss = torch.clamp(self.discriminative_loss, min=0.0, max=10.0) if isinstance(self.discriminative_loss,
                                                                                  torch.Tensor) else min(
            max(self.discriminative_loss, 0.0), 10.0)

        # 组合损失
        self.total_loss = avg_moe_loss + 0.1 * da_loss

        return final_output

    def get_transport_residual(self, degraded, restored):

        return degraded - restored


# Optimal Transport Divergence Loss (OTD Loss)
class OTDLoss(nn.Module):
    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def fourier_regularizer(self, residual, degradation_type='general'):

        b, c, h, w = residual.shape

        # === 裁剪输入 ===
        residual = torch.clamp(residual, -1.0, 1.0)

        # 计算FFT
        try:
            fft_residual = torch.fft.fft2(residual, norm='ortho')
            fft_magnitude = torch.abs(fft_residual)

            # === 裁剪FFT幅度，避免极值 ===
            fft_magnitude = torch.clamp(fft_magnitude, max=10.0)

            fft_shifted = torch.fft.fftshift(fft_magnitude, dim=(-2, -1))
        except Exception as e:
            print(f"Warning: FFT failed: {e}, returning zero loss")
            return torch.tensor(0.0, device=residual.device, requires_grad=True)

        # 创建频率权重
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, device=residual.device),
            torch.arange(w, device=residual.device),
            indexing='ij'
        )

        freq_y = (y - cy) / h
        freq_x = (x - cx) / w
        freq_radius = torch.sqrt(freq_y ** 2 + freq_x ** 2)

        # === 关键修复：处理批量的degradation_type ===
        if isinstance(degradation_type, torch.Tensor):
            # 如果是tensor，需要为batch中每个样本计算不同的权重
            reg_losses = []

            for i in range(b):
                # 获取单个样本的退化类型
                deg_type = degradation_type[i].item()

                # 根据退化类型选择权重
                if deg_type in [1, 'denoise', 'noise']:  # denoise
                    freq_weight = torch.sigmoid(10 * (freq_radius - 0.3))
                elif deg_type in [3, 'deblur', 'blur']:  # deblur
                    freq_weight = torch.exp(-((freq_radius - 0.2) ** 2) / (2 * 0.1 ** 2))
                else:  # dehaze (0), dedark (2) 或其他
                    freq_weight = torch.exp(-freq_radius / 0.2) * (1 - torch.exp(-freq_radius / 0.05))

                freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)
                weighted_spectrum = fft_shifted[i:i + 1] * freq_weight

                # 计算单个样本的损失
                sample_loss = torch.mean(weighted_spectrum ** 2).sqrt()
                reg_losses.append(sample_loss)

            # 平均所有样本的损失
            reg_loss = torch.stack(reg_losses).mean()

        else:
            # 如果不是tensor（单个样本或字符串），使用原来的逻辑
            if degradation_type in [1, 'denoise', 'noise']:
                freq_weight = torch.sigmoid(10 * (freq_radius - 0.3))
            elif degradation_type in [3, 'deblur', 'blur']:
                freq_weight = torch.exp(-((freq_radius - 0.2) ** 2) / (2 * 0.1 ** 2))
            else:
                freq_weight = torch.exp(-freq_radius / 0.2) * (1 - torch.exp(-freq_radius / 0.05))

            freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)
            weighted_spectrum = fft_shifted * freq_weight
            reg_loss = torch.mean(weighted_spectrum ** 2).sqrt()

        # === 归一化并裁剪 ===
        reg_loss = reg_loss / (h * w) ** 0.5
        reg_loss = torch.clamp(reg_loss, 0, 1.0)

        # === 检测NaN ===
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            return torch.tensor(0.0, device=residual.device, requires_grad=True)

        return reg_loss

    def forward(self, degraded, restored, target=None, degradation_type='general'):

        # 裁剪输入
        degraded = torch.clamp(degraded, 0, 1)
        restored = torch.clamp(restored, 0, 1)
        if target is not None:
            target = torch.clamp(target, 0, 1)

        # 传输残差
        transport_residual = degraded - restored
        transport_residual = torch.clamp(transport_residual, -1.0, 1.0)

        # 基础成本
        if target is not None:
            base_cost = self.l1_loss(restored, target)
        else:
            base_cost = self.l2_loss(restored, degraded)

        # FFT正则化
        freq_reg = self.fourier_regularizer(transport_residual, degradation_type)

        # 总损失
        total_loss = base_cost + self.lambda_reg * freq_reg

        # === 最终检查 ===
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Invalid OTD loss, using base cost only")
            total_loss = base_cost

        return total_loss, {
            'base_cost': base_cost.item(),
            'freq_reg': freq_reg.item(),
            'total': total_loss.item()
        }

