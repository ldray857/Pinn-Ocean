# Pinn-Ocean: 耦合移位窗口自注意力与物理约束连续坐标的海洋三维温盐场重建框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GeoAI](https://img.shields.io/badge/Domain-GeoAI%20%26%20海洋人工智能-brightgreen.svg)]()
[![曾宪梓计划](https://img.shields.io/badge/专项-曾宪梓教育基金会拔尖创新人才培育计划-orange.svg)]()

[English](README.md) | [中文说明文档](README.zh.md)

---

## 一、 项目背景与科学问题

利用海表高频二维卫星观测（海表温度 SST、海面高度异常 SLA、海表盐度 SSS、风场等）高保真反演海洋次表层（0–1000m）三维温度和盐度场，对全球气候事件（AMOC、ENSO）预警、水下声传播路径模拟以及国家海洋国土安全保障具有重大战略价值。

传统原位观测网络（如 Argo 浮标网）在时空覆盖上存在明显的稀疏性与滞后性；而传统深度学习方法（如二维 CNN 逐层堆叠）常面临三大瓶颈：
1. 感受野受限：难以捕获大洋多尺度动力关联与中尺度涡旋空间遥相关；
2. 违背物理法则：纯数据驱动的“黑盒”模型在观测盲区易出现违背热力学常识的现象，如深层海水密度小于表层的反常密度倒置与异常逆温；
3. 有限差分离散误差：逐层网格计算使得垂直导数截断误差随深度累积，深层反演精度急剧下降。

针对上述瓶颈，Pinn-Ocean 耦合 Swin Transformer 空间自注意力与物理信息神经网络（PINN）连续坐标解码，通过引入 PyTorch autograd 自动微分机制，将 TEOS-10 海水状态方程与海水层结稳定条件作为物理损失约束，提升三维反演精度与物理一致性。

---

## 二、 训练数据来源

数据主要来自欧盟哥白尼海洋服务（CMEMS）与国际 Argo 计划。

### 2.1 区域与时间
- 空间范围：西北太平洋（145°E–165°E, 30°N–40°N），深度 0–1000m。为开阔大洋，无陆地掩码。
- 时间范围：2013 年 1 月至 2021 年 12 月（月平均，共 108 个月）。
  - 训练集：2013–2018 年（72 个月）
  - 验证集：2019–2020 年（24 个月）
  - 测试集：2021 年（12 个月）

### 2.2 数据集清单

| 变量 | 数据集 / 来源 | 分辨率 | 深度 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| SLA（海面高度异常） | cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m | 0.125° | 表层 | 输入特征 |
| SST（海表温度） | METOFFICE-GLO-SST-L4-REP-OBS-SST (OSTIA) | 0.05° | 表层 | 输入特征 |
| SSS（海表盐度） | cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c | 0.25° | 表层 | 输入特征 |
| Wind U/V（海面风场） | cmems_obs-wind_glo_phy_my_l4_P1M | 0.25° | 表层 | 输入特征 |
| 经度 / 纬度 / 月份 | 网格坐标与周期月份 | - | 表层 | 输入特征 |
| 位温、实用盐度 (thetao, so) | cmems_mod_glo_phy_my_0.083deg_P1M-m (GLORYS12V1) | 1/12° (~0.083°) | 0–1000m (25层) | 训练真值 |
| 温盐原位剖面 | 国际 Argo 计划 / 中国 Argo 实时资料中心 | 离散剖面 | 0–1000m | 独立测试验证 |

---

## 三、 核心算法与网络架构

<div align="center">

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'background': '#FFFFFF',
    'primaryColor': '#FFFFFF',
    'primaryBorderColor': '#CBD5E1',
    'primaryTextColor': '#0F172A',
    'secondaryColor': '#F8FAFC',
    'tertiaryColor': '#FFFFFF',
    'mainBkg': '#FFFFFF',
    'clusterBkg': '#FFFFFF',
    'clusterBorder': '#E2E8F0',
    'lineColor': '#475569',
    'textColor': '#0F172A',
    'edgeLabelBackground': '#FFFFFF',
    'fontFamily': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  }
}}%%
flowchart TD
    subgraph S1 [" "]
        direction TB
        H1["1. 海表多源动力输入<br/>(8 通道)"]
        I1["动力要素：SST / SLA / SSS"]
        I2["边界强迫：海表风应力 (Wind U / V 抽吸)"]
        I3["时空坐标：经纬度 Lon, Lat / 月份周期 Month"]
        H1 --> I1 --- I2 --- I3
    end

    subgraph S2 [" "]
        direction TB
        H2["2. 空间自注意力编码器<br/>(Swin Transformer)"]
        E1["Patch Embedding：映射至隐空间隐维度 C"]
        E2["W-MSA / SW-MSA<br/>局部与跨窗口自注意力"]
        E3["海表高阶空间隐特征 Token 矩阵 F_surf"]
        H2 --> E1 ==> E2 ==> E3
    end

    subgraph S3 [" "]
        direction TB
        H3["3. 连续坐标 PINN 解码头<br/>(隐式神经表征)"]
        D1["垂直深度自变量<br/>z ∈ [0, 1000m] (求导启用)"]
        D2["特征级联拼接：[F_surf, z] 联合表征"]
        D3["连续 MLP 解码器：Tanh 连续平滑映射"]
        H3 --> D1 --> D2 ==> D3
    end

    subgraph S4 [" "]
        direction TB
        H4["4. 三维立体物理场预测<br/>(0–1000m)"]
        O1["三维位温场重构 T_hat<br/>(混合层 / 温跃层 / 深层)"]
        O2["三维实用盐度场重构 S_hat<br/>(次表层高盐 / 中层低盐舌)"]
        H4 --> O1 --- O2
    end

    subgraph S5 [" "]
        direction TB
        H5["5. 物理先验约束与自适应优化闭环"]
        P1["数据保真损失 L_data：GLORYS12V1 全深度 MSE 监督"]
        P2["温度递减约束 L_phy,T<br/>Autograd 求导 dT/dz ≤ 0"]
        P3["层结稳定约束 L_phy,rho<br/>TEOS-10 状态方程 drho/dz ≥ 0"]
        Opt["自适应多目标对偶平衡<br/>动态权衡与联合更新"]
        H5 --> P1 --- P2 --- P3 ==> Opt
    end

    I3 ==>|海表特征张量 X_surf| H2
    E3 ==>|空间隐特征 F_surf| H3
    D3 ==>|连续深度立体解码| H4
    O2 ==>|三维物理场全域约束| H5
    Opt -. 闭环物理梯度反传 .-> H2

    style S1 fill:#FFFFFF,stroke:#0284C7,stroke-width:1.5px,stroke-dasharray: 4 4,rx:8px,ry:8px
    style S2 fill:#FFFFFF,stroke:#7C3AED,stroke-width:1.5px,stroke-dasharray: 4 4,rx:8px,ry:8px
    style S3 fill:#FFFFFF,stroke:#059669,stroke-width:1.5px,stroke-dasharray: 4 4,rx:8px,ry:8px
    style S4 fill:#FFFFFF,stroke:#D97706,stroke-width:1.5px,stroke-dasharray: 4 4,rx:8px,ry:8px
    style S5 fill:#FFFFFF,stroke:#E11D48,stroke-width:1.5px,stroke-dasharray: 4 4,rx:8px,ry:8px

    classDef headStyle1 fill:#0284C7,stroke:#0284C7,stroke-width:1.5px,color:#FFFFFF,rx:6px,ry:6px;
    classDef headStyle2 fill:#7C3AED,stroke:#7C3AED,stroke-width:1.5px,color:#FFFFFF,rx:6px,ry:6px;
    classDef headStyle3 fill:#059669,stroke:#059669,stroke-width:1.5px,color:#FFFFFF,rx:6px,ry:6px;
    classDef headStyle4 fill:#D97706,stroke:#D97706,stroke-width:1.5px,color:#FFFFFF,rx:6px,ry:6px;
    classDef headStyle5 fill:#E11D48,stroke:#E11D48,stroke-width:1.5px,color:#FFFFFF,rx:6px,ry:6px;

    classDef inputStyle fill:#F0F9FF,stroke:#0284C7,stroke-width:1.5px,color:#0369A1,rx:6px,ry:6px;
    classDef encStyle fill:#F5F3FF,stroke:#7C3AED,stroke-width:1.5px,color:#5B21B6,rx:6px,ry:6px;
    classDef pinnStyle fill:#ECFDF5,stroke:#059669,stroke-width:1.5px,color:#047857,rx:6px,ry:6px;
    classDef outStyle fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#B45309,rx:6px,ry:6px;
    classDef phyStyle fill:#FFF1F2,stroke:#E11D48,stroke-width:1.5px,color:#BE123C,rx:6px,ry:6px;

    class H1 headStyle1;
    class H2 headStyle2;
    class H3 headStyle3;
    class H4 headStyle4;
    class H5 headStyle5;

    class I1,I2,I3 inputStyle;
    class E1,E2,E3 encStyle;
    class D1,D2,D3 pinnStyle;
    class O1,O2 outStyle;
    class P1,P2,P3,Opt phyStyle;
```

</div>

### 3.1 核心前向映射函数

网络将海表二维多动力参数与连续垂直深度坐标 $z$ 显式融合，建立高维连续重构映射：

$$
[\hat{T}, \hat{S}] = f_\theta(\text{SST}, \text{SLA}, \text{SSS}, \text{SSW}_U, \text{SSW}_V, \text{Lon}, \text{Lat}, \text{Month}, z)
$$

其中 $z \in [0, 1000\text{ m}]$ 为显式自变量，赋予模型在垂直方向上任意连续深度的解析能力。

### 3.2 空间移位窗口自注意力 (Swin Transformer)

利用局部窗口多头自注意力（W-MSA）与跨窗口移位自注意力（SW-MSA）交替提取海表特征，计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V
$$

其中 $B$ 为相对位置偏置矩阵，使得模型能够在大洋尺度下高效建模长距离空间遥相关。

### 3.3 物理先验约束损失系统

**温度垂直单调递减约束**（$\mathcal{L}_{\mathrm{phy}, T}$）：

$$
\mathcal{L}_{\mathrm{phy}, T} = \frac{1}{N} \sum_{i=1}^N \mathrm{ReLU}\left(\frac{\partial \hat{T}_i}{\partial z} + \epsilon\right)
$$

**TEOS-10 层结稳定性与防密度倒置约束**（$\mathcal{L}_{\mathrm{phy}, \rho}$）：

基于海水状态方程 $\hat{\rho} = f_{\mathrm{TEOS\text{-}10}}(\hat{S}, \hat{T}, P)$，惩罚违背静力平衡的密度倒置：

$$
\mathcal{L}_{\mathrm{phy}, \rho} = \frac{1}{N} \sum_{i=1}^N \mathrm{ReLU}\left(-\frac{\partial \hat{\rho}_i}{\partial z}\right)
$$

**自适应多目标联合优化**（$\mathcal{L}_{\mathrm{total}}$）：

$$
\mathcal{L}_{\mathrm{total}} = \exp(-\omega_1) \mathcal{L}_{\mathrm{data}} + \omega_1 + \exp(\omega_2) \mathcal{L}_{\mathrm{phy}} + \omega_2
$$

其中 $\omega_1, \omega_2$ 为可学习的对偶参数，在训练中自适应平衡数据保真度与物理约束。

---

## 四、 工程目录结构

```text
Pinn-Ocean/
├── configs/
│   ├── __init__.py
│   └── default_config.py      # 模型维度、物理损失超参数及路径配置文件
├── pinn_ocean/                # 核心算法与模型包
│   ├── __init__.py
│   ├── models/                # 神经网络架构
│   │   ├── __init__.py
│   │   ├── swin_blocks.py     # Swin Transformer 基础模块 (W-MSA/SW-MSA/PatchMerge/Expand)
│   │   └── swin_ocean_pinn.py # Swin-Ocean-PINN 端到端核心拓扑架构
│   ├── losses/                # 物理损失与自适应优化
│   │   ├── __init__.py
│   │   ├── physics_loss.py    # 基于 Autograd 的温度梯度与密度层结稳定物理损失算子
│   │   └── adaptive_loss.py   # 自适应多目标损失平衡算法
│   ├── datasets/              # 数据流管道与接口
│   │   ├── __init__.py
│   │   ├── downloader.py      # CMEMS 数据子集接口封装模块
│   │   └── ocean_dataset.py   # 支持 NetCDF4 / Xarray 的 8 通道空间网格数据加载管道
│   ├── utils/                 # 物理计算与评估指标
│   │   ├── __init__.py
│   │   ├── teos10.py          # 全链路可微的 TEOS-10 / UNESCO 海水状态方程实现
│   │   └── metrics.py         # RMSE、MAE、R2 与海洋混合层深度 (MLD) 计算工具
│   └── visualization/         # 顶刊级模块化科学绘图子包
│       ├── __init__.py
│       ├── profiles.py        # 代表站位垂直剖面重构对比绘图
│       ├── ts_diagram.py      # 温盐关系 (T-S Diagram) 物理一致性检验绘图
│       ├── scatter_density.py # 全深度 Hexbin 散点密度与拟合优度 R^2 绘图
│       └── mld.py             # 上混合层深度 (MLD) 界面反演对比绘图
├── tests/                     # 自动化单元测试套件
│   ├── __init__.py
│   └── test_pipeline.py       # 硬件、Autograd、TEOS-10 及前向反向端到端测试
├── checkpoints/               # 训练产出的最优模型权重 (.pth)
├── data/                      # 真实海洋卫星观测与 GLORYS 3D 再分析数据 (NetCDF)
│   └── 2020/                  # 2020 年度 5 核心要素已下载数据集
├── results/                   # 自动输出的 300 DPI 高清科研图件与报表
├── download_data.py           # CMEMS 开阔太平洋多源遥感与 3D 再分析数据自动化下载脚本
├── train.py                   # 完整模型训练主入口
├── evaluate.py                # 检查点评估与分层物理指标验证脚本
├── predict.py                 # 全域三维立体反演与标准 NetCDF4 数据资产导出脚本
├── visualize.py               # 一键生成全部科研图件的主入口
├── demo_test.py               # 独立自检单元测试快速入口
├── requirements.txt           # 运行环境依赖清单
├── setup.py                   # Python 包安装与打包脚本
├── LICENSE                    # MIT 开源许可证
├── README.md                  # 英文项目说明
└── README.zh.md               # 中文项目说明
```

---

## 五、 环境准备与安装

### 步骤 1：克隆仓库
```bash
git clone https://github.com/ldray857/Pinn-Ocean.git
cd Pinn-Ocean
```

### 步骤 2：创建并激活 Conda 虚拟环境
```bash
conda create -n pinn_ocean python=3.10 -y
conda activate pinn_ocean
```

### 步骤 3：安装依赖库
```bash
pip install -r requirements.txt
```

---

## 六、 快速上手与验证

### 6.1 数据获取（开阔太平洋 2013–2021 年数据）
本项目提供标准脚本直接从 CMEMS 抓取西北太平洋纯深海大洋无陆地区域（$145^\circ\text{E} - 165^\circ\text{E}, 30^\circ\text{N} - 40^\circ\text{N}$）的月度融合数据：
```bash
# 预览下载计划与网格参数（无需网络请求）
python download_data.py --dry_run

# 正式下载核心数据 (需预先运行 copernicusmarine login)
python download_data.py --targets sla glorys_3d
```

### 6.2 一键单元自检（无需外部数据）
该测试通过仿真合成批次，对前向推理、Autograd 自动微分链、海水密度求导及反向梯度传播进行闭环校验：
```bash
python demo_test.py
```

### 6.3 启动模型训练
在本地或云端算力节点针对区域或大洋尺度的 NetCDF 数据启动物理训练：
```bash
python train.py --epochs 200 --batch_size 4 --lr 3e-4 --sampling_points 800
```

### 6.4 模型性能评估与指标检验
加载最优检查点并在测试集上计算全深度温盐指标（RMSE, MAE, R²）：
```bash
python evaluate.py --checkpoint checkpoints/swin_ocean_pinn_best.pth
```

### 6.5 全域三维立体反演与数据资产导出
将训练成果用于全空间三维反演并导出为 CF 标准的 NetCDF4 成果文件（便于导入 GIS 软件）：
```bash
python predict.py --data_dir data/2020 --output_file data/2020/pacific_reconstructed_3d.nc
```

### 6.6 顶刊级科研图件一键批量生成
自动生成 4 组符合学术论文与中期汇报规范的 300 DPI 高清评估图件：
```bash
python visualize.py --data_dir data/2020 --output_dir results
```

---

## 七、 参考文献与致谢

如果本开源工作或代码结构对你的学术研究有所帮助，欢迎引用相关工作：

```bibtex
@article{wang2026cross,
  title={Cross-scale 3-D thermohaline modeling via dual-residual swin transformer with multisource ocean observations},
  author={Wang, An and Tang, Zhiwei and Huang, Zhanchao and Xia, Xiang-Gen and Su, Hua},
  journal={International Journal of Digital Earth},
  volume={19},
  number={1},
  pages={2607902},
  year={2026},
  publisher={Taylor \& Francis}
}

@article{shao2024attention,
  title={Optimized Attention-enhanced Physics-guided Neural Network for Satellite-based Ocean Subsurface Temperature Predicting},
  author={Shao, J. and Wu, Sensen and Chen, Y. and others},
  journal={Remote Sensing of Environment / IEEE TGRS},
  year={2024}
}
```

*   **项目负责人**：雷堤（浙江大学地球科学学院 地理信息科学专业 2024级）
*   **指导教师**：吴森森 研究员（浙江大学地球科学学院）
*   **立项专项**：曾宪梓教育基金会第一期“拔尖创新人才培育计划”专项

---

## 八、 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可。
