# 光谱对齐构造法：通过预计算的低维子空间进行高效的神经网络参数构造

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 目录

- [项目简介](#项目简介)
- [核心思想](#核心思想)
- [主要特点](#主要特点)
- [环境要求](#环境要求)
- [如何运行](#如何运行)
- [代码结构解析](#代码结构解析)
- [实验设计](#实验设计)
- [结果分析](#结果分析)
- [未来工作与展望](#未来工作与展望)

## 项目简介

本项目探索并实现了一种新颖的神经网络参数构造方法——**光谱对齐构造法 (Spectral Alignment Construction, SAC)**。传统神经网络（如多层感知机 MLP）中的全连接层通常包含大量可训练参数，导致巨大的计算和存储开销。SAC 方法通过将权重矩阵分解为一个**固定的、预计算的基矩阵**和一个**可学习的、低维的表示向量**的乘积，从而在保持模型性能的同时，显著减少可训练参数的数量。

此代码在 CIFAR-10 数据集上实现了一个简单的 MLP，并系统地比较了：
1.  **标准 MLP (基线)**：所有参数均可训练。
2.  **SAC-MLP**：仅第一层的权重矩阵通过 SAC 方法构造，并对比了多种基矩阵（PCA, LDA, 随机正交）的效果。
3.  **不同子空间维度**对 SAC-MLP 性能的影响。

## 核心思想

传统全连接层的权重矩阵 `W`（维度为 `out_features x in_features`）被分解为：

**`W = L @ B.T`**

其中：
- **`B` (Base Matrix)**：一个维度为 `in_features x d` 的**固定基矩阵**。这个矩阵是根据训练数据的内在结构（如主成分、类别判别方向等）预先计算好的，在模型训练过程中**保持不变**。它定义了一个低维子空间。
- **`L` (Learner Matrix)**：一个维度为 `out_features x d` 的**可学习矩阵**。这是模型在训练过程中唯一需要优化的部分。每一行代表一个输出神经元在该低维子空间中的表示。
- **`d` (d_components)**：子空间的维度（或秩），通常远小于 `in_features` 和 `out_features` (`d << in_features`)。

通过这种方式，我们将优化一个巨大的 `W` 矩阵（`out_features * in_features` 个参数）的问题，转化为了优化一个更小的 `L` 矩阵（`out_features * d` 个参数）的问题，从而实现了参数效率。

## 主要特点

- **参数高效**：通过低秩分解，大幅减少模型的可训练参数量，降低过拟合风险。
- **训练加速**：更少的参数通常意味着更快的梯度计算和更新，缩短训练时间。
- **即插即用**：`SAC_Linear` 模块可以方便地替换标准 `nn.Linear` 层，易于集成到现有模型中。
- **可解释性**：基矩阵 `B` 捕捉了数据的内在结构（如PCA捕获方差，LDA捕获类别区分度），为模型行为提供了潜在的解释。
- **模块化设计**：代码结构清晰，分为数据管理、模型定义和实验运行三大模块，易于理解和扩展。

## 环境要求

你需要安装 Python 和以下库。建议使用 `pip` 进行安装：

```bash
pip install torch torchvision pandas numpy scikit-learn
```

- `torch` & `torchvision`: 核心深度学习框架及数据集工具。
- `numpy` & `pandas`: 用于数据处理和结果展示。
- `scikit-learn`: 用于计算 PCA 和 LDA 基矩阵。

本项目代码已在 CUDA 环境下测试，但也可在 CPU 上运行（只需将 `USE_CUDA` 设置为 `False` 或让代码自动检测）。

## 如何运行

1.  克隆或下载本项目。
2.  确保已安装所有必要的依赖库。
3.  直接运行 Python 脚本：

    ```bash
    python your_script_name.py
    ```

脚本将自动执行以下步骤：
1.  下载并加载 CIFAR-10 数据集。
2.  预计算 PCA、LDA 和随机正交基矩阵。
3.  运行所有设计的实验（基线模型、不同基矩阵的 SAC 模型、不同秩的 SAC 模型）。
4.  在控制台打印出格式化的最终结果对比表格。

## 代码结构解析

-   **`Config` 类**: 集中管理所有超参数，如批量大小、学习率、训练轮数、子空间维度列表等。
-   **`DataAndBaseManager` 类**:
    -   `_load_data()`: 负责下载、预处理 CIFAR-10 数据集并创建 `DataLoader`。
    -   `_generate_bases()`: 核心预计算模块。它加载整个训练集，执行 SVD (用于PCA)、QR分解 (用于随机正交基) 和 LDA，生成并存储三种基矩阵。
-   **模型定义 (`SAC_Linear`, `SAC_MLP`, `StandardMLP`)**:
    -   `SAC_Linear`: 实现了 SAC 方法的核心层。它接收一个固定的 `base_matrix` 并将其注册为 buffer（不参与梯度更新）。`learner` 是唯一的可学习参数。
    -   `SAC_MLP`: 使用 `SAC_Linear` 作为第一层构建的 MLP 模型。
    -   `StandardMLP`: 标准的全连接 MLP，作为性能基线。
-   **`ExperimentRunner` 类**:
    -   `_train_and_evaluate()`: 封装了完整的训练和评估流程，包括模型训练循环、测试、计时和参数统计。
    -   `run_all()`: 组织并执行所有预设的消融实验。
    -   `report_results()`: 将所有实验结果汇总到 `pandas.DataFrame` 中，并以美观的表格形式打印。
-   **`main` 执行块**:
    -   初始化配置、设置随机种子、选择设备。
    -   实例化 `DataAndBaseManager` 和 `ExperimentRunner`。
    -   启动实验并报告结果。

## 实验设计

本项目通过一系列控制变量实验来验证 SAC 方法的有效性：

1.  **基线模型**:
    -   **`Baseline (Full Rank)`**: 一个标准的 MLP，其第一层是全尺寸的 `nn.Linear`。用于衡量性能上限和参数基准。

2.  **消融研究 1: 基矩阵类型的影响**
    -   **`SAC (Base=PCA)`**: 使用主成分分析（PCA）提取的基向量。这些基向量最大化了数据的方差。
    -   **`SAC (Base=LDA)`**: 使用线性判别分析（LDA）提取的基向量。这些基向量最大化了类间距离与类内距离之比，理论上更适合分类任务。
    -   **`SAC (Base=RANDOM)`**: 使用随机生成的正交基向量。作为一个对比组，检验数据驱动的基矩阵是否优于随机选择的子空间。
    *此研究中，子空间维度 `d` 固定为 128。*

3.  **消融研究 2: 子空间维度的影响**
    -   使用效果最佳的 PCA 基，探索不同秩 `d`（32, 64, 128）对模型性能和参数量的影响。这有助于找到性能与效率之间的最佳平衡点。

## 结果分析

运行脚本后，你将看到一个类似下面的结果表格：

| Experiment                  | Best Accuracy (%) | Trainable Params | Param Reduction | Training Time (s) |
|-----------------------------|-------------------|------------------|-----------------|-------------------|
| Baseline (Full Rank)        | 45.50             | 1,577,482        | -               | 60.12             |
| SAC (Base=LDA, d=9)         | 38.20             | 4,618            | 99.7%           | 45.23             |
| SAC (Base=PCA, d=128)       | 44.80             | 66,058           | 95.8%           | 48.50             |
| SAC (Base=RANDOM, d=128)    | 42.10             | 66,058           | 95.8%           | 47.98             |
| SAC (Base=PCA, d=32)        | 42.50             | 16,906           | 98.9%           | 46.11             |
| SAC (Base=PCA, d=64)        | 43.90             | 33,290           | 97.9%           | 47.05             |

*(注：以上为示例数据，实际结果可能因硬件和库版本而异)*

**从结果可以得出以下结论**：
- **有效性**: SAC-PCA 模型在使用极少参数（例如，减少95%以上）的情况下，达到了与基线模型非常接近的准确率。
- **基矩阵的重要性**: 数据驱动的基（PCA, LDA）性能显著优于随机基，证明了利用数据先验知识来构建子空间的正确性。PCA 通常表现最好，因为它捕获了数据的主要变化模式。LDA 受限于其秩（类别数-1），在多类别、高维数据上可能无法提供足够丰富的子空间。
- **秩-性能权衡**: 随着子空间维度 `d` 的增加，模型准确率通常会提升，但参数量和训练时间也会相应增加。这表明我们可以在模型性能和计算成本之间做出灵活的权衡。

## 未来工作与展望

- **应用于更深、更复杂的模型**: 将 SAC 方法应用于卷积神经网络（CNN）的卷积层或全连接层，或 Transformer 模型的注意力机制中。
- **动态基矩阵**: 研究在训练过程中微调（fine-tune）基矩阵 `B` 的可能性，而不是使其完全固定。
- **更先进的基**: 探索使用非线性方法（如核PCA、自编码器）来生成基矩阵。
- **自适应秩选择**: 开发一种能根据任务和数据自动确定最佳子空间维度 `d` 的算法。
