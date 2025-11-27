# AlphaFold 蛋白质结构预测：原理与实现

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**一个教学向的AlphaFold简化实现，帮助理解AI蛋白质结构预测的核心原理**

[课程文档](蛋白第二课-整理版.md) • [示例代码](alphafold_reproduction.py) • [可视化结果](#可视化结果)

</div>

---

## 📖 项目简介

本项目是**AI蛋白设计第二课**的配套代码实现，通过简化版的AlphaFold模型，帮助你深入理解：

- 🧬 **多序列比对(MSA)** 在蛋白质结构预测中的关键作用
- 🤖 **注意力机制(Evoformer)** 如何捕捉残基间的相互作用
- 📐 **结构预测模块** 如何从序列信息生成3D坐标
- 🎯 **AlphaFold数据库** 的查询和使用方法
- 📊 **结构质量评估** 的多种可视化手段

> ⚠️ **注意**：这是一个教学实现，旨在帮助理解原理，而非生产级别的工具。真实的AlphaFold2有更复杂的架构和数百万参数。

---

## ✨ 主要特性

- ✅ **端到端实现**：从序列输入到3D结构预测的完整流程
- ✅ **详细注释**：每个关键步骤都有深入的代码注释和原理解释
- ✅ **可视化丰富**：6种专业图表展示预测结果
- ✅ **模块化设计**：清晰的类结构，易于理解和扩展
- ✅ **实战案例**：包含新冠病毒蛋白(NSP10)的预测示例

---

## 🎯 可视化结果

运行代码后将生成以下6种可视化图表：

| 图表 | 说明 | 文件名 |
|------|------|--------|
| 🔥 **距离矩阵热图** | 展示残基间的空间距离分布 | `1_predicted_distance_matrix.png` |
| 🎯 **置信度热图** | 评估每对残基距离预测的可靠性 | `2_distance_prediction_confidence.png` |
| 📊 **保守性相关性** | MSA保守性与预测置信度的关系 | `3_conservation_vs_confidence.png` |
| 🗺️ **2D结构投影** | 蛋白质折叠的平面可视化 | `4_predicted_structure_projection.png` |
| 📈 **距离分布直方图** | 结构紧凑度的统计分析 | `5_predicted_distance_distribution.png` |
| 📉 **逐残基置信度** | 沿序列的预测质量分布 | `6_per_residue_confidence.png` |

<details>
<summary>📷 查看示例图片</summary>

所有可视化结果都保存为高分辨率PNG图片，并附有详细的解读说明文本文件。

</details>

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.12+ 
- **操作系统**: Windows / macOS / Linux
- **内存**: 建议 4GB+

### 2. 安装步骤

#### 方法一：使用 Python venv（推荐）

```bash
# 克隆项目
git clone https://github.com/your-username/alphafold-reproduction.git
cd alphafold-reproduction

# 创建虚拟环境
python -m venv aidd_env

# 激活虚拟环境
# Windows PowerShell:
.\aidd_env\Scripts\Activate.ps1

# Windows CMD:
.\aidd_env\Scripts\activate.bat

# macOS/Linux:
source aidd_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方法二：使用 Conda

```bash
# 创建conda环境
conda create -n alphafold python=3.12

# 激活环境
conda activate alphafold

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行代码

```bash
# 运行完整的AlphaFold预测流程
python alphafold_reproduction.py
```

**预期输出：**
```
AlphaFold结构预测实现环境准备完成
✓ MSA生成完成
✓ 注意力机制初始化
✓ 结构预测完成
✓ 质量评估完成
✓ 可视化图表已保存
```

---
