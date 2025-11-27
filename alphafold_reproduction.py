# -*- coding: utf-8 -*-

# 人工智能药物设计第11节课：AI蛋白设计第二课：AlphaFold结构预测原理与代码实现

# ====================================
# 2.1 环境准备和依赖安装
# ====================================

# 安装依赖包
# pip install biopython numpy pandas torch torchvision requests plotly py3Dmol biotite

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
# from biotite.structure import AtomArray # This is commented out as it might not be used directly in the provided snippet
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
import warnings
import os
from matplotlib.transforms import Bbox
warnings.filterwarnings('ignore')

# 设置随机种子确保可重现性
torch.manual_seed(42)
np.random.seed(42)

print("AlphaFold结构预测实现环境准备完成")

# ====================================
# 2.2 第一步：多序列比对(MSA)生成与处理
# ====================================

class MSAGenerator:
    """多序列比对生成器 - AlphaFold的核心输入"""
    
    def __init__(self):
        self.gap_penalty = -1
        self.mismatch_penalty = -1
        self.match_reward = 2
    
    def fetch_homologous_sequences(self, query_sequence, max_sequences=100):
        """
       获取同源序列（模拟真实的MSA数据库搜索）
       在实际应用中，这会连接到UniRef、MGnify等数据库
       """
        print("正在搜索同源序列...")
        
        # 模拟同源序列生成（实际中使用HHblits、Jackhmmer等工具）
        homologs = []
        
        # 添加原始序列
        homologs.append({
            'sequence': query_sequence,
            'identity': 100.0,
            'coverage': 100.0,
            'organism': 'Query'
        })
        
        # 生成模拟同源序列
        for i in range(max_sequences - 1):
            # 通过引入突变生成同源序列
            identity = np.random.uniform(30, 95)  # 序列一致性
            mutated_seq = self.introduce_mutations(query_sequence, identity)
            
            homologs.append({
                'sequence': mutated_seq,
                'identity': identity,
                'coverage': np.random.uniform(80, 100),
                'organism': f'Species_{i+1}'
            })
        
        return homologs
    
    def introduce_mutations(self, sequence, target_identity):
        """引入突变以达到目标序列一致性"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        seq_list = list(sequence)
        seq_length = len(sequence)
        
        # 计算需要突变的位点数
        mutations_needed = int(seq_length * (100 - target_identity) / 100)
        
        # 随机选择突变位点
        mutation_positions = np.random.choice(seq_length, 
                                            min(mutations_needed, seq_length), 
                                           replace=False)
        
        for pos in mutation_positions:
            # 选择不同的氨基酸
            original_aa = seq_list[pos]
            possible_aa = [aa for aa in amino_acids if aa != original_aa]
            seq_list[pos] = np.random.choice(possible_aa)
        
        return ''.join(seq_list)
    
    def create_msa_matrix(self, homologs):
        """创建MSA矩阵表示"""
        sequences = [h['sequence'] for h in homologs]
        max_length = max(len(seq) for seq in sequences)
        
        # 填充序列到相同长度
        aligned_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                seq += '-' * (max_length - len(seq))
            aligned_sequences.append(seq)
        
        # 转换为数字编码
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-')}
        
        msa_matrix = np.zeros((len(aligned_sequences), max_length))
        for i, seq in enumerate(aligned_sequences):
            for j, aa in enumerate(seq):
                msa_matrix[i, j] = aa_to_idx.get(aa, 20)  # 20 for unknown
        
        return msa_matrix, aligned_sequences
    
    def calculate_conservation_scores(self, msa_matrix):
        """计算保守性评分"""
        conservation_scores = []
        n_sequences, seq_length = msa_matrix.shape
        
        for pos in range(seq_length):
            column = msa_matrix[:, pos]
            # 计算香农熵作为保守性指标
            unique, counts = np.unique(column, return_counts=True)
            probabilities = counts / n_sequences
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            conservation = 1 - (entropy / np.log2(21))  # 21种可能状态
            conservation_scores.append(conservation)
        
        return np.array(conservation_scores)

# 测试MSA生成器
msa_generator = MSAGenerator()

# 使用第一课的泛素序列
ubiquitin_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

print(f"查询序列长度: {len(ubiquitin_sequence)}")
print(f"查询序列: {ubiquitin_sequence}")

# 生成MSA
homologs = msa_generator.fetch_homologous_sequences(ubiquitin_sequence, max_sequences=50)
msa_matrix, aligned_sequences = msa_generator.create_msa_matrix(homologs)
conservation_scores = msa_generator.calculate_conservation_scores(msa_matrix)

print(f"\nMSA统计信息:")
print(f"同源序列数量: {len(homologs)}")
print(f"MSA矩阵形状: {msa_matrix.shape}")
print(f"平均序列一致性: {np.mean([h['identity'] for h in homologs]):.1f}%")


# ====================================
# 2.3 第二步：实现简化版Evoformer注意力机制
# ====================================

class SimplifiedEvoformer(nn.Module):
    """简化版Evoformer - AlphaFold的核心组件"""
    
    def __init__(self, msa_dim=21, pair_dim=64, hidden_dim=128, num_heads=8):
        super().__init__()
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # MSA表示层
        self.msa_embedding = nn.Embedding(msa_dim, hidden_dim)
        
        # 残基对表示初始化
        self.pair_embedding = nn.Linear(2 * hidden_dim, pair_dim)
        
        # 注意力机制
        self.msa_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.pair_attention = nn.MultiheadAttention(pair_dim, num_heads, batch_first=True)
        
        # 前馈网络
        self.msa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        self.pair_ffn = nn.Sequential(
            nn.Linear(pair_dim, 4 * pair_dim),
            nn.ReLU(),
            nn.Linear(4 * pair_dim, pair_dim)
        )
        
        # 层归一化
        self.msa_norm1 = nn.LayerNorm(hidden_dim)
        self.msa_norm2 = nn.LayerNorm(hidden_dim)
        self.pair_norm1 = nn.LayerNorm(pair_dim)
        self.pair_norm2 = nn.LayerNorm(pair_dim)
    
    def forward(self, msa, pair_representation=None):
        batch_size, n_sequences, seq_length = msa.shape
        
        # MSA嵌入
        msa_embed = self.msa_embedding(msa.long())  # [batch, n_seq, seq_len, hidden_dim]
        
        # 重塑用于注意力计算
        msa_reshaped = msa_embed.view(-1, seq_length, self.hidden_dim)  # [batch*n_seq, seq_len, hidden_dim]
        
        # MSA行注意力（序列内残基间的注意力）
        msa_attended, _ = self.msa_attention(msa_reshaped, msa_reshaped, msa_reshaped)
        msa_attended = self.msa_norm1(msa_reshaped + msa_attended)
        
        # MSA前馈网络
        msa_output = self.msa_ffn(msa_attended)
        msa_output = self.msa_norm2(msa_attended + msa_output)
        
        # 重塑回原来的形状
        msa_output = msa_output.view(batch_size, n_sequences, seq_length, self.hidden_dim)
        
        # 生成残基对表示
        if pair_representation is None:
            # 使用MSA的第一行（查询序列）生成初始残基对表示
            query_repr = msa_output[:, 0, :, :]  # [batch, seq_len, hidden_dim]
            
            # 外积操作生成残基对特征
            pair_repr = torch.zeros(batch_size, seq_length, seq_length, self.pair_dim)
            for i in range(seq_length):
                for j in range(seq_length):
                    concat_repr = torch.cat([query_repr[:, i, :], query_repr[:, j, :]], dim=-1)
                    pair_repr[:, i, j, :] = self.pair_embedding(concat_repr)
        else:
            pair_repr = pair_representation
        
        # 残基对注意力
        pair_reshaped = pair_repr.view(-1, seq_length, self.pair_dim)
        pair_attended, _ = self.pair_attention(pair_reshaped, pair_reshaped, pair_reshaped)
        pair_attended = self.pair_norm1(pair_reshaped + pair_attended)
        
        # 残基对前馈网络
        pair_output = self.pair_ffn(pair_attended)
        pair_output = self.pair_norm2(pair_attended + pair_output)
        
        # 重塑回原来的形状
        pair_output = pair_output.view(batch_size, seq_length, seq_length, self.pair_dim)
        
        return msa_output, pair_output

# 初始化模型
evoformer = SimplifiedEvoformer()

# 准备输入数据
msa_tensor = torch.from_numpy(msa_matrix).unsqueeze(0).float()  # 添加batch维度
print(f"\nMSA张量形状: {msa_tensor.shape}")

# 前向传播
with torch.no_grad():
    msa_repr, pair_repr = evoformer(msa_tensor)

print(f"MSA表示形状: {msa_repr.shape}")
print(f"残基对表示形状: {pair_repr.shape}")


# ====================================
# 2.4 第三步：结构预测模块
# ====================================

class StructurePredictionModule(nn.Module):
    """结构预测模块 - 从表示到3D坐标"""
    
    def __init__(self, pair_dim=64, hidden_dim=128):
        super().__init__()
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        
        # 距离预测头
        self.distance_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),  # 64个距离bins
            nn.Softmax(dim=-1)
        )
        
        # 角度预测头（phi, psi, omega）
        self.angle_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * 36),  # 3个角度，每个36个bins
        )
        
        # 坐标生成层
        self.coord_generator = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # x, y, z坐标
        )
        
        # 距离bins（2-22埃，间隔0.5埃）
        self.distance_bins = torch.linspace(2.0, 22.0, 64)
        # 角度bins（-180到180度）
        self.angle_bins = torch.linspace(-180, 180, 36)
    
    def predict_distances(self, pair_representation):
        """预测残基间Ca-Ca距离"""
        batch_size, seq_len, _, pair_dim = pair_representation.shape
        
        # 距离预测
        distance_logits = self.distance_head(pair_representation)
        
        # 转换为期望距离
        distances = torch.sum(distance_logits * self.distance_bins.view(1, 1, 1, -1), dim=-1)
        
        return distances, distance_logits
    
    def predict_angles(self, pair_representation):
        """预测主链二面角"""
        batch_size, seq_len, _, pair_dim = pair_representation.shape
        
        angle_logits = self.angle_head(pair_representation)
        angle_logits = angle_logits.view(batch_size, seq_len, seq_len, 3, 36)
        
        # 应用softmax
        angle_probs = F.softmax(angle_logits, dim=-1)
        
        # 转换为期望角度
        angles = torch.sum(angle_probs * self.angle_bins.view(1, 1, 1, 1, -1), dim=-1)
        
        return angles, angle_probs
    
    def generate_coordinates(self, pair_representation, distances):
        """生成3D坐标"""
        batch_size, seq_len, _, pair_dim = pair_representation.shape
        
        # 简化的坐标生成（实际AlphaFold使用更复杂的几何约束）
        coordinates = torch.zeros(batch_size, seq_len, 3)
        
        # 第一个残基放在原点
        coordinates[:, 0, :] = 0
        
        # 第二个残基沿x轴
        if seq_len > 1:
            coordinates[:, 1, 0] = distances[:, 0, 1]
        
        # 后续残基使用距离几何
        for i in range(2, seq_len):
            # 简化的三角化方法
            d_0i = distances[:, 0, i]
            d_1i = distances[:, 1, i]
            d_01 = distances[:, 0, 1]
            
            # 计算第i个残基的坐标（在xy平面内）
            cos_angle = (d_01**2 + d_0i**2 - d_1i**2) / (2 * d_01 * d_0i + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1, 1)
            
            x = d_0i * cos_angle
            y = d_0i * torch.sqrt(1 - cos_angle**2 + 1e-8)
            
            coordinates[:, i, 0] = x
            coordinates[:, i, 1] = y
            coordinates[:, i, 2] = torch.randn_like(x) * 0.1  # 添加小的z方向扰动
        
        return coordinates
    
    def forward(self, pair_representation):
        # 预测距离
        distances, distance_logits = self.predict_distances(pair_representation)
        
        # 预测角度
        angles, angle_probs = self.predict_angles(pair_representation)
        
        # 生成坐标
        coordinates = self.generate_coordinates(pair_representation, distances)
        
        return {
            'coordinates': coordinates,
            'distances': distances,
            'distance_logits': distance_logits,
            'angles': angles,
            'angle_probs': angle_probs
        }

# 初始化结构预测模块
structure_module = StructurePredictionModule(pair_dim=64)

# 结构预测
with torch.no_grad():
    predictions = structure_module(pair_repr)

print("\n结构预测完成:")
print(f"预测坐标形状: {predictions['coordinates'].shape}")
print(f"预测距离形状: {predictions['distances'].shape}")
print(f"预测角度形状: {predictions['angles'].shape}")

# ====================================
# 2.5 第四步：AlphaFold数据库交互
# ====================================

class AlphaFoldDBClient:
    """AlphaFold数据库客户端"""
    
    def __init__(self):
        self.base_url = "https://alphafold.ebi.ac.uk/api"
        self.files_url = "https://alphafold.ebi.ac.uk/files"
    
    def search_by_uniprot_id(self, uniprot_id):
        """通过UniProt ID搜索AlphaFold结构"""
        url = f"{self.base_url}/prediction/{uniprot_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"搜索失败: {e}")
            return None
    
    def download_structure(self, uniprot_id, format='pdb'):
        """下载AlphaFold预测结构"""
        if format == 'pdb':
            url = f"{self.files_url}/AF-{uniprot_id}-F1-model_v4.pdb"
        elif format == 'cif':
            url = f"{self.files_url}/AF-{uniprot_id}-F1-model_v4.cif"
        else:
            raise ValueError("format必须是'pdb'或'cif'")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = f"AF-{uniprot_id}-F1-model_v4.{format}"
            with open(filename, 'w') as f:
                f.write(response.text)
            
            print(f"成功下载: {filename}")
            return filename
        except requests.RequestException as e:
            print(f"下载失败: {e}")
            return None
    
    def get_confidence_scores(self, uniprot_id):
        """获取置信度评分"""
        prediction_data = self.search_by_uniprot_id(uniprot_id)
        if prediction_data:
            # The API returns a list of predictions
            if isinstance(prediction_data, list) and prediction_data:
                first_entry = prediction_data[0]
                confidence_info = {
                    'overall_confidence': first_entry.get('plddt', 'N/A'),
                    'length': len(first_entry.get('sequence', '')),
                    'uniprotAccession': first_entry.get('uniprotAccession', 'N/A')
                }
                return confidence_info
        return None

# 示例：使用AlphaFold数据库
af_client = AlphaFoldDBClient()

# 泛素的UniProt ID
ubiquitin_uniprot = "P0CG48"  # 人类泛素

print(f"\n搜索AlphaFold中的泛素结构...")
prediction_info_list = af_client.search_by_uniprot_id(ubiquitin_uniprot)

if prediction_info_list and isinstance(prediction_info_list, list):
    prediction_info = prediction_info_list[0] # Take the first entry
    print("找到AlphaFold预测结构:")
    print(f"序列长度: {len(prediction_info.get('sequence', 'N/A'))}")
    print(f"模型创建日期: {prediction_info.get('modelCreatedDate', 'N/A')}")
    print(f"UniProt ID: {prediction_info.get('uniprotAccession', 'N/A')}")
    
    # 获取置信度信息
    confidence = af_client.get_confidence_scores(ubiquitin_uniprot)
    if confidence:
        print("置信度信息:")
        for key, value in confidence.items():
            print(f"  {key}: {value}")
    
    # 下载结构文件
    pdb_file = af_client.download_structure(ubiquitin_uniprot, format='pdb')
else:
    print("未找到对应的AlphaFold结构或API返回格式不正确")


# ====================================
# 2.6 第五步：结构质量评估和可视化
# ====================================

class StructureAnalyzer:
    """结构质量分析器"""
    
    def __init__(self):
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    
    def calculate_rmsd(self, coords1, coords2):
        """计算RMSD"""
        if coords1.shape != coords2.shape:
            raise ValueError("坐标形状不匹配")
        
        # 去除batch维度进行计算
        if len(coords1.shape) == 3:
            coords1 = coords1[0]
            coords2 = coords2[0]
        
        diff = coords1 - coords2
        squared_diff = torch.sum(diff**2, dim=1)
        rmsd = torch.sqrt(torch.mean(squared_diff))
        return rmsd.item()
    
    def analyze_geometry(self, coordinates):
        """分析几何质量"""
        coords = coordinates[0] if len(coordinates.shape) == 3 else coordinates
        n_residues = coords.shape[0]
        
        # 计算键长
        bond_lengths = []
        if n_residues > 1:
            for i in range(n_residues - 1):
                bond_length = torch.norm(coords[i+1] - coords[i])
                bond_lengths.append(bond_length.item())
        
        # 计算键角
        bond_angles = []
        if n_residues > 2:
            for i in range(n_residues - 2):
                v1 = coords[i] - coords[i+1]
                v2 = coords[i+2] - coords[i+1]
                
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                angle = torch.acos(torch.clamp(cos_angle, -1, 1))
                bond_angles.append(torch.rad2deg(angle).item())
        
        return {
            'bond_lengths': bond_lengths,
            'bond_angles': bond_angles,
            'avg_bond_length': np.mean(bond_lengths) if bond_lengths else 0,
            'std_bond_length': np.std(bond_lengths) if bond_lengths else 0,
            'avg_bond_angle': np.mean(bond_angles) if bond_angles else 0,
            'std_bond_angle': np.std(bond_angles) if bond_angles else 0
        }
    
    def calculate_confidence_metrics(self, predictions):
        """计算预测置信度"""
        distances = predictions['distances'][0]
        distance_logits = predictions['distance_logits'][0]
        
        # 计算距离预测的置信度
        distance_confidence = torch.max(distance_logits, dim=-1)[0]
        avg_confidence = torch.mean(distance_confidence)
        
        # 分析距离分布
        upper_tri_indices = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
        upper_tri_distances = distances[upper_tri_indices[0], upper_tri_indices[1]]
        
        return {
            'avg_confidence': avg_confidence.item(),
            'min_confidence': torch.min(distance_confidence).item(),
            'max_confidence': torch.max(distance_confidence).item(),
            'avg_distance': torch.mean(upper_tri_distances).item(),
            'std_distance': torch.std(upper_tri_distances).item()
        }

def visualize_structure_predictions(predictions, sequence, conservation_scores, save_path='.'):
    """可视化结构预测结果，并保存所有图像和说明。"""
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # --- Helper function to save explanation text ---
    def save_explanation(filename_base, title, explanation):
        filepath_txt = os.path.join(save_path, f"{filename_base}_explanation.txt")
        with open(filepath_txt, 'w', encoding='utf-8') as f:
            f.write(f"图表标题: {title}\n\n")
            f.write(explanation)
        print(f"已保存说明: {filepath_txt}")

    # --- Main Figure (2x3 grid) ---
    main_fig, main_axes = plt.subplots(2, 3, figsize=(20, 12))
    main_fig.suptitle("Simplified AlphaFold Prediction Analysis", fontsize=16)

    # Data extraction
    distances = predictions['distances'][0].numpy()
    distance_logits = predictions['distance_logits'][0]
    confidence = torch.max(distance_logits, dim=-1)[0].numpy()
    avg_confidence_per_residue = np.mean(confidence, axis=1)
    coords = predictions['coordinates'][0].numpy()
    upper_tri_indices = np.triu_indices(distances.shape[0], k=1)
    upper_tri_distances = distances[upper_tri_indices]
    residue_positions = range(1, len(avg_confidence_per_residue) + 1)

    # --- Plotting on the main figure ---
    # Plot 1: Predicted Ca-Ca Distance Matrix
    ax1 = main_axes[0, 0]
    im1 = ax1.imshow(distances, cmap='viridis')
    ax1.set_title('Predicted Ca-Ca Distance Matrix')
    ax1.set_xlabel('Residue Index'); ax1.set_ylabel('Residue Index')
    main_fig.colorbar(im1, ax=ax1, label='Distance (Å)')

    # Plot 2: Distance Prediction Confidence
    ax2 = main_axes[0, 1]
    im2 = ax2.imshow(confidence, cmap='RdYlBu_r')
    ax2.set_title('Distance Prediction Confidence')
    ax2.set_xlabel('Residue Index'); ax2.set_ylabel('Residue Index')
    main_fig.colorbar(im2, ax=ax2, label='Confidence')

    # Plot 3: Conservation vs. Prediction Confidence
    ax3 = main_axes[0, 2]
    ax3.set_title('Conservation vs. Prediction Confidence')
    ax3.scatter(conservation_scores, avg_confidence_per_residue, alpha=0.6)
    ax3.set_xlabel('Conservation Score'); ax3.set_ylabel('Average Prediction Confidence')
    ax3.grid(True, alpha=0.3)
    if len(conservation_scores) > 1 and len(avg_confidence_per_residue) > 1:
        correlation = np.corrcoef(conservation_scores, avg_confidence_per_residue)[0,1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Plot 4: Predicted Structure (XY Projection)
    ax4 = main_axes[1, 0]
    ax4.set_title('Predicted Structure (XY Projection)')
    ax4.plot(coords[:, 0], coords[:, 1], 'o-', markersize=4, linewidth=1)
    ax4.set_xlabel('X coordinate (Å)'); ax4.set_ylabel('Y coordinate (Å)')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Predicted Distance Distribution
    ax5 = main_axes[1, 1]
    ax5.set_title('Predicted Distance Distribution')
    ax5.hist(upper_tri_distances, bins=30, alpha=0.7, density=True)
    if len(upper_tri_distances) > 0:
        mean_dist = np.mean(upper_tri_distances)
        ax5.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}Å')
    ax5.set_xlabel('Distance (Å)'); ax5.set_ylabel('Density')
    ax5.legend()

    # Plot 6: Per-Residue Prediction Confidence
    ax6 = main_axes[1, 2]
    ax6.set_title('Per-Residue Prediction Confidence')
    ax6.plot(residue_positions, avg_confidence_per_residue, 'b-', linewidth=2)
    ax6.fill_between(residue_positions, avg_confidence_per_residue, alpha=0.3)
    ax6.set_xlabel('Residue Position'); ax6.set_ylabel('Average Confidence')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(bottom=min(0, ax6.get_ylim()[0]))

    # --- Save Main Figure and Show ---
    main_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    full_fig_path = os.path.join(save_path, 'full_prediction_analysis.png')
    main_fig.savefig(full_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n已保存完整图表: {full_fig_path}")
    plt.show(block=False) # Show main figure but don't block execution

    # --- Save Individual Subplots --- 
    print("\n正在保存独立的子图...")

    # Explanation Texts
    explanation1 = ("""    这张图是什么：
    这是一张“距离图”（Distogram），一个二维热力图。图中的每个像素(i, j)的颜色代表了模型预测的第i个和第j个氨基酸残基的α-碳原子（Ca）之间的空间距离（单位：埃, Å）。

    如何解读：
    - 颜色越深（或按色标越冷），代表两个残基的距离越近。
    - 颜色越浅（或按色标越暖），代表距离越远。
    - 主对角线（左上到右下）为零，因为是残基与自身的距离。
    - 图中形成的模式揭示了蛋白质的二级结构（如α-螺旋沿对角线呈现为粗线）和三级结构（如β-折叠片形成的反向平行线）。

    在AlphaFold流程中的作用：
    这是结构预测模块的核心输出之一。Evoformer模块处理完MSA和残基对信息后，会生成一个概率分布，本图就是这个分布的期望距离。它是从序列信息到三维空间结构的关键桥梁，后续的3D坐标生成将严重依赖这张距离图所包含的几何约束。
    """)
    explanation2 = ("""    这张图是什么：
    这是一张置信度热力图，展示了模型对其预测的每个残基对距离的信心程度。它在概念上类似于AlphaFold著名的pLDDT（predicted Local Distance Difference Test）分数，但这里是针对距离预测的。

    如何解读：
    - 颜色越暖（如红色、橙色），代表模型对该距离预测的置信度越高。
    - 颜色越冷（如蓝色），代表置信度越低。
    - 低置信度区域通常对应于蛋白质中高度灵活的部分，如循环（loops）或无序区域（disordered regions），这些区域本身就没有固定的结构。

    在AlphaFold流程中的作用：
    置信度是评估模型预测质量至关重要的指标。它帮助我们判断预测结果的哪些部分是可靠的，哪些部分可能不准确。在科研和药物设计中，通常只信任高置信度的结构区域。
    """)
    explanation3 = ("""    这张图是什么：
    这是一个散点图，用于比较两个关键指标：氨基酸序列的“保守性”和模型预测的“置信度”。
    - X轴（保守性分数）：根据多序列比对（MSA）计算得出。分数越高，表示该位置的氨基酸在进化过程中越不容易发生改变，通常意味着它对蛋白质的功能或结构至关重要。
    - Y轴（平均预测置信度）：对应每个残基的平均预测置信度。

    如何解读：
    - 如果点倾向于从左下到右上分布（正相关），说明保守性越高的残基，模型预测其结构的置信度也越高。这符合生物学直觉：进化上重要的残基通常构成了蛋白质稳定折叠的核心。

    在AlphaFold流程中的作用：
    这张图连接了输入的MSA信息和输出的结构质量。它验证了AlphaFold的核心假设之一：进化相关的序列信息（MSA）包含了破解其三维结构的密码。保守性高的区域通常是结构预测最准确的区域。
    """)
    explanation4 = ("""    这张图是什么：
    这是模型最终生成的蛋白质三维坐标在XY平面上的一个二维投影。它将预测出的每个残基的α-碳原子（Ca）的三维坐标(x, y, z)中的x和y值绘制出来，并用线连接。

    如何解读：
    - 这张图提供了一个蛋白质骨架折叠方式的直观、快速的预览。你可以大致看出蛋白质是紧凑的球状蛋白，还是一个伸展的结构。
    - 它不是一个完整的3D视图，但可以帮助快速识别出α-螺旋和β-折叠等二级结构元件的大致走向。

    在AlphaFold流程中的作用：
    这是结构预测模块的最终输出。在得到距离和角度等几何约束后，模型通过一个复杂的三角化或梯度下降过程来生成这些三维坐标，以最好地满足所有预测的约束。这张图是最终成果的直接可视化。
    """)
    explanation5 = ("""    这张图是什么：
    这是一个直方图，显示了蛋白质中所有非相邻残基对之间预测距离的分布情况。

    如何解读：
    - X轴是距离（Å），Y轴是该距离出现的频率（密度）。
    - 分布的形状可以告诉我们蛋白质的紧凑程度。对于一个紧凑的球状蛋白，分布会集中在较小的距离值；而对于一个细长的或多结构域的蛋白质，分布可能会更宽或出现多个峰。
    - 红色的虚线标出了所有距离的平均值。

    在AlphaFold流程中的作用：
    这张图提供了对蛋白质整体尺寸和紧凑性的宏观统计。它可以帮助我们快速判断预测出的结构是否合理（例如，一个小的单结构域蛋白不应有非常大的平均距离）。
    """)
    explanation6 = ("""    这张图是什么：
    这是一张线图，展示了沿着蛋白质氨基酸序列（从N端到C端）每个位置的平均预测置信度。

    如何解读：
    - X轴是残基在序列中的位置。
    - Y轴是该残基与其他所有残基之间距离预测的平均置信度。
    - 高峰区域代表结构预测可靠的稳定区域（如α-螺旋和β-折叠的核心）。
    - 低谷区域代表低置信度区域，通常是柔性的循环（loops）或蛋白质的N/C末端，这些区域在生物体内本身可能就是动态变化的。

    在AlphaFold流程中的作用：
    这张图对于定位结构中的“薄弱环节”非常有用。当分析一个预测结构时，研究人员会首先查看这张图，以确定哪些区域的坐标是可信的，哪些区域需要谨慎对待或通过实验进一步验证。
    """)

    # Save Plot 1
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(distances, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Distance (Å)')
    ax.set_title('Predicted Ca-Ca Distance Matrix'); ax.set_xlabel('Residue Index'); ax.set_ylabel('Residue Index')
    fig.savefig(os.path.join(save_path, '1_predicted_distance_matrix.png'), dpi=300, bbox_inches='tight')
    save_explanation('1_predicted_distance_matrix', 'Predicted Ca-Ca Distance Matrix', explanation1)
    plt.close(fig)

    # Save Plot 2
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confidence, cmap='RdYlBu_r')
    fig.colorbar(im, ax=ax, label='Confidence')
    ax.set_title('Distance Prediction Confidence'); ax.set_xlabel('Residue Index'); ax.set_ylabel('Residue Index')
    fig.savefig(os.path.join(save_path, '2_distance_prediction_confidence.png'), dpi=300, bbox_inches='tight')
    save_explanation('2_distance_prediction_confidence', 'Distance Prediction Confidence', explanation2)
    plt.close(fig)

    # Save Plot 3
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(conservation_scores, avg_confidence_per_residue, alpha=0.6)
    ax.set_title('Conservation vs. Prediction Confidence'); ax.set_xlabel('Conservation Score'); ax.set_ylabel('Average Prediction Confidence')
    ax.grid(True, alpha=0.3)
    if len(conservation_scores) > 1 and len(avg_confidence_per_residue) > 1:
        correlation = np.corrcoef(conservation_scores, avg_confidence_per_residue)[0,1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    fig.savefig(os.path.join(save_path, '3_conservation_vs_confidence.png'), dpi=300, bbox_inches='tight')
    save_explanation('3_conservation_vs_confidence', 'Conservation vs. Prediction Confidence', explanation3)
    plt.close(fig)

    # Save Plot 4
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coords[:, 0], coords[:, 1], 'o-', markersize=4, linewidth=1)
    ax.set_title('Predicted Structure (XY Projection)'); ax.set_xlabel('X coordinate (Å)'); ax.set_ylabel('Y coordinate (Å)')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
    fig.savefig(os.path.join(save_path, '4_predicted_structure_projection.png'), dpi=300, bbox_inches='tight')
    save_explanation('4_predicted_structure_projection', 'Predicted Structure (XY Projection)', explanation4)
    plt.close(fig)

    # Save Plot 5
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(upper_tri_distances, bins=30, alpha=0.7, density=True)
    if len(upper_tri_distances) > 0:
        mean_dist = np.mean(upper_tri_distances)
        ax.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}Å')
    ax.set_title('Predicted Distance Distribution'); ax.set_xlabel('Distance (Å)'); ax.set_ylabel('Density')
    ax.legend()
    fig.savefig(os.path.join(save_path, '5_predicted_distance_distribution.png'), dpi=300, bbox_inches='tight')
    save_explanation('5_predicted_distance_distribution', 'Predicted Distance Distribution', explanation5)
    plt.close(fig)

    # Save Plot 6
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(residue_positions, avg_confidence_per_residue, 'b-', linewidth=2)
    ax.fill_between(residue_positions, avg_confidence_per_residue, alpha=0.3)
    ax.set_title('Per-Residue Prediction Confidence'); ax.set_xlabel('Residue Position'); ax.set_ylabel('Average Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.savefig(os.path.join(save_path, '6_per_residue_confidence.png'), dpi=300, bbox_inches='tight')
    save_explanation('6_per_residue_confidence', 'Per-Residue Prediction Confidence', explanation6)
    plt.close(fig)

    print("\n所有子图保存完毕。")


# 执行结构分析
analyzer = StructureAnalyzer()

# 几何质量分析
geometry_analysis = analyzer.analyze_geometry(predictions['coordinates'])
print("\n几何质量分析:")
print(f"平均键长: {geometry_analysis['avg_bond_length']:.2f} ± {geometry_analysis['std_bond_length']:.2f} Å")
print(f"平均键角: {geometry_analysis['avg_bond_angle']:.1f} ± {geometry_analysis['std_bond_angle']:.1f}°")

# 置信度分析
confidence_metrics = analyzer.calculate_confidence_metrics(predictions)
print("\n预测置信度分析:")
for metric, value in confidence_metrics.items():
    print(f"{metric}: {value:.3f}")

# 可视化结果
visualize_structure_predictions(predictions, ubiquitin_sequence, conservation_scores)


# ====================================
# 2.7 第六步：完整的AlphaFold预测流程
# ====================================

class AlphaFoldPredictor:
    """完整的AlphaFold预测流程"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.msa_generator = MSAGenerator()
        self.evoformer = SimplifiedEvoformer().to(device)
        self.structure_module = StructurePredictionModule().to(device)
        self.analyzer = StructureAnalyzer()
        
    def predict_structure(self, sequence, max_msa_sequences=100):
        """端到端结构预测"""
        
        print(f"开始预测序列结构: {len(sequence)}残基")
        
        # Step 1: 生成MSA
        print("Step 1: 生成多序列比对...")
        homologs = self.msa_generator.fetch_homologous_sequences(
            sequence, max_sequences=max_msa_sequences)
        msa_matrix, aligned_sequences = self.msa_generator.create_msa_matrix(homologs)
        conservation_scores = self.msa_generator.calculate_conservation_scores(msa_matrix)
        
        # Step 2: Evoformer处理
        print("Step 2: Evoformer特征提取...")
        msa_tensor = torch.from_numpy(msa_matrix).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            msa_repr, pair_repr = self.evoformer(msa_tensor)
        
        # Step 3: 结构预测
        print("Step 3: 结构坐标生成...")
        with torch.no_grad():
            predictions = self.structure_module(pair_repr)
        
        # Step 4: 质量评估
        print("Step 4: 结构质量评估...")
        geometry_analysis = self.analyzer.analyze_geometry(predictions['coordinates'])
        confidence_metrics = self.analyzer.calculate_confidence_metrics(predictions)
        
        # 组装结果
        result = {
            'sequence': sequence,
            'msa_info': {
                'n_sequences': len(homologs),
                'conservation_scores': conservation_scores,
                'avg_identity': np.mean([h['identity'] for h in homologs])
            },
            'predictions': predictions,
            'geometry_analysis': geometry_analysis,
            'confidence_metrics': confidence_metrics,
            'success': True
        }
        
        print("结构预测完成!")
        return result
    
    def compare_with_alphafold(self, sequence, uniprot_id=None):
        """与AlphaFold数据库中的结构比较"""
        
        # 获取我们的预测
        our_prediction = self.predict_structure(sequence)
        
        if uniprot_id:
            # 尝试获取AlphaFold结构
            af_client = AlphaFoldDBClient()
            af_info_list = af_client.search_by_uniprot_id(uniprot_id)
            
            if af_info_list and isinstance(af_info_list, list):
                af_info = af_info_list[0]
                print(f"\n与AlphaFold数据库比较:")
                print(f"AlphaFold序列长度: {len(af_info.get('sequence', 'N/A'))}")
                print(f"我们预测序列长度: {len(sequence)}")
                
                # 下载AlphaFold结构进行比较
                af_pdb_file = af_client.download_structure(uniprot_id)
                if af_pdb_file:
                    print(f"已下载AlphaFold结构: {af_pdb_file}")
        
        return our_prediction
    
    def batch_prediction(self, sequences):
        """批量预测多个序列"""
        results = []
        
        for i, seq in enumerate(sequences):
            print(f"\n处理序列{i+1}/{len(sequences)}...")
            try:
                result = self.predict_structure(seq)
                results.append(result)
            except Exception as e:
                print(f"序列{i+1}预测失败: {e}")
                results.append({
                    'sequence': seq,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# 创建完整的预测器
predictor = AlphaFoldPredictor()

# 执行完整预测流程
print("\n=== 完整AlphaFold预测流程演示 ===")
result = predictor.compare_with_alphafold(ubiquitin_sequence, uniprot_id="P0CG48")

# 打印预测总结
if result and result['success']:
    print(f"\n=== 预测总结 ===")
    print(f"序列长度: {len(result['sequence'])}残基")
    print(f"MSA序列数: {result['msa_info']['n_sequences']}")
    print(f"平均序列一致性: {result['msa_info']['avg_identity']:.1f}%")
    print(f"预测置信度: {result['confidence_metrics']['avg_confidence']:.3f}")
    print(f"平均Ca-Ca距离: {result['confidence_metrics']['avg_distance']:.1f}Å")

    # 高质量可视化
    visualize_structure_predictions(result['predictions'], result['sequence'], 
                                  result['msa_info']['conservation_scores'])

# ====================================
# 3.1 案例1：新冠病毒蛋白结构预测
# ====================================

def covid_protein_analysis():
    """新冠病毒蛋白结构预测案例"""
    
    # SARS-CoV-2 Spike蛋白受体结合域(RBD)的部分序列
    spike_rbd_sequence = ("NLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGD"
                         "EVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGS"
                         "TPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF")
    
    print("\n=== 新冠病毒Spike RBD结构预测 ===")
    print(f"序列长度: {len(spike_rbd_sequence)}残基")
    
    # 执行预测
    covid_result = predictor.predict_structure(spike_rbd_sequence, max_msa_sequences=30)
    
    # 分析结果
    if covid_result and covid_result['success']:
        print(f"MSA质量: {covid_result['msa_info']['n_sequences']}序列")
        print(f"预测置信度: {covid_result['confidence_metrics']['avg_confidence']:.3f}")
        
        # 识别潜在的结合位点（高保守性区域）
        conservation = covid_result['msa_info']['conservation_scores']
        high_conservation_sites = np.where(conservation > 0.8)[0]
        
        print(f"高保守性位点 (保守性>0.8): {len(high_conservation_sites)}个")
        if len(high_conservation_sites) > 0:
            print(f"位点编号: {high_conservation_sites[:10]}...")  # 显示前10个
    
    return covid_result

# 运行COVID案例（如果时间允许）
# print("\n运行COVID案例分析...")
# covid_result = covid_protein_analysis()

print("\n脚本执行完毕。")

