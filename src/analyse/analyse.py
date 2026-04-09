"""
Probability Distribution Analysis and Visualization Tools

This module provides functionality for analyzing probability distribution correlations,
calculating RIDIT scores, and plotting histograms.
"""

# Standard library imports
from typing import Iterable, Dict, List, Optional, Tuple

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report

# =============================================================================
# Correlation Analysis Module
# =============================================================================

def analyze_correlation(
    probability_list: List[List[float]], 
    labels: Optional[List[str]] = None, 
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze correlations between probability distributions
    
    Calculate and visualize various correlation metrics including Pearson correlation coefficient,
    cosine similarity, Jensen-Shannon distance, and Spearman correlation coefficient.
    
    Parameters
    ----------
    probability_list : List[List[float]]
        List of probability distributions, each element is a probability distribution
    labels : Optional[List[str]], default=None
        List of labels for chart display
    save_path : Optional[str], default=None
        Path to save the chart
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns four correlation matrices: (Pearson correlation, cosine similarity, JS distance, Spearman correlation)
    """
    n = len(probability_list)
    if labels is None:
        labels = [f"U{i+1}" for i in range(n)]
    
    arr = np.array(probability_list)
    
    # Calculate Pearson correlation coefficient
    pearson_corr = np.corrcoef(arr)
    print("Pearson Correlation:")
    print(pearson_corr, "\n")
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(arr)
    print("Cosine Similarity:")
    print(cos_sim, "\n")
    
    # Calculate Jensen-Shannon distance
    js_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                js_dist[i, j] = 0
            else:
                js_dist[i, j] = jensenshannon(arr[i], arr[j])
    print("Jensen–Shannon Distance:")
    print(js_dist, "\n")
    
    # Calculate Spearman correlation coefficient
    spearman_corr, _ = spearmanr(arr, axis=1)
    print("Spearman Correlation:")
    print(spearman_corr, "\n")
    
    # Calculate the MSE between the probability distributions
    mse = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mse[i, j] = np.mean((np.array(probability_list[i]) - np.array(probability_list[j]))**2)
            mse[j, i] = np.mean((np.array(probability_list[j]) - np.array(probability_list[i]))**2)

    print("MSE:")
    print(mse, "\n")

    # Create visualization charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    labels = [ f'Anno {i+1}' for i in range(n)]
    fontsize = 18
    annot_fontsize = 16
    tick_fontsize = 14
    # Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, ax=axes[0,0],
                annot_kws={"size": annot_fontsize})
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha="right")
    axes[0,0].tick_params(labelsize=tick_fontsize)
    axes[0,0].set_title("Pearson Correlation", fontsize=fontsize)
    
    # Cosine similarity heatmap
    # sns.heatmap(cos_sim, annot=True, cmap="YlGnBu", 
    #             xticklabels=labels, yticklabels=labels, ax=axes[0,1])
    # axes[0,1].set_title("Cosine Similarity")
    

    sns.heatmap(mse, annot=True, cmap="YlGnBu", 
                xticklabels=labels, yticklabels=labels, ax=axes[1,0],
                annot_kws={"size": annot_fontsize})
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha="right")
    axes[1,0].tick_params(labelsize=tick_fontsize)
    axes[1,0].set_title("MSE", fontsize=fontsize)
    # Jensen-Shannon distance heatmap
    sns.heatmap(js_dist, annot=True, cmap="Reds", 
                xticklabels=labels, yticklabels=labels, ax=axes[0,1],
                annot_kws={"size": annot_fontsize})
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha="right")
    axes[0,1].tick_params(labelsize=tick_fontsize)
    axes[0,1].set_title("Jensen-Shannon Distance", fontsize=fontsize)
    
    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, ax=axes[1,1],
                annot_kws={"size": annot_fontsize})
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha="right")
    axes[1,1].tick_params(labelsize=tick_fontsize)
    axes[1,1].set_title("Spearman Correlation", fontsize=fontsize)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return pearson_corr, cos_sim, js_dist, spearman_corr


# =============================================================================
# RIDIT Analysis Module
# =============================================================================

import numpy as np
from typing import Iterable, Dict

def ridit(x: Iterable) -> Dict[int, float]:
    x_array = np.asarray(x, dtype=int).flatten()
    # 1. 获取排序后的唯一值，确保映射关系正确
    unique_vals = np.unique(x_array)
    
    # 2. 计算每个类别的频数
    # 使用 min_length 配合 x - min 以确保 bincount 覆盖所有范围
    counts = np.bincount(x_array - x_array.min())
    # 过滤掉中间可能存在的 0 频数（如果原始数据不连续）
    props = counts[counts > 0] / len(x_array)
    
    # 3. 计算 RIDIT 公式: R = Cumulative_Previous + (Current_Prop / 2)
    # np.cumsum 得到的累积分布：[p1, p1+p2, p1+p2+p3...]
    cumdist = np.cumsum(props)
    
    # 在开头插入 0，并去掉最后一项，得到“前一类别的累积分布”
    # [0, p1, p1+p2, ...]
    prev_cumdist = np.insert(cumdist[:-1], 0, 0.0)
    
    ridit_scores = prev_cumdist + (props / 2.0)
    
    return dict(zip(unique_vals, ridit_scores))


def calculate_krippendorff_alpha(ridit_data: List[List[float]]) -> float:
    """
    Calculate Krippendorff's Alpha coefficient
    
    Statistical measure for assessing inter-rater reliability.
    
    Parameters
    ----------
    ridit_data : List[List[float]]
        RIDIT scoring data
        
    Returns
    -------
    float
        Krippendorff's Alpha coefficient
    """
    return krippendorff.alpha(reliability_data=ridit_data, level_of_measurement='interval')


def calculate_binned_data(
    probability_list: List[List[float]], 
    bin_num: int = 5
) -> List[List[float]]:
    """
    Bin probability data and calculate RIDIT scores
    
    Parameters
    ----------
    probability_list : List[List[float]]
        List of probability distributions
    bin_num : int, default=5
        Number of bins
        
    Returns
    -------
    List[List[float]]
        RIDIT scoring data
    """
    probability_list = np.array(probability_list)
    binned_data = np.floor(probability_list * bin_num).astype(int)
    binned_data[binned_data == bin_num] = bin_num - 1

    ridit_map = ridit(binned_data.flatten().tolist())
    ridit_data = [[] for _ in range(len(probability_list))]
    
    for idx, i in enumerate(binned_data):
        for _, j in enumerate(i):
            ridit_data[idx].append(ridit_map[j])

    return ridit_data


# =============================================================================
# Visualization Module
# =============================================================================

def plot_histograms(
    arrays: List[List[float]], 
    titles: Optional[List[str]] = None, 
    bins: int = 30, 
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot histograms for multiple arrays
    
    Create a subplot for each array showing its probability distribution.
    
    Parameters
    ----------
    arrays : List[List[float]]
        List of data arrays to plot
    titles : Optional[List[str]], default=None
        List of titles for each subplot
    bins : int, default=30
        Number of histogram bins
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    save_path : Optional[str], default=None
        Path to save the chart
    """
    n = len(arrays)
    if titles is None:
        titles = [f'Array {i+1}' for i in range(n)]
    
    fig, axes = plt.subplots(2,n//2, figsize=figsize)
    axes = axes.flatten()
    annotate_num = 1
    annotate_dict = {
        'qwen_3_32b_thinking': 'Qwen3-VL-32B Thinking',
        'qwen_3_32b': 'Qwen3-VL-32B',
        'qwen3_omni_30b_thinking': 'Qwen3-Omni-30B Thinking',
        'qwen3_omni_30b': 'Qwen3-Omni-30B',
    }
    fontsize = 27
    tick_fontsize = 20
    para_fontsize = 18
    for ax, data, title in zip(axes, arrays, titles):
        if title in ['dzhang98', 'amart233', 'wjurayj1', 'rkriz']:
            title = f'Annotator {annotate_num}'
            annotate_num += 1

        if title in annotate_dict:
            title = annotate_dict[title]
        ax.hist(data, bins=50, edgecolor='black')
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(labelsize=para_fontsize)
        ax.set_xlabel('Value', fontsize=tick_fontsize)
        ax.set_ylabel('Frequency', fontsize=tick_fontsize)
        ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for ax in axes[n:]:
        ax.set_visible(False)
    
    # fig.suptitle('Probability Distributions', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)

def plot_distribution(
    array: List[float],
    titles: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot a distribution
    """
    plt.figure(figsize=(10, 8))


    for d, t in zip(array, titles):
        sns.kdeplot(d, label=t, fill=False, alpha=0.5)
    plt.title('distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path)


def calculate_score(probability_list: List[List[float]],labels: List[str],titles: List[str]) -> List[float]:
    """
    Calculate the accuracy score for each probability distribution
    """
    result = {}
    for p,t in zip(probability_list,titles):
        result[t] = []
        for i in p:
            if i > 0.5:
                result[t].append(1)
            else:
                result[t].append(0)
                
    for title in result:
        print(title)
        print('accuracy_score: ',accuracy_score(labels, result[title]))
        print('f1_score: ',f1_score(labels, result[title]))
        print('precision_score: ',precision_score(labels, result[title]))
        print('recall_score: ',recall_score(labels, result[title],average='micro'))
        # print('classification_report: ',classification_report(labels, result[title]))
        print('\n')
    return result,labels


