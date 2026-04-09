"""
Main Analysis Script

This module analyzes human annotation data and LLM model output probability distributions,
calculating correlation, accuracy, and other metrics.
"""

import json
import os
import random
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

import numpy as np
from src.analyse.analyse import (
    calculate_score,
    analyze_correlation,
    plot_histograms,
    calculate_binned_data,
    calculate_krippendorff_alpha
)
from sklearn.metrics import accuracy_score
# Set random seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Configuration Class
# =============================================================================

@dataclass
class AnalysisConfig:
    """Analysis configuration class"""
    human_data_path: str = './human_data/human_data_full.json'
    result_folder: str = './training_data/human'
    method: str = 'mean'
    model_list: List[str] = field(default_factory=lambda: [
        # 'qwenvl_3b', 'qwenvl_7b', 'qwenvl_32b',
        # 'qwenomni_3b', 'qwenomni_7b',
        # 'qwen3vl_4b', 'qwen3vl_8b', 'qwen3vl_32b',
        # 'qwenvl_32b_NLI', 'qwenvl_32b_NT'
        'qwen_3_32b_thinking','qwen_3_32b',
        'qwen3_omni_30b_thinking','qwen3_omni_30b',
        # 'trained',
    ])
    selected_model_list: List[str] = field(default_factory=lambda: [
        # 'qwen_3_32b_thinking','qwen_3_32b',
        # 'qwen3_omni_30b_thinking','qwen3_omni_30b',
        # 'trained',
        # 'merged_result',
    ])
    selected_human_list: List[str] = field(default_factory=lambda: [
        'dzhang98', 'amart233', 'wjurayj1', 'rkriz'
    ])
    correlation_save_path: str = './training_data/human_probability_correlation.png'
    histogram_save_path: str = './training_data/human_probability.png'


# =============================================================================
# Probability Extraction Utility Functions
# =============================================================================

def extract_probability_from_answer(answer: str) -> float:
    """
    Extract probability value from answer containing <answer> tags
    
    Parameters
    ----------
    answer : str
        Answer string containing <answer> tags
        
    Returns
    -------
    float
        Extracted probability value, returns 0 if extraction fails
    """
    if '<answer>' in answer:
        answer = answer.split('<answer>')[1].split('</answer>')[0]
        if '%' in answer:
            answer = answer.split('%')[0].strip()
            return float(answer)
    return 0.0


def extract_probability(answer: Union[str, float]) -> float:
    """
    Extract probability value from answer
    
    Supports multiple formats:
    - Float: returns directly
    - String containing <answer> tags
    - String containing %
    - String containing range (e.g., "0.5-0.7"), returns average value
    
    Parameters
    ----------
    answer : Union[str, float]
        Answer, can be string or float
        
    Returns
    -------
    float
        Extracted probability value, returns 0.0 if extraction fails
    """
    if isinstance(answer, float):
        return answer
    
    if not isinstance(answer, str):
        return 0.0
    
    try:
        # Handle case with <answer> tags
        if '<answer>' in answer:
            answer = answer.split('<answer>')[1].split('</answer>')[0]
        
        # Handle case with %
        if '%' in answer:
            answer = answer.split('%')[0].strip()
        
        # Handle range values (e.g., "0.5-0.7"), return average
        if '-' in answer:
            parts = answer.split('-')
            if len(parts) == 2:
                answer1 = parts[0].strip()
                answer2 = parts[1].strip()
                return (float(answer1) + float(answer2)) / 2
        
        return float(answer)
    except (ValueError, AttributeError):
        return 0.0

def extract_probability_from_answer_list(
    answer_list: List[Union[str, float, List[float]]],
    answer_type: str = 'probability',
    return_type: str = 'mean'
) -> float:
    """
    Extract probability value from answer list
    
    Parameters
    ----------
    answer_list : List[Union[str, float, List[float]]]
        List of answers
    answer_type : str, default='score'
        Answer type: 'score' or 'probability'
    return_type : str, default='max'
        Return type: 'max' or 'mean'
        
    Returns
    -------
    float
        Extracted probability value
        
    Raises
    ------
    ValueError
        When answer_type or return_type is invalid
    """
    if answer_type == 'score':
        # Find position of maximum value
        max_number = max(answer_list)
        max_number_position = answer_list.index(max_number)
        bins = 10
        step_size = 1.0 / bins
        expectation_values = np.array([i * step_size + 0.5 * step_size for i in range(bins)])
        sum_ = (expectation_values * np.array(answer_list)).sum()
        
        if return_type == 'max':
            return max_number_position / 9.0
        elif return_type == 'mean':
            return sum_
        else:
            raise ValueError(f'Invalid return type: {return_type}')
        
    elif answer_type == 'probability':
        probability_list = [extract_probability(answer) for answer in answer_list]

        if return_type == 'max':
            a = []
            b = [] 
            for p in probability_list:
                if p < 0.5:
                    a.append(p)
                else:
                    b.append(p)
            if len(a) > len(b):
                return float(np.mean(a))
            else:
                return float(np.mean(b))
        elif return_type == 'mean':
            return float(np.mean(probability_list))
        else:
            print('random choice')
            return float(np.max(probability_list))
    else:
        raise ValueError(f'Invalid answer type: {answer_type}')
    return 0.0

# =============================================================================
# Data Filtering Utility Functions
# =============================================================================

def filter_data(
    human_data: Dict[str, List[Dict]],
    llm_data: Union[Dict, List[Dict]],
    reference_key: str = 'dzhang98'
) -> List[Dict]:
    """
    Filter LLM data based on human annotation data
    
    Only keeps LLM data items that match videos and claims in human annotation data.
    
    Parameters
    ----------
    human_data : Dict[str, List[Dict]]
        Human annotation data, keys are annotator names, values are data item lists
    llm_data : Union[Dict, List[Dict]]
        LLM data, can be dict or list
    reference_key : str, default='dzhang98'
        Reference annotator key for matching
        
    Returns
    -------
    List[Dict]
        Filtered LLM data list
    """
    # Unfold llm_data (if dict, unfold all values)
    if isinstance(llm_data, dict):
        unfold_llm_data = [item for sublist in llm_data.values() for item in sublist]
    else:
        unfold_llm_data = llm_data
    
    filtered_llm_data = []
    reference_data = human_data.get(reference_key, [])
    
    for item in reference_data:
        if 'video' not in item.keys():
            video = item['path']
        else:
            video = item['video']
        claim = item['claim']
        
        for llm_item in unfold_llm_data:
            if 'video' not in llm_item.keys():
                llm_video = llm_item['path'].split('/')[-1]
            else:
                llm_video = llm_item['video'].split('/')[-1]
            llm_claim = llm_item['claim']
            
            if video == llm_video and claim == llm_claim:
                filtered_llm_data.append(llm_item)
                break
    
    return filtered_llm_data


def filter_probability(
    probability: List[List[float]],
    method: str = 'keep'
) -> List[List[float]]:
    """
    Filter probability data according to specified method
    
    Parameters
    ----------
    probability : List[List[float]]
        Probability data list, each element is a probability distribution
    method : str, default='keep'
        Filtering method:
        - 'keep': keep all data
        - '01': only keep positions with value 0 or 1
        - '_0_1': do not keep positions with value 0 or 1
        
    Returns
    -------
    List[List[float]]
        Filtered probability data
        
    Raises
    ------
    ValueError
        When method is invalid
    """
    if method == 'keep':
        return probability
    
    elif method == '01':  # Only keep positions with value 0 or 1
        idx = set()
        for p in probability:
            for index, p_i in enumerate(p):
                if abs(p_i - 0.0) < 1e-10 or abs(p_i - 1.0) < 1e-10:
                    idx.add(index)

        new_probability = []
        for p in probability:
            temp = [p[i] for i in range(len(p)) if i in idx]
            new_probability.append(temp)
        return new_probability

    elif method == '_0_1':  # Do not keep positions with value 0 or 1
        idx = set()
        for p in probability:
            for index, p_i in enumerate(p):
                if abs(p_i - 0.0) >= 1e-10 and abs(p_i - 1.0) >= 1e-10:
                    idx.add(index)
        
        new_probability = []
        for p in probability:
            temp = [p[i] for i in range(len(p)) if i in idx]
            new_probability.append(temp)

        return new_probability
    else:
        raise ValueError(f'Invalid method: {method}')



# =============================================================================
# Data Loading and Processing Functions
# =============================================================================

def load_human_data(file_path: str) -> Dict[str, List[Dict]]:
    """
    Load human annotation data
    
    Parameters
    ----------
    file_path : str
        Human annotation data file path
        
    Returns
    -------
    Dict[str, List[Dict]]
        Human annotation data dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_human_probabilities(
    human_data: Dict[str, List[Dict]]
) -> Dict[str, List[float]]:
    """
    Extract probability values from human annotation data
    
    Parameters
    ----------
    human_data : Dict[str, List[Dict]]
        Human annotation data
        
    Returns
    -------
    Dict[str, List[float]]
        Probability list for each human annotator
    """
    human_probability = {}
    for people in human_data:
        human_probability[people] = [
            item['p'] / 1 for item in human_data[people]
        ]
    return human_probability


def extract_labels(
    human_data: Dict[str, List[Dict]],
    reference_model_path: str
) -> List[str]:
    """
    Extract labels from reference model
    
    Parameters
    ----------
    human_data : Dict[str, List[Dict]]
        Human annotation data
    reference_model_path : str
        Reference model data file path
        
    Returns
    -------
    List[str]
        Label list
    """
    with open(reference_model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_llm_data = filter_data(human_data, data)
    return [item['label'] for item in filtered_llm_data]


def extract_llm_probabilities(
    human_data: Dict[str, List[Dict]],
    model_list: List[str],
    result_folder: str
) -> Dict[str, List[float]]:
    """
    Extract probability values from LLM data
    
    Parameters
    ----------
    human_data : Dict[str, List[Dict]]
        Human annotation data, used to filter LLM data
    model_list : List[str]
        Model name list
    result_folder : str
        Result folder path
        
    Returns
    -------
    Dict[str, List[float]]
        Probability list for each model
    """
    llm_probability = {}
    
    for model in model_list:
        llm_path = os.path.join(result_folder, f'{model}.json')
        
        if not os.path.exists(llm_path):
            print(f'Warning: {llm_path} not found, skipping...')
            continue
        
        with open(llm_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        filtered_llm_data = filter_data(human_data, llm_data)
        llm_probability[model] = []
        
        for data in filtered_llm_data:
            if isinstance(data['answer'], list):
                probability = extract_probability_from_answer_list(data['answer'], return_type='random')
            else:
                probability = extract_probability(data['answer'])
                print(probability)
            # If probability > 1, might be percentage format, divide by 100
            if probability > 1:
                probability /= 100.0
            
            llm_probability[model].append(probability)

    return llm_probability


# =============================================================================
# Analysis Functions
# =============================================================================

def calculate_krippendorff_alpha_for_models(
    human_probability: Dict[str, List[float]],
    llm_probability: Dict[str, List[float]],
    selected_human_list: List[str],
    selected_model_list: List[str],
    bin_num: int = 10
) -> None:
    """
    Calculate Krippendorff's Alpha coefficient for each selected model
    
    Parameters
    ----------
    human_probability : Dict[str, List[float]]
        Human annotation probability dictionary
    llm_probability : Dict[str, List[float]]
        LLM model probability dictionary
    selected_human_list : List[str]
        Selected human annotator list
    selected_model_list : List[str]
        Selected model list
    bin_num : int, default=10
        Number of bins
    """
    result = {}
    for model in selected_model_list:
        if model not in llm_probability:
            print(f'Warning: {model} not found in llm_probability, skipping...')
            continue
        
        temp = (
            [human_probability[people] for people in selected_human_list] +
            [llm_probability[model]]
        )
        temp = filter_probability(temp)
        binned_data = calculate_binned_data(temp, bin_num=bin_num)
        alpha = calculate_krippendorff_alpha(binned_data)
        human_prob = np.mean([human_probability[people] for people in selected_human_list])
        llm_prob = np.mean([llm_probability[model]])
        mse = np.mean((human_prob - llm_prob) ** 2)
        print(f'{model} krippendorff_alpha: {alpha}, mse: {mse}')
        result[model] = alpha
    return result

def perform_analysis(
    human_probability: Dict[str, List[float]],
    llm_probability: Dict[str, List[float]],
    labels: List[str],
    config: AnalysisConfig
) -> None:
    """
    Perform complete analysis pipeline
    
    Parameters
    ----------
    human_probability : Dict[str, List[float]]
        Human annotation probability dictionary
    llm_probability : Dict[str, List[float]]
        LLM model probability dictionary
    labels : List[str]
        Label list
    config : AnalysisConfig
        Analysis configuration
    """
    # Combine selected probability data
    combined_probability = (
        [human_probability[people] for people in config.selected_human_list] +
        [llm_probability[model] for model in config.selected_model_list]
    )
    combined_probability = filter_probability(combined_probability, method='keep')
    
    # Generate titles
    combined_titles = list(config.selected_human_list) + list(config.selected_model_list)
    
    # Calculate accuracy and other metrics
    
    # Analyze correlation
    analyze_correlation(
        combined_probability,
        combined_titles,
        save_path=config.correlation_save_path
    )
    binned_data = calculate_binned_data(combined_probability, bin_num=10)
    alpha = calculate_krippendorff_alpha(binned_data)
    print(f'alpha: {alpha}')
    # Plot histograms
    plot_histograms(
        combined_probability,
        titles=combined_titles,
        figsize=(36, 2 * len(combined_titles)),
        save_path=config.histogram_save_path
    )
    result,labels = calculate_score(combined_probability, labels, combined_titles)

    # Calculate Krippendorff's Alpha
    alpha_result = calculate_krippendorff_alpha_for_models(
        human_probability,
        llm_probability,
        config.selected_human_list,
        config.selected_model_list
    )


    return result, labels, alpha_result

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function: execute complete analysis pipeline"""
    # Initialize configuration
    config = AnalysisConfig()
    
    # Load human annotation data
    human_data = load_human_data(config.human_data_path)
    
    # Extract human annotation probabilities
    human_probability = extract_human_probabilities(human_data)
    
    # Extract labels (using reference model)
    reference_model_path = os.path.join(config.result_folder, 'qwen_3_32b_thinking.json')
    labels = extract_labels(human_data, reference_model_path)
    
    # Extract LLM model probabilities
    llm_probability = extract_llm_probabilities(
        human_data,
        config.model_list,
        config.result_folder
    )


    result, label_, alpha_result = perform_analysis(
        human_probability,
        llm_probability,
        labels,
        config
    )



if __name__ == "__main__":

    main()