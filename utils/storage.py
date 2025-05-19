import json
import os
import datetime
from typing import Dict, Any, List

def save_results(results: List[Dict[str, Any]], output_dir: str, experiment_name: str = None) -> str:
    """
    Save results to disk
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the saved results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    if experiment_name:
        filename = f"{experiment_name}_{timestamp}.json"
    else:
        filename = f"results_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath

def load_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Load results from disk
    
    Args:
        filepath: Path to the results file
        
    Returns:
        List of result dictionaries
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def save_accuracy(accuracy: Dict[str, Any], output_dir: str, experiment_name: str = None) -> str:
    """
    Save accuracy metrics to disk
    
    Args:
        accuracy: Accuracy metrics dictionary
        output_dir: Directory to save results
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the saved accuracy file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    if experiment_name:
        filename = f"{experiment_name}_accuracy_{timestamp}.json"
    else:
        filename = f"accuracy_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Save accuracy
    with open(filepath, 'w') as f:
        json.dump(accuracy, f, indent=2)
    
    return filepath

def save_experiment_config(config: Dict[str, Any], output_dir: str, experiment_name: str = None) -> str:
    """
    Save experiment configuration to disk
    
    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save results
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the saved config file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    if experiment_name:
        filename = f"{experiment_name}_config_{timestamp}.json"
    else:
        filename = f"config_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Save config
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath
