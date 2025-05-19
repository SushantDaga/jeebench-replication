import json
import os
import datetime
import sys
import re
from typing import Dict, Any, List, Optional, Union

def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename by replacing invalid characters
    
    Args:
        name: The string to sanitize
        
    Returns:
        Sanitized string safe for use in filenames
    """
    # Replace characters that are problematic in filenames
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    result = name
    for char in invalid_chars:
        result = result.replace(char, '_')
    return result

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
        # Sanitize experiment name for safe file paths
        safe_name = sanitize_filename(experiment_name)
        filename = f"{safe_name}_{timestamp}.json"
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
        # Sanitize experiment name for safe file paths
        safe_name = sanitize_filename(experiment_name)
        filename = f"{safe_name}_accuracy_{timestamp}.json"
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
        # Sanitize experiment name for safe file paths
        safe_name = sanitize_filename(experiment_name)
        filename = f"{safe_name}_config_{timestamp}.json"
    else:
        filename = f"config_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Save config
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath

def save_incremental_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    experiment_name: str,
    current_index: int,
    total_count: int
) -> str:
    """
    Save results incrementally during processing
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        experiment_name: Name of the experiment
        current_index: Current question index
        total_count: Total number of questions
        
    Returns:
        Path to the saved checkpoint file
    """
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize experiment name for safe file paths
    safe_name = sanitize_filename(experiment_name)
    
    # Generate checkpoint filename
    filename = f"{safe_name}_checkpoint_{current_index}_of_{total_count}_{timestamp}.json"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Create checkpoint data with progress information
    checkpoint_data = {
        "results": results,
        "progress": {
            "current_index": current_index,
            "total_count": total_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_name": experiment_name
        }
    }
    
    # Save checkpoint
    with open(filepath, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Also save a "latest" checkpoint that gets overwritten
    latest_path = os.path.join(checkpoint_dir, f"{safe_name}_latest.json")
    with open(latest_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    return filepath

def find_latest_checkpoint(output_dir: str, experiment_name: str = None) -> Optional[str]:
    """
    Find the latest checkpoint file for an experiment
    
    Args:
        output_dir: Directory to search for checkpoints
        experiment_name: Optional name of the experiment
        
    Returns:
        Path to the latest checkpoint file, or None if not found
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    
    # First try to find the latest checkpoint file
    if experiment_name:
        safe_name = sanitize_filename(experiment_name)
        latest_path = os.path.join(checkpoint_dir, f"{safe_name}_latest.json")
        if os.path.exists(latest_path):
            return latest_path
    
    # If no latest file or no experiment name, find the most recent checkpoint
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.json'):
            if not experiment_name:
                # If no experiment name provided, include all checkpoints
                filepath = os.path.join(checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
            else:
                # If experiment name provided, check if the filename starts with the sanitized name
                safe_name = sanitize_filename(experiment_name)
                if filename.startswith(safe_name):
                    filepath = os.path.join(checkpoint_dir, filename)
                    checkpoints.append((filepath, os.path.getmtime(filepath)))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing results and progress information
    """
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    return checkpoint_data
