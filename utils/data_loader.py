import json
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse the dataset.json file
    
    Args:
        file_path: Path to the dataset.json file
        
    Returns:
        List of question dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_questions_by_subject(data: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
    """
    Filter questions by subject (chem, phy, math)
    
    Args:
        data: List of question dictionaries
        subject: Subject to filter by (chem, phy, math)
        
    Returns:
        Filtered list of question dictionaries
    """
    return [q for q in data if q['subject'] == subject]

def get_questions_by_type(data: List[Dict[str, Any]], q_type: str) -> List[Dict[str, Any]]:
    """
    Filter questions by type (MCQ, MCQ(multiple), Integer, Numeric)
    
    Args:
        data: List of question dictionaries
        q_type: Question type to filter by
        
    Returns:
        Filtered list of question dictionaries
    """
    return [q for q in data if q['type'] == q_type]
