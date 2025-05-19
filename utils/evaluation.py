from typing import Dict, Any, List, Union

def is_correct(parsed_answer: Union[str, int, float], gold_answer: Union[str, int, float], question_type: str) -> bool:
    """
    Check if the parsed answer matches the gold answer
    
    Args:
        parsed_answer: Parsed answer from LLM
        gold_answer: Gold answer from dataset
        question_type: Type of question
        
    Returns:
        Boolean indicating if the answer is correct
    """
    if question_type == "MCQ":
        # Case-insensitive comparison for MCQ
        return str(parsed_answer).upper() == str(gold_answer).upper()
    
    elif question_type == "MCQ(multiple)":
        # Sort the letters for MCQ(multiple) to handle different orderings
        parsed_sorted = ''.join(sorted(str(parsed_answer).upper()))
        gold_sorted = ''.join(sorted(str(gold_answer).upper()))
        return parsed_sorted == gold_sorted
    
    elif question_type == "Integer":
        # Convert to integers for comparison
        try:
            return int(parsed_answer) == int(gold_answer)
        except (ValueError, TypeError):
            return False
    
    elif question_type == "Numeric":
        # Compare with tolerance for floating point
        try:
            return abs(float(parsed_answer) - float(gold_answer)) < 0.01
        except (ValueError, TypeError):
            return False
    
    return False

def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate accuracy metrics from results
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with accuracy metrics
    """
    total = len(results)
    if total == 0:
        return {
            'overall': 0.0,
            'by_subject': {},
            'by_type': {},
            'by_provider': {},
            'by_technique': {}
        }
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    
    # Overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Accuracy by subject
    subjects = set(r['question']['subject'] for r in results)
    subject_accuracy = {}
    for subject in subjects:
        subject_results = [r for r in results if r['question']['subject'] == subject]
        subject_total = len(subject_results)
        subject_correct = sum(1 for r in subject_results if r.get('is_correct', False))
        subject_accuracy[subject] = subject_correct / subject_total if subject_total > 0 else 0
    
    # Accuracy by question type
    types = set(r['question']['type'] for r in results)
    type_accuracy = {}
    for q_type in types:
        type_results = [r for r in results if r['question']['type'] == q_type]
        type_total = len(type_results)
        type_correct = sum(1 for r in type_results if r.get('is_correct', False))
        type_accuracy[q_type] = type_correct / type_total if type_total > 0 else 0
    
    # Accuracy by provider
    providers = set(r.get('provider', 'unknown') for r in results)
    provider_accuracy = {}
    for provider in providers:
        provider_results = [r for r in results if r.get('provider', 'unknown') == provider]
        provider_total = len(provider_results)
        provider_correct = sum(1 for r in provider_results if r.get('is_correct', False))
        provider_accuracy[provider] = provider_correct / provider_total if provider_total > 0 else 0
    
    # Accuracy by technique
    techniques = set(r.get('technique', 'unknown') for r in results)
    technique_accuracy = {}
    for technique in techniques:
        technique_results = [r for r in results if r.get('technique', 'unknown') == technique]
        technique_total = len(technique_results)
        technique_correct = sum(1 for r in technique_results if r.get('is_correct', False))
        technique_accuracy[technique] = technique_correct / technique_total if technique_total > 0 else 0
    
    # Accuracy by provider and technique
    provider_technique_accuracy = {}
    for provider in providers:
        provider_technique_accuracy[provider] = {}
        for technique in techniques:
            pt_results = [r for r in results 
                         if r.get('provider', 'unknown') == provider 
                         and r.get('technique', 'unknown') == technique]
            pt_total = len(pt_results)
            pt_correct = sum(1 for r in pt_results if r.get('is_correct', False))
            provider_technique_accuracy[provider][technique] = pt_correct / pt_total if pt_total > 0 else 0
    
    return {
        'overall': accuracy,
        'by_subject': subject_accuracy,
        'by_type': type_accuracy,
        'by_provider': provider_accuracy,
        'by_technique': technique_accuracy,
        'by_provider_technique': provider_technique_accuracy,
        'total_questions': total,
        'correct_answers': correct
    }

def print_accuracy_report(accuracy: Dict[str, Any]) -> None:
    """
    Print a formatted accuracy report
    
    Args:
        accuracy: Accuracy metrics dictionary
    """
    print("\n===== ACCURACY REPORT =====")
    print(f"Overall Accuracy: {accuracy['overall']:.2%} ({accuracy.get('correct_answers', 0)}/{accuracy.get('total_questions', 0)})")
    
    print("\n--- By Subject ---")
    for subject, acc in accuracy.get('by_subject', {}).items():
        print(f"{subject}: {acc:.2%}")
    
    print("\n--- By Question Type ---")
    for q_type, acc in accuracy.get('by_type', {}).items():
        print(f"{q_type}: {acc:.2%}")
    
    print("\n--- By Provider ---")
    for provider, acc in accuracy.get('by_provider', {}).items():
        print(f"{provider}: {acc:.2%}")
    
    print("\n--- By Technique ---")
    for technique, acc in accuracy.get('by_technique', {}).items():
        print(f"{technique}: {acc:.2%}")
    
    print("\n--- By Provider and Technique ---")
    for provider, techniques in accuracy.get('by_provider_technique', {}).items():
        print(f"\n{provider}:")
        for technique, acc in techniques.items():
            print(f"  {technique}: {acc:.2%}")
    
    print("\n============================")
