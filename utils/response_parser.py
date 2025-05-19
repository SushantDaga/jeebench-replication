import re
from typing import Dict, Any, Optional, Union

def parse_mcq_answer(answer: str) -> str:
    """
    Parse MCQ answer (single option)
    
    Args:
        answer: Raw answer string
        
    Returns:
        Parsed answer (A, B, C, or D)
    """
    # Clean and extract just the option letter
    answer = answer.strip().upper()
    match = re.search(r'[ABCD]', answer)
    if match:
        return match.group(0)
    return answer

def parse_mcq_multiple_answer(answer: str) -> str:
    """
    Parse MCQ(multiple) answer (combination of options)
    
    Args:
        answer: Raw answer string
        
    Returns:
        Parsed answer (combination of A, B, C, D)
    """
    # Clean and extract just the option letters
    answer = answer.strip().upper()
    # Extract all occurrences of A, B, C, D
    matches = re.findall(r'[ABCD]', answer)
    if matches:
        # Remove duplicates and sort
        unique_matches = sorted(set(matches))
        return ''.join(unique_matches)
    return answer

def parse_integer_answer(answer: str) -> int:
    """
    Parse Integer answer
    
    Args:
        answer: Raw answer string
        
    Returns:
        Parsed integer answer
    """
    # Extract the first integer from the answer
    match = re.search(r'\d+', answer)
    if match:
        return int(match.group(0))
    return 0  # Default if no integer found

def parse_numeric_answer(answer: str) -> float:
    """
    Parse Numeric answer (float with up to 2 decimal places)
    
    Args:
        answer: Raw answer string
        
    Returns:
        Parsed numeric answer
    """
    # Extract the first float from the answer
    match = re.search(r'-?\d+(\.\d+)?', answer)
    if match:
        # Round to 2 decimal places
        return round(float(match.group(0)), 2)
    return 0.0  # Default if no numeric value found

def extract_final_answer(response: str) -> str:
    """
    Extract the final answer from the response
    
    Args:
        response: Raw LLM response
        
    Returns:
        Extracted final answer
    """
    # Look for "FINAL ANSWER:" pattern
    match = re.search(r'FINAL ANSWER:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If no explicit final answer, return an empty string
    # This will likely cause the answer to be incorrect, but that's better than
    # potentially extracting the wrong answer from elsewhere in the response
    return ""

def parse_answer(response: str, question_type: str) -> Union[str, int, float]:
    """
    Parse the LLM response based on question type
    
    Args:
        response: Raw LLM response
        question_type: Type of question (MCQ, MCQ(multiple), Integer, Numeric)
        
    Returns:
        Parsed answer in the appropriate format
    """
    final_answer = extract_final_answer(response)
    
    parsers = {
        "MCQ": parse_mcq_answer,
        "MCQ(multiple)": parse_mcq_multiple_answer,
        "Integer": parse_integer_answer,
        "Numeric": parse_numeric_answer
    }
    
    if question_type not in parsers:
        raise ValueError(f"Question type {question_type} not supported")
    
    return parsers[question_type](final_answer)

def parse_provider_response(provider: str, response: Dict[str, Any], question_type: str) -> Union[str, int, float]:
    """
    Parse the response from a specific provider
    
    Args:
        provider: Provider name
        response: Provider response dictionary
        question_type: Type of question
        
    Returns:
        Parsed answer
    """
    # Extract the text response
    if provider == "openai":
        text_response = response.get("response", "")
    elif provider == "anthropic":
        text_response = response.get("response", "")
    elif provider == "gemini":
        text_response = response.get("response", "")
    elif provider == "openrouter":
        text_response = response.get("response", "")
    else:
        raise ValueError(f"Provider {provider} not supported")
    
    # Parse the answer based on question type
    return parse_answer(text_response, question_type)
