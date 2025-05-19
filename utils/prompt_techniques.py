from typing import Dict, Any, List, Callable

def create_direct_prompt(question: Dict[str, Any]) -> str:
    """
    Create a simple direct prompt for the question
    
    Args:
        question: Question dictionary from the dataset
        
    Returns:
        Formatted prompt string
    """
    q_type = question["type"]
    
    prompt = f"""
You are an expert in solving {question['subject']} problems. 
Answer the following question:

{question['question']}

IMPORTANT INSTRUCTIONS:
- The question type is: {q_type}
- Your final answer must be in the following format based on the question type:
  - If MCQ: Your answer must be a single option (A, B, C, or D)
  - If MCQ(multiple): Your answer must be a combination of options (like AC, BCD, etc.)
  - If Integer: Your answer must be a non-negative integer
  - If Numeric: Your answer must be a number with up to 2 decimal places

At the end of your response, include a line that starts with "FINAL ANSWER:" followed by your answer in the required format.
"""
    return prompt

def create_cot_prompt(question: Dict[str, Any]) -> str:
    """
    Create a Chain of Thought prompt for the question
    
    Args:
        question: Question dictionary from the dataset
        
    Returns:
        Formatted prompt string
    """
    q_type = question["type"]
    
    prompt = f"""
You are an expert in solving {question['subject']} problems. 
Answer the following question using step-by-step reasoning:

{question['question']}

IMPORTANT INSTRUCTIONS:
- The question type is: {q_type}
- First, break down the problem and solve it step by step
- Show your complete reasoning process
- Your final answer must be in the following format based on the question type:
  - If MCQ: Your answer must be a single option (A, B, C, or D)
  - If MCQ(multiple): Your answer must be a combination of options (like AC, BCD, etc.)
  - If Integer: Your answer must be a non-negative integer
  - If Numeric: Your answer must be a number with up to 2 decimal places

At the end of your response, include a line that starts with "FINAL ANSWER:" followed by your answer in the required format.
"""
    return prompt

def create_cot_sc_single_trace_prompt(question: Dict[str, Any], trace_id: int = 1) -> str:
    """
    Create a Chain of Thought prompt for a single trace in CoT with Self Consistency
    
    Args:
        question: Question dictionary from the dataset
        trace_id: ID of the current trace (used for tracking but not included in prompt)
        
    Returns:
        Formatted prompt string
    """
    q_type = question["type"]
    
    prompt = f"""
You are an expert in solving {question['subject']} problems. 
Answer the following question using step-by-step reasoning:

{question['question']}

IMPORTANT INSTRUCTIONS:
- The question type is: {q_type}
- Break down the problem and solve it step by step
- Show your complete reasoning process
- Your final answer must be in the following format based on the question type:
  - If MCQ: Your answer must be a single option (A, B, C, or D)
  - If MCQ(multiple): Your answer must be a combination of options (like AC, BCD, etc.)
  - If Integer: Your answer must be a non-negative integer
  - If Numeric: Your answer must be a number with up to 2 decimal places

At the end of your response, include a line that starts with "FINAL ANSWER:" followed by your answer in the required format.
"""
    return prompt

def create_cot_sc_prompt(question: Dict[str, Any], num_traces: int = 3) -> str:
    """
    Create a Chain of Thought with Self Consistency prompt for the question
    This is used when sending a single request (not the proper CoT-SC implementation)
    
    Args:
        question: Question dictionary from the dataset
        num_traces: Number of reasoning traces to generate
        
    Returns:
        Formatted prompt string
    """
    q_type = question["type"]
    
    prompt = f"""
You are an expert in solving {question['subject']} problems. 
Answer the following question using multiple independent reasoning paths:

{question['question']}

IMPORTANT INSTRUCTIONS:
- The question type is: {q_type}
- Generate {num_traces} different reasoning paths to solve this problem
- For each path:
  1. Label it as "Reasoning Path #N:" (where N is 1, 2, etc.)
  2. Solve the problem step by step using a different approach
  3. Provide a conclusion for that path
- After generating all paths, determine the most consistent answer across all paths
- Your final answer must be in the following format based on the question type:
  - If MCQ: Your answer must be a single option (A, B, C, or D)
  - If MCQ(multiple): Your answer must be a combination of options (like AC, BCD, etc.)
  - If Integer: Your answer must be a non-negative integer
  - If Numeric: Your answer must be a number with up to 2 decimal places

At the end of your response, include a line that starts with "FINAL ANSWER:" followed by your answer in the required format.
"""
    return prompt

def get_prompt_function(technique: str) -> Callable:
    """
    Get the appropriate prompt function based on technique
    
    Args:
        technique: Prompting technique name
        
    Returns:
        Prompt function
    """
    techniques = {
        "direct": create_direct_prompt,
        "cot": create_cot_prompt,
        "cot_sc": create_cot_sc_prompt
    }
    
    if technique not in techniques:
        raise ValueError(f"Technique {technique} not supported")
    
    return techniques[technique]
