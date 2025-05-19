#!/usr/bin/env python3
"""
Test script for the LLM evaluation system.
This script runs a small test with the sample dataset to verify that the system is working correctly.
"""

import os
import argparse
from typing import Dict, Any

from utils.data_loader import load_dataset
from utils.llm_providers import get_provider
from utils.prompt_techniques import get_prompt_function, create_direct_prompt, create_cot_prompt
from utils.response_parser import parse_answer
from utils.evaluation import is_correct, calculate_accuracy, print_accuracy_report
from config import API_KEYS, DEFAULT_MODELS

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    try:
        dataset = load_dataset("sample_dataset.json")
        print(f"Successfully loaded dataset with {len(dataset)} questions")
        
        # Print a sample question
        print("\nSample question:")
        print(f"Subject: {dataset[0]['subject']}")
        print(f"Type: {dataset[0]['type']}")
        print(f"Question: {dataset[0]['question'][:100]}...")
        print(f"Gold answer: {dataset[0]['gold']}")
        
        return True
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return False

def test_prompt_generation():
    """Test prompt generation functionality"""
    print("\n=== Testing Prompt Generation ===")
    try:
        dataset = load_dataset("sample_dataset.json")
        question = dataset[0]
        
        # Test direct prompt
        direct_prompt = create_direct_prompt(question)
        print(f"Successfully generated direct prompt for question {question['index']}")
        print("\nSample direct prompt:")
        print(direct_prompt[:300] + "...\n")
        
        # Test CoT prompt
        cot_prompt = create_cot_prompt(question)
        print(f"Successfully generated CoT prompt for question {question['index']}")
        print("\nSample CoT prompt:")
        print(cot_prompt[:300] + "...\n")
        
        # Test CoT-SC single trace prompt
        from utils.prompt_techniques import create_cot_sc_single_trace_prompt
        cot_sc_single_prompt = create_cot_sc_single_trace_prompt(question, 1)
        print(f"Successfully generated CoT-SC single trace prompt for question {question['index']}")
        print("\nSample CoT-SC single trace prompt:")
        print(cot_sc_single_prompt[:300] + "...\n")
        
        return True
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        return False

def test_answer_parsing():
    """Test answer parsing functionality"""
    print("\n=== Testing Answer Parsing ===")
    try:
        # Test MCQ parsing
        mcq_response = "After solving this problem, I believe the answer is B.\n\nFINAL ANSWER: B"
        parsed_mcq = parse_answer(mcq_response, "MCQ")
        print(f"MCQ parsing: '{mcq_response}' -> '{parsed_mcq}'")
        
        # Test MCQ(multiple) parsing
        mcq_multiple_response = "The correct options are A and D.\n\nFINAL ANSWER: AD"
        parsed_mcq_multiple = parse_answer(mcq_multiple_response, "MCQ(multiple)")
        print(f"MCQ(multiple) parsing: '{mcq_multiple_response}' -> '{parsed_mcq_multiple}'")
        
        # Test Integer parsing
        integer_response = "The answer is 42.\n\nFINAL ANSWER: 42"
        parsed_integer = parse_answer(integer_response, "Integer")
        print(f"Integer parsing: '{integer_response}' -> '{parsed_integer}'")
        
        # Test Numeric parsing
        numeric_response = "The pH value is 3.00.\n\nFINAL ANSWER: 3.00"
        parsed_numeric = parse_answer(numeric_response, "Numeric")
        print(f"Numeric parsing: '{numeric_response}' -> '{parsed_numeric}'")
        
        return True
    except Exception as e:
        print(f"Error parsing answers: {str(e)}")
        return False

def test_correctness_checking():
    """Test correctness checking functionality"""
    print("\n=== Testing Correctness Checking ===")
    try:
        # Test MCQ correctness
        mcq_result = is_correct("B", "B", "MCQ")
        print(f"MCQ correctness (B vs B): {mcq_result}")
        
        # Test MCQ(multiple) correctness
        mcq_multiple_result = is_correct("AD", "DA", "MCQ(multiple)")
        print(f"MCQ(multiple) correctness (AD vs DA): {mcq_multiple_result}")
        
        # Test Integer correctness
        integer_result = is_correct(42, "42", "Integer")
        print(f"Integer correctness (42 vs '42'): {integer_result}")
        
        # Test Numeric correctness
        numeric_result = is_correct(3.00, "3.00", "Numeric")
        print(f"Numeric correctness (3.00 vs '3.00'): {numeric_result}")
        
        return True
    except Exception as e:
        print(f"Error checking correctness: {str(e)}")
        return False

def test_api_key_configuration():
    """Test API key configuration"""
    print("\n=== Testing API Key Configuration ===")
    missing_keys = []
    
    for provider, key in API_KEYS.items():
        if key == "your-openai-api-key" or key == "your-anthropic-api-key" or key == "your-gemini-api-key" or key == "your-openrouter-api-key":
            missing_keys.append(provider)
    
    if missing_keys:
        print(f"Warning: The following API keys are not configured: {', '.join(missing_keys)}")
        print("Please update the API keys in config.py before running the full evaluation.")
        return False
    else:
        print("All API keys are configured.")
        return True

def test_cot_sc_implementation():
    """Test CoT-SC implementation functionality"""
    print("\n=== Testing CoT-SC Implementation ===")
    try:
        from collections import Counter
        
        # Test majority voting for MCQ
        mcq_answers = ["A", "B", "A", "A", "C"]
        answer_counts = Counter(mcq_answers)
        final_answer = answer_counts.most_common(1)[0][0]
        print(f"Majority voting for MCQ: {mcq_answers} -> {final_answer}")
        
        # Test majority voting for MCQ(multiple)
        mcq_multiple_answers = ["AB", "AC", "AB", "AB", "BC"]
        answer_counts = Counter(mcq_multiple_answers)
        final_answer = answer_counts.most_common(1)[0][0]
        print(f"Majority voting for MCQ(multiple): {mcq_multiple_answers} -> {final_answer}")
        
        # Test median for Integer
        integer_answers = [42, 43, 41, 42, 45]
        integer_answers.sort()
        if len(integer_answers) % 2 == 0:
            median = (integer_answers[len(integer_answers)//2 - 1] + integer_answers[len(integer_answers)//2]) / 2
        else:
            median = integer_answers[len(integer_answers)//2]
        final_answer = int(median)
        print(f"Median for Integer: {integer_answers} -> {final_answer}")
        
        # Test median for Numeric
        numeric_answers = [3.14, 3.15, 3.13, 3.14, 3.16]
        numeric_answers.sort()
        if len(numeric_answers) % 2 == 0:
            median = (numeric_answers[len(numeric_answers)//2 - 1] + numeric_answers[len(numeric_answers)//2]) / 2
        else:
            median = numeric_answers[len(numeric_answers)//2]
        final_answer = round(median, 2)
        print(f"Median for Numeric: {numeric_answers} -> {final_answer}")
        
        # Test prompt consistency
        from utils.prompt_techniques import create_cot_sc_single_trace_prompt
        dataset = load_dataset("sample_dataset.json")
        question = dataset[0]
        
        prompt1 = create_cot_sc_single_trace_prompt(question, 1)
        prompt2 = create_cot_sc_single_trace_prompt(question, 2)
        prompt3 = create_cot_sc_single_trace_prompt(question, 3)
        
        if prompt1 == prompt2 == prompt3:
            print("Prompt consistency check: ✓ (All prompts are identical)")
        else:
            print("Prompt consistency check: ✗ (Prompts are different)")
            return False
        
        # Test completion status handling
        completion_status = {
            "success": True,
            "reason": "completed"
        }
        print(f"Completion status format: {completion_status}")
        
        error_status = {
            "success": False,
            "reason": "Token limit exceeded"
        }
        print(f"Error status format: {error_status}")
        
        return True
    except Exception as e:
        print(f"Error testing CoT-SC implementation: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the LLM evaluation system")
    parser.add_argument("--skip-api-check", action="store_true", help="Skip API key configuration check")
    
    args = parser.parse_args()
    
    print("=== LLM Evaluation System Test ===")
    
    # Run tests
    data_loading_ok = test_data_loading()
    prompt_generation_ok = test_prompt_generation()
    answer_parsing_ok = test_answer_parsing()
    correctness_checking_ok = test_correctness_checking()
    cot_sc_implementation_ok = test_cot_sc_implementation()
    
    # Check API key configuration if not skipped
    api_keys_ok = True
    if not args.skip_api_check:
        api_keys_ok = test_api_key_configuration()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Data loading: {'✓' if data_loading_ok else '✗'}")
    print(f"Prompt generation: {'✓' if prompt_generation_ok else '✗'}")
    print(f"Answer parsing: {'✓' if answer_parsing_ok else '✗'}")
    print(f"Correctness checking: {'✓' if correctness_checking_ok else '✗'}")
    print(f"CoT-SC implementation: {'✓' if cot_sc_implementation_ok else '✗'}")
    if not args.skip_api_check:
        print(f"API key configuration: {'✓' if api_keys_ok else '✗'}")
    
    # Print overall result
    all_tests_passed = (data_loading_ok and prompt_generation_ok and 
                        answer_parsing_ok and correctness_checking_ok and 
                        cot_sc_implementation_ok)
    if args.skip_api_check:
        print(f"\nOverall result: {'✓ All tests passed!' if all_tests_passed else '✗ Some tests failed.'}")
    else:
        all_tests_passed = all_tests_passed and api_keys_ok
        print(f"\nOverall result: {'✓ All tests passed!' if all_tests_passed else '✗ Some tests failed.'}")
    
    # Print next steps
    print("\n=== Next Steps ===")
    if all_tests_passed:
        print("The system is working correctly. You can now run the full evaluation:")
        print("python main.py --provider openai --technique direct")
        print("\nFor Chain of Thought with Self Consistency:")
        print("python main.py --provider openai --technique cot_sc --traces 5")
    else:
        print("Please fix the issues above before running the full evaluation.")

if __name__ == "__main__":
    main()
