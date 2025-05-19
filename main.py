#!/usr/bin/env python3
"""
Main execution script for LLM evaluation system.
"""

import os
import argparse
import json
import signal
import sys
from typing import Dict, Any, List
import time

from utils.data_loader import load_dataset
from utils.llm_providers import get_provider
from utils.prompt_techniques import get_prompt_function
from utils.response_parser import parse_answer
from utils.evaluation import is_correct, calculate_accuracy, print_accuracy_report
from utils.storage import (
    save_results, 
    save_accuracy, 
    save_experiment_config,
    save_incremental_results,
    find_latest_checkpoint,
    load_checkpoint,
    sanitize_filename
)
from config import (
    API_KEYS, 
    DEFAULT_MODELS, 
    DEFAULT_PARAMS, 
    DEFAULT_NUM_TRACES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_DATASET_PATH,
    PARAM_DESCRIPTIONS
)

from collections import Counter

# Global flag to track interruption
interrupted = False

def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C)"""
    global interrupted
    if not interrupted:
        print("\nKeyboard interrupt received. Completing current evaluation and saving results...")
        interrupted = True
    else:
        print("\nSecond interrupt received. Exiting immediately...")
        sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def run_evaluation(
    dataset_path: str,
    provider_name: str,
    model_name: str,
    technique: str,
    num_traces: int = DEFAULT_NUM_TRACES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int = None,
    verbose: bool = False,
    llm_params: Dict[str, Any] = None,
    resume_from: int = None,
    checkpoint_path: str = None,
    save_every: int = 5  # Save every N questions
) -> Dict[str, Any]:
    """
    Run evaluation on the dataset
    
    Args:
        dataset_path: Path to the dataset
        provider_name: Name of the LLM provider
        model_name: Name of the model
        technique: Prompting technique
        num_traces: Number of traces for CoT with Self Consistency
        output_dir: Directory to save results
        limit: Optional limit on number of questions to evaluate
        verbose: Whether to print verbose output
        llm_params: Custom parameters for the LLM API calls
        resume_from: Index to resume from (0-based)
        checkpoint_path: Path to a checkpoint file to resume from
        save_every: Save checkpoint every N questions
        
    Returns:
        Dictionary with evaluation results
    """
    # Create experiment name (sanitization will be done in the storage functions)
    experiment_name = f"{provider_name}_{model_name}_{technique}"
    
    # Initialize results list and start index
    results = []
    start_index = 0
    
    # Check if resuming from checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path)
        if "results" in checkpoint_data:
            results = checkpoint_data["results"]
            if "progress" in checkpoint_data and "current_index" in checkpoint_data["progress"]:
                start_index = checkpoint_data["progress"]["current_index"]
            else:
                # Assume we've completed all questions in the results list
                start_index = len(results)
            print(f"Resuming from question {start_index+1}")
    elif resume_from is not None:
        start_index = resume_from
        print(f"Resuming from question {start_index+1}")
    else:
        # Look for latest checkpoint if not explicitly specified
        latest_checkpoint = find_latest_checkpoint(output_dir, experiment_name)
        if latest_checkpoint:
            print(f"Found latest checkpoint: {latest_checkpoint}")
            checkpoint_data = load_checkpoint(latest_checkpoint)
            if "results" in checkpoint_data:
                results = checkpoint_data["results"]
                if "progress" in checkpoint_data and "current_index" in checkpoint_data["progress"]:
                    start_index = checkpoint_data["progress"]["current_index"]
                else:
                    start_index = len(results)
                print(f"Resuming from question {start_index+1}")
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Limit dataset if specified
    if limit is not None and limit > 0:
        dataset = dataset[:limit]
    
    # Get provider
    provider = get_provider(provider_name, API_KEYS[provider_name])
    
    # Get prompt function
    prompt_func = get_prompt_function(technique)
    
    # Use provided parameters or get defaults
    params = llm_params if llm_params is not None else DEFAULT_PARAMS.get(provider_name, {})
    
    # Track start time for progress reporting
    start_time = time.time()
    
    # Process questions starting from start_index
    for i in range(start_index, len(dataset)):
        # Check for keyboard interrupt
        if interrupted:
            print(f"\nInterrupted at question {i+1}/{len(dataset)}. Saving results...")
            break
            
        question = dataset[i]
        print(f"Processing question {i+1}/{len(dataset)}...")
        
        try:
            # Handle CoT-SC differently
            if technique == "cot_sc":
                # For CoT-SC, we send multiple independent requests
                trace_responses = []
                trace_answers = []
                
                for trace_id in range(1, num_traces + 1):
                    print(f"  Generating trace {trace_id}/{num_traces}...")
                    
                    # Create prompt for this trace
                    from utils.prompt_techniques import create_cot_sc_single_trace_prompt
                    trace_prompt = create_cot_sc_single_trace_prompt(question, trace_id)
                    
                    if verbose:
                        print(f"\nPrompt for trace {trace_id}:\n{trace_prompt[:300]}...\n")
                    
                    # Generate response for this trace
                    try:
                        trace_response = provider.generate(
                            prompt=trace_prompt,
                            model=model_name,
                            **params
                        )
                        
                        # Add completion status information
                        trace_response["completion_status"] = {
                            "success": True,
                            "reason": trace_response.get("finish_reason", "completed")
                        }
                        
                        if verbose:
                            print(f"\nResponse for trace {trace_id}:\n{trace_response['response'][:300]}...\n")
                        
                        # Parse answer for this trace
                        trace_answer = parse_answer(
                            trace_response["response"],
                            question["type"]
                        )
                    except Exception as e:
                        error_message = str(e)
                        print(f"Error generating response for trace {trace_id}: {error_message}")
                        
                        # Create a minimal response with error information
                        trace_response = {
                            "provider": provider_name,
                            "model": model_name,
                            "response": f"Error: {error_message}",
                            "raw_response": None,
                            "completion_status": {
                                "success": False,
                                "reason": error_message
                            }
                        }
                        
                        # Use a fallback answer
                        if question["type"] in ["MCQ", "MCQ(multiple)"]:
                            trace_answer = "A"  # Fallback to first option
                        elif question["type"] == "Integer":
                            trace_answer = 0
                        else:  # Numeric
                            trace_answer = 0.0
                    
                    if verbose:
                        print(f"Parsed answer for trace {trace_id}: {trace_answer}")
                    
                    # Store trace response and answer
                    trace_responses.append(trace_response)
                    trace_answers.append(trace_answer)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1)
                
                # Perform majority voting to determine the final answer
                if question["type"] in ["MCQ", "MCQ(multiple)"]:
                    # For MCQ and MCQ(multiple), use the most common answer
                    answer_counts = Counter(trace_answers)
                    final_answer = answer_counts.most_common(1)[0][0]
                elif question["type"] in ["Integer", "Numeric"]:
                    # For Integer and Numeric, use the median answer
                    try:
                        # Convert all answers to float for comparison
                        numeric_answers = [float(ans) for ans in trace_answers]
                        # Sort the answers
                        numeric_answers.sort()
                        # Get the median
                        if len(numeric_answers) % 2 == 0:
                            # Even number of answers, take the average of the middle two
                            median = (numeric_answers[len(numeric_answers)//2 - 1] + numeric_answers[len(numeric_answers)//2]) / 2
                        else:
                            # Odd number of answers, take the middle one
                            median = numeric_answers[len(numeric_answers)//2]
                        
                        # Convert back to the appropriate type
                        if question["type"] == "Integer":
                            final_answer = int(median)
                        else:  # Numeric
                            final_answer = round(median, 2)
                    except (ValueError, TypeError):
                        # If conversion fails, use the first answer as a fallback
                        final_answer = trace_answers[0]
                else:
                    # For any other type, use the first answer
                    final_answer = trace_answers[0]
                
                # Check correctness
                correct = is_correct(
                    final_answer,
                    question["gold"],
                    question["type"]
                )
                
                if verbose:
                    print(f"All trace answers: {trace_answers}")
                    print(f"Final answer (majority vote): {final_answer}")
                    print(f"Gold answer: {question['gold']}")
                    print(f"Correct: {correct}")
                
                # Process trace responses to make them JSON serializable
                processed_trace_responses = []
                for trace_response in trace_responses:
                    # Create a copy without the raw_response field
                    processed_response = {k: v for k, v in trace_response.items() if k != 'raw_response'}
                    processed_trace_responses.append(processed_response)
                
                # Store result
                result = {
                    "question": question,
                    "trace_prompts": [create_cot_sc_single_trace_prompt(question, t) for t in range(1, num_traces + 1)],
                    "trace_responses": processed_trace_responses,  # Store processed response objects
                    "trace_answers": trace_answers,
                    "final_answer": final_answer,
                    "gold_answer": question["gold"],
                    "is_correct": correct,
                    "provider": provider_name,
                    "model": model_name,
                    "technique": technique,
                    "num_traces": num_traces,
                    "llm_params": params  # Store the LLM parameters used
                }
                
                results.append(result)
                
                # Save incremental results
                if (i + 1) % save_every == 0 or interrupted:
                    checkpoint_path = save_incremental_results(
                        results,
                        output_dir,
                        experiment_name,
                        i + 1,
                        len(dataset)
                    )
                    print(f"Checkpoint saved: {checkpoint_path}")
                
            else:
                # For direct and CoT, we send a single request
                # Create prompt
                prompt = prompt_func(question)
                
                if verbose:
                    print(f"\nPrompt:\n{prompt}\n")
                
                # Generate response
                response_data = provider.generate(
                    prompt=prompt,
                    model=model_name,
                    **params
                )
                
                if verbose:
                    print(f"\nResponse:\n{response_data['response']}\n")
                
                # Parse answer
                parsed_answer = parse_answer(
                    response_data["response"],
                    question["type"]
                )
                
                if verbose:
                    print(f"Parsed answer: {parsed_answer}")
                    print(f"Gold answer: {question['gold']}")
                
                # Check correctness
                correct = is_correct(
                    parsed_answer,
                    question["gold"],
                    question["type"]
                )
                
                if verbose:
                    print(f"Correct: {correct}")
                
                # Add completion status information
                response_data["completion_status"] = {
                    "success": True,
                    "reason": response_data.get("finish_reason", "completed")
                }
                
                # Process response to make it JSON serializable
                processed_response = {k: v for k, v in response_data.items() if k != 'raw_response'}
                
                # Store result
                result = {
                    "question": question,
                    "prompt": prompt,
                    "response": processed_response,  # Store processed response object
                    "parsed_answer": parsed_answer,
                    "gold_answer": question["gold"],
                    "is_correct": correct,
                    "provider": provider_name,
                    "model": model_name,
                    "technique": technique,
                    "llm_params": params  # Store the LLM parameters used
                }
                
                results.append(result)
                
                # Save incremental results
                if (i + 1) % save_every == 0 or interrupted:
                    checkpoint_path = save_incremental_results(
                        results,
                        output_dir,
                        experiment_name,
                        i + 1,
                        len(dataset)
                    )
                    print(f"Checkpoint saved: {checkpoint_path}")
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
        except Exception as e:
            print(f"Error processing question {i+1}: {str(e)}")
            continue
    
    # Calculate accuracy
    accuracy = calculate_accuracy(results)
    
    # Print accuracy report
    print_accuracy_report(accuracy)
    
    # Save results
    experiment_name = f"{provider_name}_{model_name}_{technique}"
    results_path = save_results(results, output_dir, experiment_name)
    accuracy_path = save_accuracy(accuracy, output_dir, experiment_name)
    
    # Save experiment config
    config = {
        "dataset_path": dataset_path,
        "provider": provider_name,
        "model": model_name,
        "technique": technique,
        "num_traces": num_traces if technique == "cot_sc" else None,
        "limit": limit,
        "llm_params": params,  # Include LLM parameters in the config
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = save_experiment_config(config, output_dir, experiment_name)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Accuracy saved to: {accuracy_path}")
    print(f"Config saved to: {config_path}")
    
    return {
        "results": results,
        "accuracy": accuracy,
        "results_path": results_path,
        "accuracy_path": accuracy_path,
        "config_path": config_path
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate LLMs on question answering")
    
    # Basic arguments
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset")
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "anthropic", "gemini", "openrouter"], help="LLM provider")
    parser.add_argument("--model", type=str, help="Model name (if not specified, uses default for provider)")
    parser.add_argument("--technique", type=str, required=True, choices=["direct", "cot", "cot_sc"], help="Prompting technique")
    parser.add_argument("--traces", type=int, default=DEFAULT_NUM_TRACES, help="Number of traces for CoT with Self Consistency")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    # Checkpoint and resume arguments
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument("--resume-from", type=int, help="Resume from a specific question index (0-based)")
    parser.add_argument("--checkpoint", type=str, help="Path to a specific checkpoint file to resume from")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N questions")
    
    # LLM parameters
    parser.add_argument("--temperature", type=float, help=PARAM_DESCRIPTIONS["temperature"])
    parser.add_argument("--max-tokens", type=int, help=PARAM_DESCRIPTIONS["max_tokens"])
    parser.add_argument("--top-p", type=float, help=PARAM_DESCRIPTIONS["top_p"])
    parser.add_argument("--top-k", type=int, help=PARAM_DESCRIPTIONS["top_k"])
    parser.add_argument("--frequency-penalty", type=float, help=PARAM_DESCRIPTIONS["frequency_penalty"])
    parser.add_argument("--presence-penalty", type=float, help=PARAM_DESCRIPTIONS["presence_penalty"])
    parser.add_argument("--stop", type=str, nargs="+", help=PARAM_DESCRIPTIONS["stop"])
    
    args = parser.parse_args()
    
    # Use default model if not specified
    model = args.model if args.model else DEFAULT_MODELS.get(args.provider)
    
    # Get default parameters for the provider
    params = DEFAULT_PARAMS.get(args.provider, {}).copy()
    
    # Override with command line arguments if provided
    if args.temperature is not None:
        params["temperature"] = args.temperature
    
    if args.max_tokens is not None:
        if args.provider == "gemini":
            params["max_output_tokens"] = args.max_tokens
        else:
            params["max_tokens"] = args.max_tokens
    
    if args.top_p is not None:
        params["top_p"] = args.top_p
    
    if args.top_k is not None and args.provider in ["anthropic", "gemini"]:
        params["top_k"] = args.top_k
    
    if args.frequency_penalty is not None and args.provider in ["openai", "openrouter"]:
        params["frequency_penalty"] = args.frequency_penalty
    
    if args.presence_penalty is not None and args.provider in ["openai", "openrouter"]:
        params["presence_penalty"] = args.presence_penalty
    
    if args.stop is not None:
        if args.provider in ["openai", "openrouter"]:
            params["stop"] = args.stop
        elif args.provider == "anthropic":
            params["stop_sequences"] = args.stop
        elif args.provider == "gemini":
            params["stop_sequences"] = args.stop
    
    # Handle resume options
    checkpoint_path = None
    resume_from = None
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        # Find the latest checkpoint
        experiment_name = f"{args.provider}_{model}_{args.technique}"
        checkpoint_path = find_latest_checkpoint(args.output, experiment_name)
        if not checkpoint_path:
            print("No checkpoint found to resume from.")
    elif args.resume_from is not None:
        resume_from = args.resume_from
    
    results = run_evaluation(
        dataset_path=args.dataset,
        provider_name=args.provider,
        model_name=model,
        technique=args.technique,
        num_traces=args.traces,
        output_dir=args.output,
        limit=args.limit,
        verbose=args.verbose,
        llm_params=params,  # Pass the custom parameters
        resume_from=resume_from,
        checkpoint_path=checkpoint_path,
        save_every=args.save_every
    )
    
    print(f"Evaluation complete!")

if __name__ == "__main__":
    main()
