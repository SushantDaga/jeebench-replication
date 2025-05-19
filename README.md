# LLM Evaluation System for JEE Questions

This project implements a system for evaluating Large Language Models (LLMs) on JEE (Joint Entrance Examination) questions. It supports multiple LLM providers and prompting techniques.

## Features

- Support for multiple LLM providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - OpenRouter
- Multiple prompting techniques:
  - Direct prompting
  - Chain of Thought (CoT)
  - Chain of Thought with Self Consistency (CoT-SC) - properly implemented with multiple independent traces and majority voting
- Automatic answer parsing based on question type:
  - MCQ (single option)
  - MCQ (multiple options)
  - Integer
  - Numeric
- Comprehensive accuracy evaluation:
  - Overall accuracy
  - Accuracy by subject
  - Accuracy by question type
  - Accuracy by provider
  - Accuracy by technique
- Result storage and analysis

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd jee
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Open `config.py`
   - Replace placeholder API keys with your actual keys

## Prompting Techniques

### Direct Prompting
The model is directly asked to solve the problem and provide an answer.

### Chain of Thought (CoT)
The model is asked to solve the problem step-by-step, showing its reasoning process before providing the final answer.

### Chain of Thought with Self Consistency (CoT-SC)
This technique involves:
1. Generating multiple independent reasoning paths (traces) for the same question
2. Getting an answer from each path
3. Using majority voting (for MCQ/MCQ(multiple)) or median (for Integer/Numeric) to determine the final answer

Key implementation details:
- Sends separate requests for each reasoning trace with identical prompts
- Uses exactly the same prompt for each trace to ensure true independence
- Captures completion status information (success/failure and reason) for debugging
- Aggregates results using majority voting or median calculation
- Handles API errors gracefully with fallback answers

## Usage

### Basic Usage

```bash
python main.py --provider openai --technique direct
```

### Command Line Arguments

#### Basic Arguments
- `--dataset`: Path to the dataset file (default: `dataset.json`)
- `--provider`: LLM provider to use (`openai`, `anthropic`, `gemini`, `openrouter`)
- `--model`: Model name (if not specified, uses default for provider)
- `--technique`: Prompting technique (`direct`, `cot`, `cot_sc`)
- `--traces`: Number of traces for CoT with Self Consistency (default: 3)
- `--output`: Output directory for results (default: `results`)
- `--limit`: Limit number of questions to evaluate
- `--verbose`: Print verbose output

#### Checkpoint and Resume Arguments
- `--resume`: Resume from the latest checkpoint
- `--resume-from`: Resume from a specific question index (0-based)
- `--checkpoint`: Path to a specific checkpoint file to resume from
- `--save-every`: Save checkpoint every N questions (default: 5)

#### LLM Parameters

You can customize LLM parameters directly from the command line:

- `--temperature`: Controls randomness (0.0 is deterministic, higher values increase randomness)
- `--max-tokens`: Maximum number of tokens to generate
- `--top-p`: Nucleus sampling (1.0 considers all tokens, lower values restrict to more likely tokens)
- `--top-k`: Only sample from the top K options for each next token (for Anthropic and Gemini)
- `--frequency-penalty`: Penalizes repeated tokens (for OpenAI and OpenRouter)
- `--presence-penalty`: Penalizes tokens already present (for OpenAI and OpenRouter)
- `--stop`: Sequences where the API will stop generating further tokens

### Examples

Evaluate OpenAI with direct prompting:
```bash
python main.py --provider openai --technique direct
```

Evaluate Anthropic with Chain of Thought:
```bash
python main.py --provider anthropic --technique cot
```

Evaluate Gemini with Chain of Thought with Self Consistency:
```bash
python main.py --provider gemini --technique cot_sc --traces 5
```

Limit evaluation to 10 questions:
```bash
python main.py --provider openrouter --technique direct --limit 10
```

Using custom LLM parameters:
```bash
python main.py --provider openai --technique direct --temperature 0.7 --max-tokens 2000
```

Using Chain of Thought with Self Consistency with custom parameters:
```bash
python main.py --provider anthropic --technique cot_sc --traces 5 --temperature 0.2 --top-p 0.9
```

Resume from the latest checkpoint:
```bash
python main.py --provider openai --technique direct --resume
```

Resume from a specific question index:
```bash
python main.py --provider openai --technique direct --resume-from 10
```

Resume from a specific checkpoint file:
```bash
python main.py --provider openai --technique direct --checkpoint results/checkpoints/openai_gpt-4_direct_latest.json
```

Save checkpoints more frequently:
```bash
python main.py --provider openai --technique direct --save-every 1
```

## Dataset Format

The dataset should be a JSON file containing a list of dictionaries, each with the following keys:

- `description`: Paper where the question occurs
- `index`: Number of question
- `subject`: Subject to which question belongs (chem, phy, math)
- `type`: Type of question (MCQ, MCQ(multiple), Integer, Numeric)
- `question`: Actual question text
- `gold`: Actual answer of question

Example:
```json
[
  {
    "description": "JEE Main 2022 Paper 1",
    "index": 1,
    "subject": "math",
    "type": "MCQ",
    "question": "If the sum of the first 20 terms of the series 1 + 3 + 5 + ... is equal to the sum of the first n terms of the series 2 + 4 + 6 + ..., then the value of n is:",
    "gold": "A"
  },
  ...
]
```

## Result Storage and Debugging

The system stores comprehensive information about each evaluation run:

- **Results**: All responses, prompts, parsed answers, and correctness information
- **Accuracy**: Overall accuracy and breakdowns by subject, question type, provider, and technique
- **Configuration**: All parameters used for the evaluation run
- **Completion Status**: For each API call, the system captures:
  - Success/failure status
  - Reason for completion (e.g., "stop", "length", "content_filter")
  - Error messages if applicable

This information is valuable for debugging and analyzing model performance.

## Robustness Features

The system includes several features to ensure robustness when working with large datasets:

### Graceful Keyboard Interrupt Handling

You can safely interrupt the evaluation process at any time by pressing Ctrl+C. The system will:
- Complete the current question evaluation
- Save all results processed so far
- Exit gracefully

Pressing Ctrl+C twice will force an immediate exit.

### Incremental Saving

Results are saved incrementally during processing:
- By default, a checkpoint is saved every 5 questions
- You can adjust this frequency with the `--save-every` parameter
- Checkpoints are stored in the `results/checkpoints` directory
- Each checkpoint contains all results processed so far and progress information

### Resume Functionality

You can resume an interrupted evaluation in several ways:
- `--resume`: Automatically find and resume from the latest checkpoint
- `--resume-from`: Resume from a specific question index
- `--checkpoint`: Resume from a specific checkpoint file

This is particularly useful for:
- Recovering from unexpected errors or interruptions
- Running evaluations in multiple sessions
- Continuing after API rate limits are hit

## Project Structure

```
jee/
├── dataset.json                 # Dataset file
├── main.py                      # Main execution script
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading and parsing
│   ├── llm_providers.py         # API clients for different LLM providers
│   ├── prompt_techniques.py     # Different prompting strategies
│   ├── response_parser.py       # Parse LLM responses (strict FINAL ANSWER parsing)
│   ├── evaluation.py            # Accuracy calculation
│   └── storage.py               # Save results to disk
├── config.py                    # Configuration (API keys, etc.)
├── requirements.txt             # Dependencies
└── results/                     # Directory to store results
```

## License

MIT
