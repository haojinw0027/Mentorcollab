import json
import os
import argparse
import time
import torch
from datasets import load_dataset, load_from_disk
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Dict, Any
from utils import apply_instruct_template, MODEL_NAME_DICT, INSTRUCT_MODEL_LIST
import random
import math
import yaml
import random
from utils import BranchPredictionMLP

# Random seed for reproducibility (identical to single_inference.py)
RANDOM_SEED = 609
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# MMLU-STEM学科列表
def get_dataset_com_hard_intervention(split='test',
                        num_sample=None,
                        input_key='Q',
                        output_key='A',
                        **kwargs):
    dataset = []
    if split == 'test':
        with open('Com2/benckmark/com2/hard.json', 'r') as f:
            dataset = json.load(f)
    else:
        with open('Com2/benckmark/train.json', 'r') as f:
            dataset = json.load(f)
    input_data = []
    output_data = []
    if split == 'test':
        for example in dataset:
            if example['type'] == "intervention":
                input_data.append({
                    "crime": example['crime'],
                    "facts": example['facts'],
                    "question": example['Q'],
                    "options": example['O']
                })
                output_data.append(example['A'])
            else:
                continue
    else:
        for example in dataset:
            if example['type'] == "intervention":
                input_data.append({
                    "question": example['question'],
                    "options": example['options']
                })
                output_data.append(example['answer'])
            else:
                continue
    if num_sample is not None and num_sample < len(input_data):
        input_data = input_data[:num_sample]
        output_data = output_data[:num_sample]
    return input_data, output_data, input_key, output_key

def get_dataset_mmlu_pro(split='test',
                        num_sample=None,
                        input_key='question',
                        output_key='answer',
                        subject=None,
                        few_shot='yes',
                        **kwargs):
    """Load MMLU-Pro dataset from TIGER-Lab/MMLU-Pro"""
    mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro")
    validation_dataset = mmlu_pro['validation']  # Load validation set for few-shot examples
    
    if split == 'test':
        dataset = load_from_disk('data/mmlu_pro_test')
    else:
        dataset = load_from_disk('data/mmlu_pro_train')
    # fix random seed for reproducibility
    if num_sample is not None and num_sample < len(dataset):
        dataset = dataset.select(range(num_sample))
    input_data = []
    output_data = []
    for example in dataset:
        current_example = {
            "question": example['question'],
            "options": [opt for opt in example['options'] if opt != "N/A"],  # Filter out N/A options
            "answer": example['answer'],
            "category": example.get('category', 'unknown')
        }
        prompt = current_example['question'] + "\n"
        for i, choice in enumerate(current_example['options']):
            prompt += f"{chr(ord('A') + i)}. {choice}\n"
        
        input_data.append({
            "context": "",
            "input": prompt,
            "category": current_example["category"],
            "original_question": current_example["question"],
            "options": current_example["options"]
        })
        
        # Answer is the letter (A, B, C, D, etc.)
        answer_idx = example['answer']
        if isinstance(answer_idx, str):
            # If answer is already a letter, use it directly
            if len(answer_idx) == 1 and answer_idx.isalpha():
                output_data.append(answer_idx.upper())
            else:
                # Try to convert to int
                try:
                    answer_idx = int(answer_idx)
                    output_data.append(chr(65 + answer_idx))
                except ValueError:
                    output_data.append('A')  # Default fallback
        else:
            output_data.append(chr(65 + int(answer_idx)))
    
    return input_data, output_data, input_key, output_key

def get_dataset_math(split='test',
                        num_sample=None,
                        input_key='problem',
                        output_key='solution',
                        **kwargs):
    """Load MATH dataset from nlile/hendrycks-MATH-benchmark"""
    math = load_dataset("nlile/hendrycks-MATH-benchmark")
    dataset = math[split]
    
    # fix random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    if num_sample is not None and num_sample < len(dataset):
        random_indexes = np.random.choice(len(dataset), num_sample, replace=False)
        dataset = dataset.select(random_indexes)
    
    input_data = []
    output_data = []
    for example in dataset:
        input_data.append({
            "context": "",
            "input": example[input_key]
        })
        output_data.append(example[output_key])
    
    return input_data, output_data, input_key, output_key

def get_dataset_minerva(split='train',
                        num_sample=None,
                        input_key='problem',
                        output_key='solution',
                        **kwargs):
    """Load Minerva dataset from TIGER-Lab/Minerva"""
    minerva = load_dataset("knoveleng/Minerva-Math")
    dataset = minerva['train']
    np.random.seed(RANDOM_SEED)
    if num_sample is not None and num_sample < len(dataset):
        random_indexes = np.random.choice(len(dataset), num_sample, replace=False)
        dataset = dataset.select(random_indexes)
    
    input_data = []
    output_data = []
    for example in dataset:
        input_data.append({
            "context": "",
            "input": example[input_key]
        })
        output_data.append(example[output_key])
    
    return input_data, output_data, input_key, output_key

def get_dataset_supergpqa(split='test',
                         num_sample=None,
                         input_key='question',
                         output_key='answer',
                         subject=None,
                         **kwargs):
    """Load SuperGPQA dataset from m-a-p/SuperGPQA"""
    supergpqa = load_dataset("m-a-p/SuperGPQA")
    dataset = supergpqa['train']

    # Filter by subject if specified (unless subject is "mixed")
    if subject is not None and subject != "mixed":
        print(f"Filtering SuperGPQA dataset for subject: {subject}")

        # Check if subject exists in the dataset
        available_subjects = set(example.get('field', 'unknown') for example in dataset)
        if subject not in available_subjects:
            print(f"Available subjects: {sorted(available_subjects)}")
            raise ValueError(f"Subject '{subject}' not found in SuperGPQA dataset. Available subjects: {sorted(available_subjects)}")

        # Filter dataset by subject
        filtered_data = []
        for example in dataset:
            if example.get('field') == subject:
                filtered_data.append(example)

        print(f"Found {len(filtered_data)} questions for subject '{subject}'")
        dataset = filtered_data
    else:
        # If subject is None or "mixed", use all questions
        if subject == "mixed":
            print(f"Using mixed subjects - randomly sampling from all subjects in SuperGPQA dataset")
        dataset = list(dataset)
    np.random.seed(RANDOM_SEED)
    # Sample based on split: test -> from end, train -> from beginning
    if num_sample is not None and num_sample < len(dataset):
        if split == 'test':
            # For test split, select from the end
            dataset = dataset[-num_sample:]
            print(f"Selected last {num_sample} samples from test split")
        else:
            # For train split (or other splits), select from the beginning
            dataset = dataset[:num_sample]
            print(f"Selected first {num_sample} samples from {split} split")
    
    input_data = []
    output_data = []
    for example in dataset:
        # Format question with multiple choice options
        question = example.get('question', '')
        options = example.get('options', [])
        
        # Build formatted question with options
        formatted_question = f"{question}\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, etc.
            formatted_question += f"{letter}) {option}\n"
        
        input_data.append({
            "context": "",
            "input": formatted_question.strip(),
            "field": example.get('field', 'unknown'),
            "discipline": example.get('discipline', 'unknown'),
            "subfield": example.get('subfield', 'unknown')
        })
        
        # Get correct answer - handle both string and integer formats
        answer = example.get('answer_letter', 0)
        output_data.append(answer)
    
    return input_data, output_data, input_key, output_key

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TruthfulQA, HaluEval, or MMLU-Pro inference with triple model approach using vLLM")
    # Dataset selection
    parser.add_argument("--benchmark", type=str, choices=["mmlu_pro", "MATH", "supergpqa", "minerva", "com_hard_intervention"], default="mmlu_pro",
                        help="Benchmark to evaluate (mmlu_pro, MATH, or supergpqa)")
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path (e.g., Qwen/Qwen3-8B-Base)")
    parser.add_argument("--expert_model", type=str, required=True,
                        help="Expert model name/path (e.g., Qwen/Qwen3-14B)")
    parser.add_argument("--confidence_threshold", type=float, default=0.40,
                        help="Confidence threshold for activating judge model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--base_port", type=int, default=8000,
                        help="Port for base model vLLM serve endpoint")
    parser.add_argument("--expert_port", type=int, default=8001,
                        help="Port for expert model vLLM serve endpoint")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for accessing restricted models")
    parser.add_argument("--consult_model", type=str,default = 'base')
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to process")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject to filter for MMLU-Pro and SuperGPQA datasets (e.g., 'math', 'physics' for MMLU-Pro; 'Clinical Medicine', 'Biology' for SuperGPQA)")
    parser.add_argument("--few-shot", type=str, default='yes', choices=['yes', 'no'],
                        help="Whether to use few-shot prompting")
    parser.add_argument("--complete_tokens", type=int, default=16,
                        help="Number of tokens to complete")
    parser.add_argument("--use_mlp", action="store_true",
                        help="Use trained MLP for branch decision instead of base model consultation")
    parser.add_argument("--mlp_path", type=str, default=None,
                        help="Path to trained MLP model (.pth file)")
    parser.add_argument("--mlp_threshold", type=float, default=0.5,
                        help="MLP decision threshold (default: 0.5, >threshold chooses A/base)")
    parser.add_argument("--reference_model", type=str, default=None,
                        help="Reference model for extracting hidden states (should match MLP training model)")
    parser.add_argument("--reference_device", type=str, default="cuda:4",
                        help="Device for reference model")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to use for testing")
    parser.add_argument("--drop_proportion", type=int, default=5,
                        help="Proportion of samples to drop from the dataset")
    return parser.parse_args()

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load existing results from a JSON file, or initialize if not present."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return []

def save_results(results: List[Dict[str, Any]], file_path: str):
    """Save the updated results to the JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def formulate_prompt_com_hard_intervention(crime, facts, question, options, model_name):
    """
    Formulate prompt for Com Hard Intervention questions using 5-shot examples from com_hard_intervention.yaml.
    """
    with open('./config/com_hard_intervention.yaml', 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    prompt_template = yaml_content['prompt_format'][0]
    full_prompt = prompt_template.format(crime=crime, facts=facts, question=question, options=options)
    return full_prompt

def formulate_prompt_mmlu_pro(question, model_name):
    """
    Formulate prompt for MMLU-Pro questions using 5-shot CoT format.
    The question parameter already contains the full CoT prompt with few-shot examples.
    """
    # For MMLU-Pro with CoT, the question already contains the complete prompt
    # including few-shot examples and the target question, so we return it directly
    system_prompt = "Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I or J.\n"
    instruct_prompt = f"Question:\n{question}"
    response_prompt = "Answer:\nLet's think step by step. "
    return apply_instruct_template(
        model_name=model_name,
        system_prompt=system_prompt,
        instruct_prompt=instruct_prompt,
        response_prompt=response_prompt
    )
    return question

def formulate_prompt_minerva(question, model_name):
    """
    Formulate prompt for Minerva questions using 5-shot CoT format.
    """
    with open('./config/minerva.yaml', 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    prompt_template = yaml_content['prompt_format'][0]
    full_prompt = prompt_template.format(question)
    system_prompt = "Think step by step."
    return apply_instruct_template(
        model_name=model_name,
        system_prompt=system_prompt,
        instruct_prompt=full_prompt,
        response_prompt=""
    )

def formulate_prompt_math(question: str, model_name: str = "") -> str:
    """
    Formulate prompt for MATH questions using few-shot examples from train split.
    Uses 4 examples from the training set as demonstrations.
    """
    # Load few-shot examples from train split (cached for efficiency)
    if not hasattr(formulate_prompt_math, '_few_shot_examples'):
        try:
            math_train = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
            # Select first 4 examples for consistency
            formulate_prompt_math._few_shot_examples = list(math_train.select(range(4)))
        except Exception as e:
            print(f"Warning: Could not load MATH train examples: {e}")
            formulate_prompt_math._few_shot_examples = []
    
    # Build few-shot prompt
    few_shot_text = ""
    for i, example in enumerate(formulate_prompt_math._few_shot_examples):
        few_shot_text += f"Problem: {example['problem']}\n"
        few_shot_text += f"Solution: {example['solution']}\n\n"
    
    # Add current question
    full_prompt = few_shot_text + f"Problem: {question}\nSolution: "
    
    if model_name:
        system_prompt = "Think step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt="",
            add_bos=True
        )
    else:
        # Fallback to simple formatting if no model name provided
        return full_prompt

def formulate_prompt_supergpqa(question: str, model_name: str = "") -> str:
    """
    Format question for SuperGPQA using 5-shot examples from super_gpqa.yaml.
    """
    # Load prompt template from super_gpqa.yaml
    try:
        with open('./config/super_gpqa.yaml', 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
        
        # Extract the prompt template
        prompt_template = yaml_content['prompt_format'][0]
        
        # Format with the current question
        full_prompt = prompt_template.format(question)
    except FileNotFoundError:
        # Fallback to inline template if yaml file not found
        print("Warning: super_gpqa.yaml not found, using fallback prompt")
        full_prompt = f"""Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question: 
{question}

Answer: Let's think step by step. """
    
    if model_name:
        system_prompt = "You are an expert in science who answers multiple choice questions step by step."
        return apply_instruct_template(
            model_name=model_name,
            system_prompt=system_prompt,
            instruct_prompt=full_prompt,
            response_prompt="",
            add_bos=True
        )
    else:
        # Fallback to simple formatting if no model name provided
        return full_prompt

class TripleModelGenerator:
    def __init__(self, base_model_name: str, expert_model_name: str,
                 hf_token: str, base_port: int = 8000, expert_port: int = 8001, consult_model: str = 'base', complete_tokens: int = 16,
                 use_mlp: bool = False, mlp_path: str = None, mlp_threshold: float = 0.5, reference_model: str = None, reference_device: str = "cuda:4", drop_proportion: int = 5):

        self.base_model_name = base_model_name
        self.expert_model_name = expert_model_name
        self.base_port = base_port
        self.expert_port = expert_port
        self.confidence_threshold = 0.40
        self.consult_model = consult_model
        self.complete_tokens = complete_tokens
        self.use_mlp = use_mlp
        self.mlp_threshold = mlp_threshold
        self.reference_device = reference_device
        self.drop_proportion = drop_proportion
        # Test connections to vLLM serve endpoints
        self._test_connections()
        print(f"Using base model at port {base_port}")
        print(f"Using expert model at port {expert_port}")

        # Load tokenizers
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, token=hf_token, trust_remote_code=True
        )
        self.expert_tokenizer = AutoTokenizer.from_pretrained(
            expert_model_name, token=hf_token, trust_remote_code=True
        )

        # Set pad tokens
        for tokenizer in [self.base_tokenizer, self.expert_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        print("Tokenizers loaded successfully!")

        # Load MLP if requested
        self.mlp_model = None
        self.reference_model = None
        self.reference_tokenizer = None

        if use_mlp:
            if mlp_path is None:
                raise ValueError("--mlp_path is required when --use_mlp is set")
            if reference_model is None:
                raise ValueError("--reference_model is required when --use_mlp is set")

            print(f"\n=== Loading MLP for Branch Decision ===")
            print(f"MLP model path: {mlp_path}")
            print(f"MLP threshold: {mlp_threshold}")
            print(f"Reference model: {reference_model}")

            # Load reference model for extracting hidden states
            device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
            print(f"Loading reference model on {device}...")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                reference_model,
                torch_dtype=torch.bfloat16,
                device_map=self.reference_device
            )
            self.reference_model.eval()

            self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token

            # Load MLP model
            config = AutoConfig.from_pretrained(reference_model)
            # Handle different config structures (Gemma3 has text_config, others have hidden_size directly)
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                hidden_size = config.text_config.hidden_size
            else:
                raise ValueError(f"Cannot find hidden_size in config for {reference_model}")
            self.mlp_model = BranchPredictionMLP(hidden_size=hidden_size)
            self.mlp_model.load_state_dict(torch.load(mlp_path, map_location='cpu'))
            self.mlp_model = self.mlp_model.to(device)
            self.mlp_model.eval()

            print(f"✓ MLP loaded successfully!")
            print(f"✓ Reference model loaded successfully!")
            print("=" * 50)

        print("Triple model generator initialized!")
    
    def _test_connections(self):
        """Test connections to vLLM serve endpoints and verify model names"""
        print(f"Testing connection to base model endpoint at port {self.base_port}...")
        try:
            test_url = f"http://localhost:{self.base_port}/v1/models"
            response = requests.get(test_url, timeout=10)
            response.raise_for_status()
            models_info = response.json()
            
            # Extract deployed model name
            deployed_model = None
            if 'data' in models_info and len(models_info['data']) > 0:
                deployed_model = models_info['data'][0]['id']
            
            # Verify model name matches
            if deployed_model:
                # Extract model name from full path if needed
                deployed_model_name = deployed_model.split('/')[-1] if '/' in deployed_model else deployed_model
                expected_model_name = self.base_model_name.split('/')[-1] if '/' in self.base_model_name else self.base_model_name
                
                if deployed_model_name.lower() != expected_model_name.lower():
                    raise Exception(f"Model name mismatch at port {self.base_port}: expected '{self.base_model_name}' but found '{deployed_model}'")
                
                print(f"✓ Base model endpoint connected successfully (verified: {deployed_model})")
            else:
                print(f"✓ Base model endpoint connected successfully (model name verification skipped - no model info)")
                
        except Exception as e:
            raise Exception(f"Failed to connect to base model endpoint at port {self.base_port}: {e}")
        
        print(f"Testing connection to expert model endpoint at port {self.expert_port}...")
        try:
            test_url = f"http://localhost:{self.expert_port}/v1/models"
            response = requests.get(test_url, timeout=10)
            response.raise_for_status()
            models_info = response.json()
            
            # Extract deployed model name
            deployed_model = None
            if 'data' in models_info and len(models_info['data']) > 0:
                deployed_model = models_info['data'][0]['id']
            
            # Verify model name matches
            if deployed_model:
                # Extract model name from full path if needed
                deployed_model_name = deployed_model.split('/')[-1] if '/' in deployed_model else deployed_model
                expected_model_name = self.expert_model_name.split('/')[-1] if '/' in self.expert_model_name else self.expert_model_name
                
                if deployed_model_name.lower() != expected_model_name.lower():
                    raise Exception(f"Model name mismatch at port {self.expert_port}: expected '{self.expert_model_name}' but found '{deployed_model}'")
                
                print(f"✓ Expert model endpoint connected successfully (verified: {deployed_model})")
            else:
                print(f"✓ Expert model endpoint connected successfully (model name verification skipped - no model info)")
                
        except Exception as e:
            raise Exception(f"Failed to connect to expert model endpoint at port {self.expert_port}: {e}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for activating judge model."""
        self.confidence_threshold = threshold
    
    def generate_sequence(self, port: int, prompt: str, max_tokens: int = 16) -> str:
        """Generate a sequence of tokens using vLLM serve endpoint"""
        url = f"http://localhost:{port}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["choices"] and result["choices"][0]["text"]:
                return result["choices"][0]["text"]
            else:
                return ""

        except Exception as e:
            print(f"Error calling vLLM serve endpoint: {e}")
            return ""

    def get_next_token_with_logits(self, port: int, prompt: str, tokenizer) -> Tuple[int, float]:
        """Get next token and its probability using vLLM serve endpoint"""
        url = f"http://localhost:{port}/v1/completions"

        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["choices"] and result["choices"][0]["text"]:
                generated_text = result["choices"][0]["text"]
                token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                else:
                    token_id = tokenizer.eos_token_id

                # Extract probability from logprobs if available
                if "logprobs" in result["choices"][0] and result["choices"][0]["logprobs"]:
                    logprobs = result["choices"][0]["logprobs"]
                    if "token_logprobs" in logprobs and logprobs["token_logprobs"]:
                        probability = np.exp(logprobs["token_logprobs"][0]) if logprobs["token_logprobs"][0] is not None else 0.0
                    else:
                        probability = 0.5  # Default confidence
                else:
                    probability = 0.5  # Default confidence
            else:
                token_id = tokenizer.eos_token_id
                probability = 0.0

        except Exception as e:
            print(f"Error calling vLLM serve endpoint: {e}")
            token_id = tokenizer.eos_token_id
            probability = 0.0

        return token_id, probability

    def extract_hidden_state_at_branch(self, prompt: str) -> torch.Tensor:
        """
        Extract hidden state at branch point for MLP decision.
        Branch point is right before "My choice: Option ("

        Args:
            prompt: Full prompt up to the branch point

        Returns:
            Hidden state tensor
        """
        if self.reference_model is None:
            raise ValueError("Reference model not loaded. Set --use_mlp and --reference_model")

        # Find branch point
        branch_marker = "My choice: Option ("
        if branch_marker not in prompt:
            raise ValueError(f"Branch marker '{branch_marker}' not found in prompt")

        # Get text up to branch point
        text_before_branch = prompt[:prompt.index(branch_marker)]

        # Tokenize
        input_ids = self.reference_tokenizer(text_before_branch, return_tensors="pt").input_ids
        device = next(self.reference_model.parameters()).device
        input_ids = input_ids.to(device)

        # Extract hidden state
        with torch.no_grad():
            outputs = self.reference_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Get last hidden state from last layer
            last_hidden_state = outputs.hidden_states[-1]
            # Extract hidden state at last token position
            hidden_state = last_hidden_state[0, -1, :].cpu()

        return hidden_state

    def predict_branch_with_mlp(self, prompt: str) -> Tuple[str, float]:
        """
        Use MLP to predict which branch (A or B) to take.

        Args:
            prompt: Full prompt including branch point

        Returns:
            Tuple of (choice, score) where choice is 'A' or 'B' and score is MLP output
        """
        if self.mlp_model is None:
            raise ValueError("MLP model not loaded. Set --use_mlp and --mlp_path")

        # Extract hidden state
        hidden_state = self.extract_hidden_state_at_branch(prompt)

        # Predict with MLP
        device = next(self.mlp_model.parameters()).device
        with torch.no_grad():
            hidden_state = hidden_state.unsqueeze(0).float().to(device)
            score = self.mlp_model(hidden_state).item()

        # Decision: score > threshold → A, else → B
        choice = 'A' if score > self.mlp_threshold else 'B'

        return choice, score

    def generate_with_triple_model(self, prompt: str, max_new_tokens: int = 64, benchmark: str = "truthfulqa") -> Dict[str, Any]:
        """Generate text using triple model approach."""
        generated_tokens = []
        token_choices = []
        base_model_count = 0
        expert_model_count = 0
        judge_decisions = 0
        next_tokens = []
        
        current_prompt = prompt
        original_prompt = prompt
        generated_text = ""
        original_question = prompt.split("Question: ")[1].split("\n")[0] if "Question: " in prompt else ""
        while True:
            # Initialize variables for this step
            newly_generated_text = current_prompt[len(original_prompt):]
            newly_generated_length = len(self.base_tokenizer.encode(newly_generated_text, add_special_tokens=False))
            if newly_generated_length >= max_new_tokens:
                break
            base_token = ''
            expert_token = ''
            expert_probability = 0.0
            # Get next token from base model
            base_token_id, base_probability = self.get_next_token_with_logits(
                self.base_port, current_prompt, self.base_tokenizer
            )
            initial_base_probability = base_probability
            base_token = self.base_tokenizer.decode([base_token_id], skip_special_tokens=True)
            process = 0
            while True:
                if process > 10:
                    break
                if not base_token:
                    break
                base_token_id, base_probability = self.get_next_token_with_logits(
                    self.base_port, current_prompt + base_token, self.base_tokenizer
                )
                _base_token = self.base_tokenizer.decode([base_token_id], skip_special_tokens=True)
                if not _base_token:
                    break
                if _base_token[0] == ' ' or _base_token[0] == '\n':
                    break
                else:
                    base_token += _base_token
                process += 1
            random_choice = random.randint(1,100)
            if random_choice >= self.drop_proportion:
                next_token = base_token
                model_used = "base_none_mentor"
                base_model_count += 1
                confidence = initial_base_probability
            else:
                expert_token_id, expert_probability = self.get_next_token_with_logits(
                    self.expert_port, current_prompt, self.expert_tokenizer
                )
                expert_token = self.expert_tokenizer.decode([expert_token_id], skip_special_tokens=True)
                initial_expert_probability = expert_probability
                # 改
                process = 0
                while True:
                    if process > 10:
                        break
                    if not expert_token:
                        break
                    expert_token_id, expert_probability = self.get_next_token_with_logits(
                        self.expert_port, current_prompt + expert_token, self.expert_tokenizer
                    )
                    _expert_token = self.expert_tokenizer.decode([expert_token_id], skip_special_tokens=True)
                    if not _expert_token:
                        break
                    if _expert_token[0] == ' ' or _expert_token[0] == '\n':
                        break
                    else:
                        expert_token += _expert_token
                    process += 1
                if base_token == expert_token:
                    next_token = base_token
                    model_used = "base_same"
                    base_model_count += 1
                    confidence = initial_base_probability
                else:
                    base_completion = self.generate_sequence(self.base_port, current_prompt, max_tokens=self.complete_tokens)
                    expert_completion = self.generate_sequence(self.expert_port, current_prompt, max_tokens=self.complete_tokens)
                    consult_prompt = current_prompt + f"\nNow I will choose the next sequence that could lead to the correct answer. Option (A): {base_completion}, Option (B): {expert_completion}. My choice: Option ("

                    # Use MLP or base model for decision
                    if self.use_mlp:
                        # MLP-based decision
                        try:
                            mlp_choice, mlp_score = self.predict_branch_with_mlp(consult_prompt)
                            consult_token = mlp_choice
                            print(f"MLP Decision: {mlp_choice} (score: {mlp_score:.4f}, threshold: {self.mlp_threshold})")
                        except Exception as e:
                            print(f"MLP Error: {e}, falling back to base model")
                            consult_token_id, consult_probability = self.get_next_token_with_logits(self.base_port, consult_prompt, self.base_tokenizer)
                            consult_token = self.base_tokenizer.decode([consult_token_id], skip_special_tokens=True)
                    else:
                        # Original base model consultation
                        consult_token_id, consult_probability = self.get_next_token_with_logits(self.base_port, consult_prompt, self.base_tokenizer)
                        consult_token = self.base_tokenizer.decode([consult_token_id], skip_special_tokens=True)

                    if consult_token == "A" or consult_token == " A":
                        next_token = base_completion
                        model_used = "base_mentor_mlp" if self.use_mlp else "base_mentor"
                        base_model_count += 1
                        confidence = initial_base_probability
                        base_consult = base_completion
                        expert_consult = expert_completion
                    elif consult_token == "B" or consult_token == " B":
                        next_token = expert_completion
                        model_used = "expert_mentor_mlp" if self.use_mlp else "expert_mentor"
                        expert_model_count += 1
                        confidence = initial_expert_probability
                        base_consult = base_completion
                        expert_consult = expert_completion
                    else:
                        print(f"Error: {consult_token}")
                        if self.consult_model == 'base':
                            model_used = "base_error"
                        elif self.consult_model == 'expert':
                            model_used = "expert_error"
                        next_token = base_completion
                        confidence = initial_base_probability
                        base_consult = base_completion
                        expert_consult = expert_completion
                # Store token information with all probabilities
            token_choices.append({
                "selected_token_text": next_token,
                "base_token_text": base_token,
                "expert_token_text": expert_token,
                "random_choice": random_choice,
                "model_used": model_used,
                "confidence": confidence,
                "base_confidence": initial_base_probability,
                "expert_confidence": initial_expert_probability if expert_token!='' else 0.0,
                "base_consult": base_consult if model_used == "base_mentor_mlp" or model_used == "expert_mentor_mlp" or model_used == "base_error" or model_used == "expert_error" else "",
                "expert_consult": expert_consult if model_used == "base_mentor_mlp" or model_used == "expert_mentor_mlp" or model_used == "base_error" or model_used == "expert_error" else "",
                'mlp_score': mlp_score if self.use_mlp and (model_used == "base_mentor_mlp" or model_used == "expert_mentor_mlp") else None
            })
            if next_token == "":
                break
            if len(next_tokens) < 3:
                next_tokens.append(next_token)
            else:
                next_tokens.pop(0)
                next_tokens.append(next_token)
            if len(next_tokens) == 3 and next_tokens[0] == next_tokens[1] == next_tokens[2]:
                break
            # Update prompt and generated text
            generated_text += next_token
            current_prompt = prompt + generated_text
            
            # Check for end conditions
            if any(stop.lower() in generated_text.lower() for stop in ['\n\nPremises:', '\n\nProblem:', '\n\nQuestion:', '<|endoftext|>', '</s>']):
                break
        newly_generated_text = current_prompt[len(original_prompt):]
        newly_generated_length = len(self.base_tokenizer.encode(newly_generated_text, add_special_tokens=False))
        return {
            "generated_text": generated_text.strip(),
            "token_choices": token_choices,
            "base_model_tokens": base_model_count,
            "total_tokens": newly_generated_length,
            "expert_model_tokens": expert_model_count,
            "judge_decisions": judge_decisions,
            "base_usage_ratio": base_model_count / len(token_choices) if len(token_choices) else 0,
            "expert_usage_ratio": expert_model_count / len(token_choices) if len(token_choices) else 0,
        }

def main():
    args = parse_arguments()
    if args.base_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Base model {args.base_model} not found in MODEL_NAME_DICT")
    if args.expert_model not in MODEL_NAME_DICT.keys():
        raise ValueError(f"Expert model {args.expert_model} not found in MODEL_NAME_DICT")
    
    # Set HuggingFace token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    
    print(f"Base model endpoint: http://localhost:{args.base_port}")
    print(f"Expert model endpoint: http://localhost:{args.expert_port}")
    
    # Load dataset based on benchmark
    if args.benchmark == "mmlu_pro":
        print("Loading MMLU-Pro dataset...")
        input_data, output_data, input_key, output_key = get_dataset_mmlu_pro(
            split='test',
            num_sample=args.num_samples,
            subject=args.subject,
            few_shot=args.few_shot
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out,
                'category': inp.get('category', 'unknown'),
                'original_question': inp.get('original_question', ''),
                'options': inp.get('options', [])
            })
    elif args.benchmark == "com_hard_intervention":
        input_data, output_data, input_key, output_key = get_dataset_com_hard_intervention(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'crime': inp['crime'],
                'facts': inp['facts'],
                'question': inp['question'],
                'options': inp['options'],
                'correct_answer': out
            })
    elif args.benchmark == "MATH":
        print("Loading MATH dataset...")
        input_data, output_data, input_key, output_key = get_dataset_math(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out
            })
    elif args.benchmark == "supergpqa":
        print("Loading SuperGPQA dataset...")
        input_data, output_data, input_key, output_key = get_dataset_supergpqa(
            split=args.split,
            num_sample=args.num_samples,
            subject=args.subject
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out,
                'field': inp.get('field', 'unknown'),
                'discipline': inp.get('discipline', 'unknown'),
                'subfield': inp.get('subfield', 'unknown')
            })
    elif args.benchmark == 'minerva':
        print("Loading Minerva dataset...")
        input_data, output_data, input_key, output_key = get_dataset_minerva(
            split=args.split,
            num_sample=args.num_samples
        )
        # Convert to format similar to other datasets
        test_set = []
        for i, (inp, out) in enumerate(zip(input_data, output_data)):
            test_set.append({
                'id': i,
                'question': inp['input'],
                'correct_answer': out
            })
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # Initialize triple model generator
    generator = TripleModelGenerator(
        base_model_name=args.base_model,
        expert_model_name=args.expert_model,
        hf_token=args.hf_token,
        base_port=args.base_port,
        expert_port=args.expert_port,
        consult_model=args.consult_model,
        complete_tokens=args.complete_tokens,
        use_mlp=args.use_mlp,
        mlp_path=args.mlp_path,
        mlp_threshold=args.mlp_threshold,
        reference_model=args.reference_model,
        reference_device=args.reference_device,
        drop_proportion=args.drop_proportion
    )
    generator.set_confidence_threshold(args.confidence_threshold)
    
    # Create output file path
    # Handle subject-specific output directory for MMLU-Pro and SuperGPQA
    if (args.benchmark == 'mmlu_pro' or args.benchmark == 'supergpqa') and args.subject:
        dataset_dir = f'{args.benchmark}_{args.subject}_results'
    else:
        dataset_dir = f'{args.benchmark}_results'
    
    if args.consult_model == 'base':
        if args.few_shot == 'yes':
            output_file = f'./result/{dataset_dir}/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/self_judge_word_seq_{args.complete_tokens}_mlp_mixed_random_{args.drop_proportion}.json'
        else:
            output_file = f'./result/{dataset_dir}/zero_shot/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/self_judge_word_seq_{args.complete_tokens}_mlp_mixed_random_{args.drop_proportion}.json'
    if args.consult_model == 'expert':
        if args.few_shot == 'yes':
            output_file = f'./result/{dataset_dir}/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/self_judge_word_seq_{args.complete_tokens}_mlp_mixed_random_{args.drop_proportion}.json'
        else:
            output_file = f'./result/{dataset_dir}/zero_shot/{MODEL_NAME_DICT[args.base_model]}/{args.split}/{MODEL_NAME_DICT[args.expert_model]}/self_judge_word_seq_{args.complete_tokens}_mlp_mixed_random_{args.drop_proportion}.json'
    
    print(f"Output file: {output_file}")
    print(f"Configuration:")
    print(f"  - Benchmark: {args.benchmark}")
    print(f"  - Base model: {args.base_model}")
    print(f"  - Expert model: {args.expert_model}")
    print(f"  - Confidence threshold: {args.confidence_threshold}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Base model port: {args.base_port}")
    print(f"  - Expert model port: {args.expert_port}")
    print(f"  - Split: {args.split}")
    print(f"  - Drop proportion: {args.drop_proportion}")
    # Load existing results
    results = load_results(output_file)
    processed_ids = {result['id'] for result in results}
    
    # Prepare unprocessed items based on benchmark
    unprocessed_items = []
    if args.benchmark == "mmlu_pro":
        for i, item in enumerate(test_set):
            # Create unique ID for mmlu_pro
            item_id = f"mmlu_pro###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "MATH":
        for i, item in enumerate(test_set):
            # Create unique ID for MATH
            item_id = f"MATH###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "supergpqa":
        for i, item in enumerate(test_set):
            # Create unique ID for supergpqa
            item_id = f"supergpqa###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "minerva":
        for i, item in enumerate(test_set):
            # Create unique ID for minerva
            item_id = f"minerva###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    elif args.benchmark == "com_hard_intervention":
        for i, item in enumerate(test_set):
            # Create unique ID for com_hard_intervention
            item_id = f"com_hard_intervention###{i}"
            if item_id not in processed_ids:
                item_with_id = dict(item)
                item_with_id['id'] = item_id
                unprocessed_items.append(item_with_id)
    if not unprocessed_items:
        print("All items have been processed already!")
        return
    
    print(f"Processing {len(unprocessed_items)} unprocessed items...")
    
    # Process items
    for item in tqdm(unprocessed_items, desc="Processing questions"):
        # Generate prompt based on benchmark
        if args.benchmark == "mmlu_pro":
            prompt = formulate_prompt_mmlu_pro(item["question"], args.base_model)
            item_id = item["id"]
            question = item.get("original_question", item["question"])
        elif args.benchmark == "MATH":
            prompt = formulate_prompt_math(item["question"], args.base_model)
            item_id = item["id"]
            question = item["question"]
        elif args.benchmark == "supergpqa":
            prompt = formulate_prompt_supergpqa(item["question"], args.base_model)
            item_id = item["id"]
            question = item["question"]
        elif args.benchmark == "minerva":
            prompt = formulate_prompt_minerva(item["question"], args.base_model)
            item_id = item["id"]
            question = item["question"]
        elif args.benchmark == "com_hard_intervention":
            prompt = formulate_prompt_com_hard_intervention(item["crime"], item["facts"], item["question"], item["options"], args.base_model)
            item_id = item["id"]
            question = item["question"]
        # Generate response using triple model approach
        generation_result = generator.generate_with_triple_model(prompt, args.max_new_tokens, args.benchmark)
        # Save result
        result = {
            "id": item_id,
            "question": question,
            "prompt": prompt,
            "answer": generation_result["generated_text"],
            "base_model_tokens": generation_result["base_model_tokens"],
            "expert_model_tokens": generation_result["expert_model_tokens"],
            "total_tokens": generation_result["total_tokens"],
            "judge_decisions": generation_result["judge_decisions"],
            "base_usage_ratio": generation_result["base_usage_ratio"],
            "expert_usage_ratio": generation_result["expert_usage_ratio"],
            "token_choices": generation_result["token_choices"],
            "base_model": args.base_model,
            "expert_model": args.expert_model,
            "confidence_threshold": args.confidence_threshold,
            "benchmark": args.benchmark,
            "base_port": args.base_port,
            "expert_port": args.expert_port,
            "correct_answer": item['gold_answer'] if args.benchmark == "tinymmlu" else "",
            "choices": item['choices'] if args.benchmark == "tinymmlu" else "",
            "subject": item['subject'] if args.benchmark == "tinymmlu" else ""
        }
        
        # Add benchmark-specific fields
        if args.benchmark == "mmlu_pro":
            result["correct_answer"] = item["correct_answer"]
            result["category"] = item.get("category", "unknown")
            result["options"] = item.get("options", [])
        elif args.benchmark == "MATH":
            result["correct_answer"] = item["correct_answer"]
        elif args.benchmark == "supergpqa":
            result["correct_answer"] = item["correct_answer"]
            result["field"] = item.get("field", "unknown")
            result["discipline"] = item.get("discipline", "unknown")
            result["subfield"] = item.get("subfield", "unknown")
        elif args.benchmark == "minerva":
            result["correct_answer"] = item["correct_answer"]
        elif args.benchmark == "com_hard_intervention":
            result["correct_answer"] = item["correct_answer"]
        results.append(result)
        save_results(results, output_file)

    # Calculate statistics
    total_base_tokens = sum(result["base_model_tokens"] for result in results)
    total_expert_tokens = sum(result["expert_model_tokens"] for result in results)
    total_tokens = total_base_tokens + total_expert_tokens
    
    print(f"\nInference completed! Results saved to {output_file}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Total responses generated: {len(results)}")
    
    print(f"\nToken usage statistics:")
    print(f"  - Base model tokens: {total_base_tokens} ({total_base_tokens/total_tokens*100:.1f}%)")
    print(f"  - Expert model tokens: {total_expert_tokens} ({total_expert_tokens/total_tokens*100:.1f}%)")
    print(f"  - Total tokens: {total_tokens}")
    
    # Calculate confidence statistics
    all_base_confidences = []
    all_expert_confidences = []
    
    for result in results:
        for token_choice in result.get("token_choices", []):
            all_base_confidences.append(token_choice.get("base_confidence", 0))
            all_expert_confidences.append(token_choice.get("expert_confidence", 0))
    
    if all_base_confidences:
        print(f"\nConfidence statistics:")
        print(f"  - Average base confidence: {sum(all_base_confidences)/len(all_base_confidences):.3f}")
        if all_expert_confidences:
            print(f"  - Average expert confidence (when consulted): {sum(all_expert_confidences)/len(all_expert_confidences):.3f}")

if __name__ == "__main__":
    main() 
