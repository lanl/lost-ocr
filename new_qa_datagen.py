"""
Question-Answer Pair Generation Module

This module generates synthetic question-answer pairs from document abstracts using
Large Language Models (LLMs). It processes OCR-extracted text files and creates
Q&A pairs for evaluating retrieval systems and measuring hallucination in RAG pipelines.

Key Features:
- Batch processing of documents organized by distortion level
- Robust JSON parsing with error recovery
- Rate limit handling with exponential backoff
- Progress tracking with tqdm
- JSONL output format for efficient streaming

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

import os
import json
import logging
import time
import openai
from tqdm import tqdm
import re

# Set up logging configuration for debugging and monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SambanovaAPIClient:
    """
    Client for interacting with SambaNova's LLM API for Q&A generation.
    
    This class provides a wrapper around OpenAI's API client configured to work
    with SambaNova's endpoints. It's specifically optimized for generating
    high-quality question-answer pairs from document abstracts.
    
    Attributes:
        model_name (str): The LLM model to use for generation
        api_key (str): Authentication key for API access
        api_base (str): Base URL for SambaNova API
    """
    
    def __init__(self, model_name="Meta-Llama-3.3-70B-Instruct", api_key=None):
        """
        Initialize the SambaNova API client for Q&A generation.
        
        Args:
            model_name (str): The SambaNova model to use. Default is 
                            "Meta-Llama-3.3-70B-Instruct" which provides
                            good balance between quality and speed
            api_key (str, optional): API key for authentication. If None,
                                    will look for OPENAI_API_KEY in environment
        
        Raises:
            ValueError: If no API key is provided or found
        """
        self.model_name = model_name
        
        # Attempt to get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")

        # Configure OpenAI client to use SambaNova endpoint
        openai.api_key = api_key
        openai.api_base = "https://api.sambanova.ai/v1"

    def predict(self, prompt_content, temperature=0.4):
        """
        Generate Q&A pairs using the LLM with robust error handling.
        
        This method sends a prompt to the API and handles various failure modes
        including rate limiting, network issues, and API errors. It uses
        exponential backoff for retries to ensure reliable batch processing.
        
        Args:
            prompt_content (str): The formatted prompt containing the abstract
                                and instructions for Q&A generation
            temperature (float): Controls creativity/randomness. Default 0.4
                               provides good balance between creativity and
                               consistency for Q&A generation
        
        Returns:
            str: Generated Q&A pairs in JSON format, or error message if
                generation fails after all retries
        
        Note:
            The method implements intelligent retry logic with exponential
            backoff, starting from 1 second and doubling up to 10 retries
        """
        max_retries = 10  # Maximum retry attempts
        retry_delay = 1   # Initial delay between retries (seconds)

        for attempt in range(max_retries):
            try:
                # Make API call to generate Q&A pairs
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant generating Q&A pairs."
                        },
                        {
                            "role": "user", 
                            "content": prompt_content
                        }
                    ],
                    temperature=temperature
                )
                # Extract and return the generated content
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                # Handle rate limiting with exponential backoff
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    # Log error for other failure modes
                    logger.error(f"API error: {e}")
                    return f"ERROR: {e}"


def generate_qa_pairs(client, abstract):
    """
    Generate question-answer pairs from a document abstract.
    
    This function uses prompt engineering to generate exactly 10 nuanced Q&A pairs
    from a given abstract. The pairs are designed to test understanding and
    retrieval capabilities while being challenging enough to reveal hallucination.
    
    Args:
        client (SambanovaAPIClient): Configured API client for generation
        abstract (str): The document abstract/text to generate Q&A pairs from
    
    Returns:
        list: List of dictionaries, each containing 'question' and 'answer' keys.
              Returns empty list if generation fails or parsing errors occur.
    
    Prompt Engineering Notes:
        - Requests exactly 10 pairs for consistency
        - Emphasizes "nuanced" to get non-trivial questions
        - Requires JSON format for easy parsing
        - Explicitly forbids additional commentary to ensure clean output
    """
    # Carefully crafted prompt for consistent Q&A generation
    prompt = (
        "Given the following abstract, generate exactly 10 nuanced Q&A pairs. "
        "Respond ONLY with valid JSON as a list of dictionaries, each containing "
        "exactly two keys: 'question' and 'answer'. "
        "Do not add any other commentary or text.\n\n"
        f"Abstract:\n{abstract}"
    )
    
    # Get response from API
    response = client.predict(prompt)
    
    # Check for API errors
    if response.startswith("ERROR") or "deprecated" in response.lower():
        logger.error(f"API returned error: {response}")
        return []

    # Use robust JSON extraction to handle potential formatting issues
    def robust_json_extract(response_text):
        """
        Extract Q&A pairs from potentially malformed JSON response.
        
        This inner function handles cases where the LLM doesn't return
        perfectly formatted JSON, using regex to extract valid Q&A pairs.
        
        Args:
            response_text (str): Raw text response from LLM
        
        Returns:
            list: Extracted Q&A pairs as list of dictionaries
        """
        # Regex pattern to extract individual Q&A dictionaries
        # Handles variations in whitespace and formatting
        pattern = r'\{\s*"question"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern, response_text)

        if not matches:
            logger.error("No valid Q&A pairs found in model output.")
            return []

        # Convert regex matches to dictionary format
        qa_pairs = [{"question": q, "answer": a} for q, a in matches]
        return qa_pairs

    return robust_json_extract(response)


def process_llama_txt_files(root_dir, output_dir, client, files_per_jsonl=5000):
    """
    Process all LLaMA-extracted text files and generate Q&A pairs.
    
    This function traverses the directory structure containing OCR-extracted text,
    generates Q&A pairs for each document, and saves them in JSONL format.
    It's designed to handle large datasets efficiently with batch saving.
    
    Directory Structure Expected:
        root_dir/
        ├── easy/
        ├── medium/
        └── hard/
            └── {doc_folder}/
                └── llama_txt/
                    └── *.txt (OCR extracted text files)
    
    Args:
        root_dir (str): Root directory containing the document hierarchy
        output_dir (str): Directory to save generated Q&A pairs
        client (SambanovaAPIClient): API client for Q&A generation
        files_per_jsonl (int): Number of files to include in each JSONL batch.
                              Default 5000 for manageable file sizes.
    
    Output Format:
        JSONL files with each line containing:
        {
            "file_title": "document_name.txt",
            "abstract": "full text content",
            "qa_pairs": [{"question": "...", "answer": "..."}, ...]
        }
    
    Features:
        - Progress tracking with tqdm
        - Automatic batching to prevent memory issues
        - Skips invalid or empty abstracts
        - Saves progress incrementally
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize batch tracking variables
    qa_batch = []        # Current batch of Q&A pairs
    file_count = 0       # Files in current batch
    batch_number = 1     # Current batch number

    # Process each difficulty level
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_dir = os.path.join(root_dir, difficulty)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(difficulty_dir):
            continue

        # Process each document folder
        for subdir in os.listdir(difficulty_dir):
            # Path to llama_txt directory
            subdir_path = os.path.join(difficulty_dir, subdir, "llama_txt")
            
            # Skip if llama_txt directory doesn't exist
            if not os.path.isdir(subdir_path):
                continue

            # Get all text files in the directory
            txt_files = [f for f in os.listdir(subdir_path) if f.endswith(".txt")]
            
            # Process each text file with progress bar
            for txt_file in tqdm(txt_files, desc=f"Processing {difficulty}/{subdir}"):
                txt_path = os.path.join(subdir_path, txt_file)
                
                # Read the abstract/document text
                with open(txt_path, 'r', encoding='utf-8') as f:
                    abstract = f.read().strip()

                # Validate abstract content
                if not abstract or abstract.startswith("ERROR"):
                    logger.info(f"Skipping invalid or empty abstract: {txt_path}")
                    continue

                # Generate Q&A pairs for this abstract
                qa_pairs = generate_qa_pairs(client, abstract)
                
                # Create entry for this document
                entry = {
                    "file_title": txt_file,
                    "abstract": abstract,
                    "qa_pairs": qa_pairs
                }
                
                # Add to current batch
                qa_batch.append(entry)
                file_count += 1

                # Save batch if it reaches the limit
                if file_count >= files_per_jsonl:
                    output_path = os.path.join(output_dir, f"qa_pairs_batch_{batch_number}.jsonl")
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        for item in qa_batch:
                            out_f.write(json.dumps(item) + "\n")
                    logger.info(f"Saved {file_count} entries to {output_path}")
                    
                    # Reset for next batch
                    qa_batch = []
                    file_count = 0
                    batch_number += 1

    # Save any remaining entries in the final batch
    if qa_batch:
        output_path = os.path.join(output_dir, f"qa_pairs_batch_{batch_number}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for item in qa_batch:
                out_f.write(json.dumps(item) + "\n")
        logger.info(f"Saved remaining {len(qa_batch)} entries to {output_path}")


def safe_json_parse(response):
    """
    Safely parse potentially malformed JSON with automatic fixing.
    
    This function attempts to parse JSON and implements common fixes for
    issues that can occur with LLM-generated JSON, such as truncation or
    missing brackets.
    
    Args:
        response (str): Raw JSON string to parse
    
    Returns:
        list/dict: Parsed JSON object, or empty list if parsing fails
    
    Common Fixes Applied:
        - Adds missing opening bracket
        - Removes trailing partial objects
        - Adds missing closing bracket
    """
    try:
        # Try standard JSON parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # Attempt to auto-fix common issues
        logger.warning("Attempting to fix malformed JSON from model output.")
        response = response.strip()
        
        # Fix missing opening bracket
        if not response.startswith("["):
            response = "[" + response
            
        # Fix missing closing bracket and remove partial objects
        if not response.endswith("]"):
            # Remove trailing partial object (incomplete JSON)
            response = re.sub(r',\s*\{[^}]*$', '', response)
            response += "]"
            
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to fix and load JSON: {e}")
            return []


def main():
    """
    Main entry point for Q&A generation pipeline.
    
    This function sets up the configuration and initiates the batch processing
    of documents to generate Q&A pairs. It's designed to be run as a standalone
    script or imported as a module.
    
    Configuration:
        - API Key: Your SambaNova API key
        - Root Directory: Path to processed OCR text files
        - Output Directory: Where to save generated Q&A pairs
    
    Usage:
        python new_qa_datagen.py
        
    Output:
        Creates JSONL files in output directory with Q&A pairs for each document
    """
    # API configuration
    api_key = "your-api-key-here"  # Replace with your actual API key
    
    # Initialize API client
    client = SambanovaAPIClient(api_key=api_key)
    
    # Set directories
    root_dir = "/path/to/your/data"  # Update to your data directory
    output_dir = "./qa_pairs_output_llama"
    
    # Start processing
    logger.info("Starting Q&A pair generation...")
    process_llama_txt_files(root_dir, output_dir, client)
    logger.info("Q&A generation complete!")


if __name__ == "__main__":
    main()
