"""
SambaNova LLaMA Vision OCR Extraction Module

This module handles Optical Character Recognition (OCR) using the LLaMA Vision model
(Llama-3.2-90B-Vision-Instruct) through SambaNova's API. It processes document images
and extracts text content, organizing outputs by distortion level (easy/medium/hard).

Key Features:
- Batch processing of document images with automatic retry logic
- Exponential backoff for rate limit handling
- Preserves directory structure for organized output
- Skips already processed files to support resumption

Author: Alex Most, Manish Bhattarai
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

import os
import time
import base64
import logging
import openai

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SambanovaAPIClient:
    """
    Client for interacting with SambaNova's LLaMA Vision API.
    
    This class wraps the OpenAI-compatible API to provide OCR functionality
    using large vision-language models. It includes automatic retry logic
    for handling rate limits and API errors.
    
    Attributes:
        model_name (str): Name of the SambaNova model to use
        api_key (str): API key for authentication
        api_base (str): Base URL for SambaNova API endpoint
    """
    
    def __init__(self, model_name="Llama-3.2-90B-Vision-Instruct", api_key=None):
        """
        Initialize the SambaNova API client.
        
        Args:
            model_name (str): The SambaNova model to use for OCR extraction.
                            Default is "Llama-3.2-90B-Vision-Instruct"
            api_key (str, optional): API key for SambaNova. If None, will look for
                                    OPENAI_API_KEY environment variable
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.model_name = model_name

        # Try to get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")

        # Configure OpenAI package to use SambaNova's endpoint
        # This allows us to use OpenAI's client library with SambaNova's API
        openai.api_key = api_key
        openai.api_base = "https://api.sambanova.ai/v1"

    def predict(self, image, temperature=0.3):
        """
        Extract text from an image using LLaMA Vision model.
        
        This method sends an image to the API and receives extracted text.
        It includes sophisticated retry logic with exponential backoff to handle
        rate limiting and temporary API issues gracefully.
        
        Args:
            image (str): Base64 encoded image string
            temperature (float): Controls randomness in model output. Lower values
                               (e.g., 0.3) make output more deterministic, which is
                               preferred for OCR tasks. Range: 0.0 to 1.0
        
        Returns:
            str: Extracted text from the image, or error message if extraction fails
                The text is returned exactly as it appears in the image, enclosed
                in double quotes per the prompt instructions
        
        Note:
            The method will retry up to 20 times with exponential backoff
            when encountering rate limits, making it robust for batch processing
        """
        max_retries = 20  # Maximum number of retry attempts
        retry_delay = 1   # Initial delay in seconds between retries

        for attempt in range(max_retries):
            try:
                # Make API call to extract text from image
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    # Prompt engineering for accurate OCR extraction
                                    # This prompt ensures clean text extraction without additional commentary
                                    "text": ("Extract only the text from the image and "
                                             "enclose the entire content in double quotes. "
                                             "Do not add any extra information or descriptions. "
                                             "Provide the text exactly as it appears in the image.")
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        # Image is passed as base64-encoded data URL
                                        "url": f"data:image/jpeg;base64,{image}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=temperature,  # Low temperature for consistent OCR
                )
                # Return the extracted text content
                return response.choices[0].message.content

            except Exception as e:
                # Handle rate limiting with exponential backoff
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    # Calculate exponential backoff delay (1, 2, 4, 8, 16... seconds)
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Log other errors and return error message
                    logger.error(f"Error calling API: {str(e)}")
                    return f"ERROR: Failed to get response from API: {str(e)}"

        # If all retries exhausted, return error
        return "ERROR: Max retry attempts reached"


def process_images(root_dir, api_client):
    """
    Process all images in the directory structure and extract text using OCR.
    
    This function traverses a specific directory structure organized by document
    distortion levels (easy/medium/hard), processes PNG images, and saves the
    extracted text. It's designed to handle large batches of documents efficiently.
    
    Directory Structure Expected:
        root_dir/
        ├── easy/           # Low distortion documents
        ├── medium/         # Medium distortion documents
        └── hard/           # High distortion documents
            └── {doc_folder}/
                ├── images/     # Input PNG files
                └── llama_txt/  # Output text files (created by this function)
    
    Args:
        root_dir (str): Root directory containing the document hierarchy
        api_client (SambanovaAPIClient): Configured API client for OCR extraction
    
    Features:
        - Automatically creates output directories (llama_txt) if they don't exist
        - Skips already processed files (unless they contain errors)
        - Reprocesses files with ERROR messages
        - Maintains the same filename structure (image.png -> image.txt)
        - Provides detailed logging of processing progress
    
    Note:
        The function expects document folders to follow a specific naming pattern
        like '125167-download.pdf-0.68' where the last number represents the
        distortion score used for categorization
    """
    
    # Iterate through top-level directories (easy, medium, hard)
    for top_level in os.listdir(root_dir):
        top_level_path = os.path.join(root_dir, top_level)
        
        # Skip if not a directory
        if not os.path.isdir(top_level_path):
            continue

        logger.info(f"Processing top-level directory: {top_level_path}")

        # Process each document folder within the distortion level
        for subdir in os.listdir(top_level_path):
            subdir_path = os.path.join(top_level_path, subdir)
            
            # Skip if not a directory
            if not os.path.isdir(subdir_path):
                continue

            # Define input and output directories
            images_dir = os.path.join(subdir_path, "images")
            llama_txt_dir = os.path.join(subdir_path, "llama_txt")
            
            # Check if images directory exists
            if not os.path.exists(images_dir):
                logger.info(f"No images directory found at {images_dir}, skipping.")
                continue

            # Create output directory if it doesn't exist
            os.makedirs(llama_txt_dir, exist_ok=True)
            
            # Get all PNG files in the images directory
            image_files = [
                f for f in os.listdir(images_dir) 
                if f.lower().endswith(".png")
            ]
            # Sort for consistent processing order
            image_files.sort()

            # Process each image file
            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                
                # Generate corresponding text file path
                txt_filename = os.path.splitext(image_file)[0] + ".txt"
                txt_path = os.path.join(llama_txt_dir, txt_filename)
                
                # Check if file needs processing
                need_to_process = True
                if os.path.exists(txt_path):
                    # Read existing file to check if it's an error
                    with open(txt_path, "r", encoding="utf-8") as existing_txt:
                        content = existing_txt.read()
                        if not content.startswith("ERROR"):
                            # Valid text exists, skip processing
                            logger.info(f"    Skipping {image_path}; valid text file found.")
                            need_to_process = False
                        else:
                            # Error in previous attempt, retry
                            logger.info(f"    Reprocessing {image_path} because it contains an error.")
                
                if not need_to_process:
                    continue

                logger.info(f"  Processing image: {image_path}")

                # Read and encode image as base64
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                base64_string = base64.b64encode(image_bytes).decode("utf-8")

                try:
                    # Extract text using API
                    extracted_text = api_client.predict(
                        image=base64_string, 
                        temperature=0.3  # Low temperature for consistent OCR
                    )
                except Exception as e:
                    logger.error(f"    Error calling predict on {image_path}: {e}")
                    continue

                # Save extracted text to file
                txt_filename = os.path.splitext(image_file)[0] + ".txt"
                txt_path = os.path.join(llama_txt_dir, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as out_f:
                    out_f.write(extracted_text)

                logger.info(f"    Wrote extracted text to {txt_path}")


def main():
    """
    Main entry point for the OCR extraction pipeline.
    
    This function sets up the API client and initiates the batch processing
    of document images. It's designed to be run as a standalone script or
    imported as a module.
    
    Configuration:
        - API Key: Set your SambaNova API key here or as environment variable
        - Root Directory: Path to the document dataset organized by distortion level
        - Model: Uses Llama-3.2-90B-Vision-Instruct for high-quality OCR
    
    Usage:
        python samba_llama.py
        
    Or with environment variable:
        export OPENAI_API_KEY="your-api-key"
        python samba_llama.py
    """
    # Configure API access
    api_key = "your-api-key-here"  # Replace with your actual API key
    
    # Initialize API client with LLaMA Vision model
    api_client = SambanovaAPIClient(
        model_name="Llama-3.2-90B-Vision-Instruct", 
        api_key=api_key
    )

    # Set root directory for document processing
    # This should point to your dataset organized by distortion levels
    root_dir = "/path/to/your/data"  # Update this path
    
    # Start processing all images in the directory structure
    process_images(root_dir, api_client)
    
    logger.info("OCR extraction complete!")


if __name__ == "__main__":
    main()
