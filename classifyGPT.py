import openai
import base64
import json
import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime

class BamsnapSVValidator:
    """
    A class to validate structural variants (deletions) from bamsnap IGV plots
    using OpenAI's GPT-4 Vision API.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the validator with OpenAI API key and model.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o for vision capabilities)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # SV validation rules for deletions
        self.deletion_rules = """SV Validation Rules for Deletions:
        
Vertical dashed lines mark the candidate deletion breakpoints. The top panel shows the sample and the bottom panel the control. Within each panel, aligned reads occupy the upper track and coverage is plotted below. Deletions are labeled as a true event (1) if signals are present in the sample panel but absent in the control.

How reads appear in a Bamsnap plot:
* Red bars in or flanking the deletion region: reads with insert size > 500bp, evidence of potential deletion
* Soft clipped parts (little colored stubs at the end of a read's grey alignment block): the part of the read that does not align to the reference genome, but the unaligned portion is still retained in the read data; this is common in structural variant detection and can indicate breakpoints or mismatches
* Gray alignment bar: the portion of each read that actually lines up to the reference sequence.

Priorities to decide a deletion event:
1. A drop in coverage in the region or drop near breakpoints for longer events
2. Soft clipped parts near breakpoints
3. Presence of red bars

CRITICAL RULE: If coverage drops occur in both the sample and control, classify the event as FALSE regardless of the difference in the drop's magnitude (i.e., still False if the drop in sample is greater than that in control).

Classification:
- True (1): Deletion signals present in sample but absent in control
- False (0): No deletion signals, or deletion signals present in both sample and control"""
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def load_training_data(self, training_path: str) -> List[Dict]:
        """
        Load training images and their corresponding JSON annotations from the specified path.
        
        Args:
            training_path: Full path to training directory (e.g., "trainingData/deletion")
            
        Returns:
            List of training examples with images and annotations
        """
        training_data = []
        training_dir = Path(training_path)
        
        if not training_dir.exists():
            self.logger.warning(f"Training directory not found: {training_dir}")
            return training_data
        
        # Look for image files in the specified directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        for img_file in training_dir.glob("*"):
            if img_file.suffix.lower() in image_extensions:
                # Look for corresponding JSON file
                json_file = img_file.with_suffix('.json')
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            annotation = json.load(f)
                        
                        training_data.append({
                            'image_path': str(img_file),
                            'annotation': annotation,
                            'encoded_image': self.encode_image(str(img_file))
                        })
                    except Exception as e:
                        self.logger.warning(f"Error loading training data for {img_file}: {e}")
        
        self.logger.info(f"Loaded {len(training_data)} training examples from {training_dir}")
        return training_data
    
    def create_training_examples_prompt(self, training_data: List[Dict]) -> str:
        """
        Create a prompt section with training examples.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Formatted prompt with training examples
        """
        examples_prompt = "\n\nTRAINING EXAMPLES:\n"
        
        for i, example in enumerate(training_data[:5]):  # Limit to 5 examples to avoid token limits
            annotation = example['annotation']
            label = annotation.get('label', 'unknown')
            comment = annotation.get('comment', 'No comment provided')
            
            examples_prompt += f"""Example {i+1}:
Classification: {label}
Reasoning: {comment}

"""
        
        return examples_prompt
    
    def _validate_single_deletion(self, image_path: str, training_data: Optional[List[Dict]] = None) -> Dict:
        """
        Internal method to validate a single deletion SV from a bamsnap plot.
        
        Args:
            image_path: Path to the bamsnap plot image
            training_data: Optional training data for few-shot learning
            
        Returns:
            Dictionary containing classification result and reasoning
        """
        try:
            # Encode the image
            encoded_image = self.encode_image(image_path)
            
            # Create base prompt
            prompt = f"""You are an expert in analyzing bamsnap IGV plots for structural variant validation.

{self.deletion_rules}

Please analyze the provided bamsnap plot and classify whether the deletion is:
- True (1): Real deletion event
- False (0): False positive

Provide your analysis in the following JSON format:
{{
    "classification": 0 or 1,
    "confidence": "high/medium/low",
    "reasoning": "Detailed explanation of your decision based on the validation rules",
    "key_features": ["list of key features observed in the plot"],
    "sample_panel_observations": "What you observe in the sample panel",
    "control_panel_observations": "What you observe in the control panel"
}}

Focus on:
1. Coverage drops in sample vs control
2. Soft clipping near breakpoints
3. Presence of red bars (large insert size reads)
4. Overall pattern differences between sample and control panels"""
            
            # Add training examples if provided
            if training_data:
                training_prompt = self.create_training_examples_prompt(training_data)
                prompt += training_prompt
            
            # Prepare messages for API call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent results
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback parsing
                    result = {
                        "classification": 0,
                        "confidence": "low",
                        "reasoning": response_text,
                        "key_features": [],
                        "sample_panel_observations": "",
                        "control_panel_observations": ""
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "classification": 0,
                    "confidence": "low",
                    "reasoning": response_text,
                    "key_features": [],
                    "sample_panel_observations": "",
                    "control_panel_observations": ""
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating deletion for {image_path}: {e}")
            return {
                "classification": 0,
                "confidence": "low",
                "reasoning": f"Error during analysis: {str(e)}",
                "key_features": [],
                "sample_panel_observations": "",
                "control_panel_observations": ""
            }
    
    def save_results_csv(self, results: List[Dict], csv_file: str) -> None:
        """
        Save validation results to CSV format.
        
        Args:
            results: List of validation results
            csv_file: Path to output CSV file
        """
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'file_path', 
                    'classification', 
                    'reasoning'
                ])
                
                # Write data rows
                for result in results:
                    writer.writerow([
                        result.get('image_path', ''),
                        result.get('classification', 0),
                        result.get('reasoning', '').replace('\n', ' ').replace('\r', ' ')  # Clean line breaks
                    ])
                    
            self.logger.info(f"CSV results saved to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV file {csv_file}: {e}")
            raise
    
    def validate_deletions(self, image_dir: str, 
                          output_file: str = None,
                          csv_output_file: str = None,
                          training_dir: str = None, 
                          file_patterns: List[str] = None,
                          parallel_processing: bool = False,
                          max_workers: int = 3) -> List[Dict]:
        """
        Validate multiple bamsnap plots for deletion SVs in a directory.
        
        Args:
            image_dir: Directory containing bamsnap plot images
            output_file: Optional output file to save results (JSON format)
            csv_output_file: Optional CSV file to save simplified results (file_path, classification, reasoning)
            training_dir: Optional directory with training data
            file_patterns: Optional list of file patterns to match (e.g., ['*deletion*', '*del*'])
            parallel_processing: Whether to process images in parallel (be careful with API limits)
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of validation results
        """
        # Load training data if provided
        training_data = []
        if training_dir:
            training_data = self.load_training_data(training_dir)
            if training_data:
                self.logger.info(f"Loaded {len(training_data)} training examples")
            else:
                self.logger.warning("No training data found")
        
        results = []
        image_path = Path(image_dir)
        
        # Find all images to process
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        images_to_process = []
        
        for img_file in image_path.glob("*"):
            if img_file.suffix.lower() in image_extensions:
                # Check file patterns if specified
                if file_patterns:
                    if any(img_file.match(pattern) for pattern in file_patterns):
                        images_to_process.append(img_file)
                else:
                    images_to_process.append(img_file)
        
        self.logger.info(f"Found {len(images_to_process)} images to process")
        
        if parallel_processing:
            # Parallel processing (use with caution due to API rate limits)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_image(img_file):
                self.logger.info(f"Processing {img_file.name}")
                result = self._validate_single_deletion(
                    str(img_file), 
                    training_data if training_data else None
                )
                result['image_file'] = img_file.name
                result['image_path'] = str(img_file)
                return result
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_image = {executor.submit(process_image, img): img for img in images_to_process}
                
                for future in as_completed(future_to_image):
                    img = future_to_image[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Completed {img.name}: {result['classification']}")
                    except Exception as e:
                        self.logger.error(f"Error processing {img.name}: {e}")
                        results.append({
                            'image_file': img.name,
                            'image_path': str(img),
                            'classification': 0,
                            'confidence': 'low',
                            'reasoning': f'Processing error: {str(e)}',
                            'key_features': [],
                            'sample_panel_observations': '',
                            'control_panel_observations': ''
                        })
        else:
            # Sequential processing (recommended for API stability)
            for img_file in images_to_process:
                self.logger.info(f"Processing {img_file.name}")
                
                result = self._validate_single_deletion(
                    str(img_file), 
                    training_data if training_data else None
                )
                
                result['image_file'] = img_file.name
                result['image_path'] = str(img_file)
                results.append(result)
                
                self.logger.info(f"Completed {img_file.name}: Classification={result['classification']}, Confidence={result['confidence']}")
        
        # Save results if output files specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"JSON results saved to {output_file}")
        
        if csv_output_file:
            self.save_results_csv(results, csv_output_file)
        
        # Print summary
        true_positives = sum(1 for r in results if r['classification'] == 1)
        false_positives = sum(1 for r in results if r['classification'] == 0)
        high_confidence = sum(1 for r in results if r['confidence'] == 'high')
        
        self.logger.info(f"""BATCH PROCESSING SUMMARY:
Total images processed: {len(results)}
True positives (real deletions): {true_positives}
False positives: {false_positives}
High confidence predictions: {high_confidence}""")
        
        return results
    
    def generate_report(self, results: List[Dict], report_file: str = None) -> str:
        """
        Generate a summary statistics report from batch validation results.
        
        Args:
            results: List of validation results from validate_deletions()
            report_file: Optional file to save the report
            
        Returns:
            Report as string
        """
        if not results:
            return "No results to generate report from."
        
        # Calculate statistics
        total = len(results)
        true_positives = sum(1 for r in results if r['classification'] == 1)
        false_positives = sum(1 for r in results if r['classification'] == 0)
        
        confidence_counts = {
            'high': sum(1 for r in results if r['confidence'] == 'high'),
            'medium': sum(1 for r in results if r['confidence'] == 'medium'),
            'low': sum(1 for r in results if r['confidence'] == 'low')
        }
        
        # Generate summary report
        report = f"""# Bamsnap Deletion SV Validation Report

## Summary Statistics
- **Total Images Processed**: {total}
- **True Positives (Real Deletions)**: {true_positives} ({true_positives/total*100:.1f}%)
- **False Positives**: {false_positives} ({false_positives/total*100:.1f}%)

## Confidence Distribution
- **High Confidence**: {confidence_counts['high']} ({confidence_counts['high']/total*100:.1f}%)
- **Medium Confidence**: {confidence_counts['medium']} ({confidence_counts['medium']/total*100:.1f}%)
- **Low Confidence**: {confidence_counts['low']} ({confidence_counts['low']/total*100:.1f}%)"""
        
        if report_file:
            with open(report_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {report_file}")
        
        return report

# Example usage and testing
def main():
    """
    Main function for processing deletion SVs with command line arguments
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Validate deletion structural variants from bamsnap IGV plots")
    parser.add_argument("image", help="Directory containing bamsnap plot images to analyze")
    parser.add_argument("-t", "--train", default="trainingData/deletion", 
                       help="Training data directory (default: trainingData/deletion)")
    parser.add_argument("-o", "--output-dir", default=None, 
                       help="Output directory for results (default: auto-generated with timestamp)")
    parser.add_argument("--output-prefix", default="deletion_results", 
                       help="Prefix for output files (default: deletion_results)")
    parser.add_argument("-m", "--model", choices=["gpt-4o", "gpt-4o-mini"], default="gpt-4o",
                       help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("-p", "--parallel", action="store_true", 
                       help="Enable parallel processing (use with caution for API limits)")
    parser.add_argument("-w", "--workers", type=int, default=2, 
                       help="Number of parallel workers (default: 2)")
    parser.add_argument("--no-training", action="store_true", 
                       help="Skip loading training data")
    parser.add_argument("--patterns", nargs="+", 
                       help="File patterns to match (e.g., --patterns '*deletion*' '*del*')")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize validator
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-your-key-here'")
        return 1
    
    validator = BamsnapSVValidator(api_key, model=args.model)
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file paths
    output_file = output_dir / f"{args.output_prefix}.json"
    csv_file = output_dir / f"{args.output_prefix}.csv"
    report_file = output_dir / f"{args.output_prefix}_summary.md"
    
    print(f"=== Processing Deletion SVs ===")
    print(f"Model: {args.model}")
    print(f"Image directory: {args.image}")
    print(f"Training directory: {args.train if not args.no_training else 'None (disabled)'}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel processing: {args.parallel}")
    
    # Process deletions
    results = validator.validate_deletions(
        image_dir=args.image,
        training_dir=args.train if not args.no_training else None,
        output_file=str(output_file),
        csv_output_file=str(csv_file),
        file_patterns=args.patterns,
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    if not results:
        print("No images were processed. Check your image directory path.")
        return 1
    
    print(f"\nProcessed {len(results)} deletion images")
    
    # Generate summary report
    print(f"\n=== Generating Summary Report ===")
    report = validator.generate_report(results, str(report_file))
    print("Summary report generated!")
    
    # Print quick stats to console
    true_positives = sum(1 for r in results if r['classification'] == 1)
    false_positives = sum(1 for r in results if r['classification'] == 0)
    high_confidence = sum(1 for r in results if r['confidence'] == 'high')
    
    print(f"\n=== Quick Stats ===")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"High confidence: {high_confidence}/{len(results)} ({high_confidence/len(results)*100:.1f}%)")
    
    print(f"\n=== Output Files Created in: {output_dir} ===")
    print(f"JSON results: {output_file.name}")
    print(f"CSV results: {csv_file.name}")
    print(f"Summary report: {report_file.name}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)