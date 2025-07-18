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

class SplitBamsnapSVValidator:
    """
    A class to validate structural variants (deletions) from separate sample/control 
    bamsnap IGV plots using OpenAI's GPT-4 Vision API.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the validator with OpenAI API key and model.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # SV validation rules for single panels
        self.panel_analysis_rules = """Panel Analysis Rules for Deletions:

Analyze this single bamsnap panel for deletion signals:

Key Features to Identify:
* Coverage drops: Significant reduction in coverage depth in the candidate region
* Red bars: Reads with insert size > 500bp (evidence of potential deletion)
* Soft clipped parts: Colored stubs at read ends indicating unaligned portions
* Gray alignment bars: Normal aligned portions of reads

Look for:
1. Clear coverage reduction in the candidate deletion region
2. Presence of red bars flanking or within the deletion region
3. Soft clipping at potential breakpoints
4. Overall disruption of normal read alignment patterns

Classify the panel as:
- HAS_DELETION: Clear deletion signals present
- NO_DELETION: No convincing deletion signals
- UNCLEAR: Ambiguous or borderline evidence"""

        # Comparison rules
        self.comparison_rules = """CRITICAL COMPARISON RULE:

A true deletion event requires:
- SAMPLE panel shows deletion signals (HAS_DELETION)
- CONTROL panel shows no deletion signals (NO_DELETION)

Classification Logic:
- True (1): SAMPLE = HAS_DELETION AND CONTROL = NO_DELETION
- False (0): Any other combination

If both panels show deletion signals OR both show no deletion signals, classify as FALSE."""

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
        Load training image pairs and their corresponding JSON annotations.
        
        Args:
            training_path: Path to training directory with sample/control pairs
            
        Returns:
            List of training examples with sample/control images and annotations
        """
        training_data = []
        training_dir = Path(training_path)
        
        if not training_dir.exists():
            self.logger.warning(f"Training directory not found: {training_dir}")
            return training_data

        # Look for JSON files first, then find corresponding image pairs
        for json_file in training_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                base_name = json_file.stem
                
                # Look for sample and control images
                sample_patterns = [f"{base_name}_sample.*", f"{base_name}.sample.*", f"sample_{base_name}.*"]
                control_patterns = [f"{base_name}_control.*", f"{base_name}.control.*", f"control_{base_name}.*"]
                
                sample_file = None
                control_file = None
                
                # Find sample file
                for pattern in sample_patterns:
                    matches = list(training_dir.glob(pattern))
                    if matches:
                        sample_file = matches[0]
                        break
                
                # Find control file  
                for pattern in control_patterns:
                    matches = list(training_dir.glob(pattern))
                    if matches:
                        control_file = matches[0]
                        break
                
                if sample_file and control_file:
                    training_data.append({
                        'sample_path': str(sample_file),
                        'control_path': str(control_file),
                        'annotation': annotation,
                        'encoded_sample': self.encode_image(str(sample_file)),
                        'encoded_control': self.encode_image(str(control_file))
                    })
                else:
                    self.logger.warning(f"Could not find sample/control pair for {json_file}")
                    
            except Exception as e:
                self.logger.warning(f"Error loading training data for {json_file}: {e}")

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
        
        for i, example in enumerate(training_data[:3]):  # Limit to 3 examples to avoid token limits
            annotation = example['annotation']
            label = annotation.get('label', 'unknown')
            sample_analysis = annotation.get('sample_analysis', 'No sample analysis provided')
            control_analysis = annotation.get('control_analysis', 'No control analysis provided')
            final_reasoning = annotation.get('final_reasoning', 'No reasoning provided')
            
            examples_prompt += f"""Example {i+1}:
Sample Analysis: {sample_analysis}
Control Analysis: {control_analysis}
Final Classification: {label}
Final Reasoning: {final_reasoning}

"""
        
        return examples_prompt

    def analyze_single_panel(self, image_path: str, panel_type: str, training_data: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze a single panel (sample or control) for deletion signals.
        
        Args:
            image_path: Path to the panel image
            panel_type: "sample" or "control"
            training_data: Optional training data for context
            
        Returns:
            Dictionary with panel analysis results
        """
        try:
            encoded_image = self.encode_image(image_path)
            
            prompt = f"""You are analyzing a {panel_type.upper()} panel from a bamsnap IGV plot.

{self.panel_analysis_rules}

Provide your analysis in JSON format:
{{
    "panel_classification": "HAS_DELETION/NO_DELETION/UNCLEAR",
    "confidence": "high/medium/low",
    "observed_features": ["list of features you observe"],
    "coverage_assessment": "description of coverage pattern",
    "reasoning": "detailed explanation of your assessment"
}}

Focus specifically on this {panel_type} panel and identify any deletion signals present."""

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )

            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "panel_classification": "UNCLEAR",
                        "confidence": "low",
                        "observed_features": [],
                        "coverage_assessment": "",
                        "reasoning": response_text
                    }
            except json.JSONDecodeError:
                result = {
                    "panel_classification": "UNCLEAR",
                    "confidence": "low", 
                    "observed_features": [],
                    "coverage_assessment": "",
                    "reasoning": response_text
                }
            
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {panel_type} panel {image_path}: {e}")
            return {
                "panel_classification": "UNCLEAR",
                "confidence": "low",
                "observed_features": [],
                "coverage_assessment": "",
                "reasoning": f"Error during analysis: {str(e)}"
            }

    def compare_panels_and_classify(self, sample_result: Dict, control_result: Dict, 
                                   sample_path: str, control_path: str, 
                                   training_data: Optional[List[Dict]] = None) -> Dict:
        """
        Compare sample and control panel results to make final classification.
        
        Args:
            sample_result: Analysis result from sample panel
            control_result: Analysis result from control panel
            sample_path: Path to sample image
            control_path: Path to control image
            training_data: Optional training data for context
            
        Returns:
            Final classification result
        """
        sample_classification = sample_result.get("panel_classification", "UNCLEAR")
        control_classification = control_result.get("panel_classification", "UNCLEAR")
        
        # Apply comparison logic
        if sample_classification == "HAS_DELETION" and control_classification == "NO_DELETION":
            final_classification = 1
            confidence = "high"
            reasoning = "Sample shows deletion signals while control does not - TRUE DELETION"
        elif sample_classification == "NO_DELETION" and control_classification == "NO_DELETION":
            final_classification = 0
            confidence = "high"
            reasoning = "Neither sample nor control show deletion signals - FALSE POSITIVE"
        elif sample_classification == "HAS_DELETION" and control_classification == "HAS_DELETION":
            final_classification = 0
            confidence = "high"
            reasoning = "Both sample and control show deletion signals - likely reference artifact - FALSE POSITIVE"
        elif sample_classification == "NO_DELETION" and control_classification == "HAS_DELETION":
            final_classification = 0
            confidence = "medium"
            reasoning = "Control shows deletion signals but sample does not - unusual pattern - FALSE POSITIVE"
        else:
            # Handle UNCLEAR cases
            final_classification = 0
            confidence = "low"
            reasoning = f"Unclear evidence in panels (Sample: {sample_classification}, Control: {control_classification}) - defaulting to FALSE POSITIVE"

        return {
            "classification": final_classification,
            "confidence": confidence,
            "reasoning": reasoning,
            "sample_analysis": sample_result,
            "control_analysis": control_result,
            "sample_path": sample_path,
            "control_path": control_path,
            "comparison_logic": f"Sample: {sample_classification}, Control: {control_classification}"
        }

    def _validate_single_deletion_pair(self, sample_path: str, control_path: str, 
                                      training_data: Optional[List[Dict]] = None) -> Dict:
        """
        Validate a deletion from a sample/control image pair.
        
        Args:
            sample_path: Path to sample panel image
            control_path: Path to control panel image
            training_data: Optional training data for few-shot learning
            
        Returns:
            Dictionary containing classification result and detailed analysis
        """
        try:
            # Analyze sample panel
            self.logger.info(f"Analyzing sample panel: {Path(sample_path).name}")
            sample_result = self.analyze_single_panel(sample_path, "sample", training_data)
            
            # Analyze control panel
            self.logger.info(f"Analyzing control panel: {Path(control_path).name}")
            control_result = self.analyze_single_panel(control_path, "control", training_data)
            
            # Compare and classify
            final_result = self.compare_panels_and_classify(
                sample_result, control_result, sample_path, control_path, training_data
            )
            
            return final_result

        except Exception as e:
            self.logger.error(f"Error validating deletion pair {sample_path}, {control_path}: {e}")
            return {
                "classification": 0,
                "confidence": "low",
                "reasoning": f"Error during analysis: {str(e)}",
                "sample_analysis": {},
                "control_analysis": {},
                "sample_path": sample_path,
                "control_path": control_path,
                "comparison_logic": "Error"
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
                    'sample_path',
                    'control_path', 
                    'classification',
                    'confidence',
                    'reasoning',
                    'sample_classification',
                    'control_classification'
                ])
                
                # Write data rows
                for result in results:
                    sample_analysis = result.get('sample_analysis', {})
                    control_analysis = result.get('control_analysis', {})
                    
                    writer.writerow([
                        result.get('sample_path', ''),
                        result.get('control_path', ''),
                        result.get('classification', 0),
                        result.get('confidence', 'low'),
                        result.get('reasoning', '').replace('\n', ' ').replace('\r', ' '),
                        sample_analysis.get('panel_classification', 'UNCLEAR'),
                        control_analysis.get('panel_classification', 'UNCLEAR')
                    ])
                    
            self.logger.info(f"CSV results saved to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV file {csv_file}: {e}")
            raise

    def validate_deletions(self, image_dir: str, 
                          output_file: str = None,
                          csv_output_file: str = None,
                          training_dir: str = None,
                          parallel_processing: bool = False,
                          max_workers: int = 3) -> List[Dict]:
        """
        Validate multiple deletion candidates from sample/control image pairs.
        
        Args:
            image_dir: Directory containing sample/control image pairs
            output_file: Optional JSON output file
            csv_output_file: Optional CSV output file
            training_dir: Optional training data directory
            parallel_processing: Whether to use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            List of validation results
        """
        # Load training data if provided
        training_data = []
        if training_dir:
            training_data = self.load_training_data(training_dir)

        results = []
        image_path = Path(image_dir)
        
        # Find sample/control pairs
        pairs = self.find_image_pairs(image_path)
        self.logger.info(f"Found {len(pairs)} sample/control pairs to process")
        
        if parallel_processing:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_pair(sample_path, control_path):
                return self._validate_single_deletion_pair(
                    str(sample_path), str(control_path), training_data
                )
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(process_pair, sample, control): (sample, control) 
                    for sample, control in pairs
                }
                
                for future in as_completed(future_to_pair):
                    sample, control = future_to_pair[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Completed pair: Classification={result['classification']}")
                    except Exception as e:
                        self.logger.error(f"Error processing pair {sample.name}, {control.name}: {e}")
        else:
            # Sequential processing
            for sample_path, control_path in pairs:
                self.logger.info(f"Processing pair: {sample_path.name}, {control_path.name}")
                
                result = self._validate_single_deletion_pair(
                    str(sample_path), str(control_path), training_data
                )
                
                results.append(result)
                self.logger.info(f"Completed pair: Classification={result['classification']}, Confidence={result['confidence']}")

        # Save results
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
Total pairs processed: {len(results)}
True positives (real deletions): {true_positives}
False positives: {false_positives}
High confidence predictions: {high_confidence}""")
        
        return results

    def find_image_pairs(self, image_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Find sample/control image pairs in the directory.
        
        Args:
            image_dir: Directory to search for image pairs
            
        Returns:
            List of (sample_path, control_path) tuples
        """
        pairs = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        # Look for files with sample/control naming patterns
        for img_file in image_dir.glob("*"):
            if img_file.suffix.lower() in image_extensions:
                name = img_file.stem
                
                # Try different naming patterns
                if "sample" in name.lower():
                    # Look for corresponding control
                    control_name = name.lower().replace("sample", "control")
                    control_candidates = [
                        image_dir / f"{control_name}{img_file.suffix}",
                        image_dir / f"{name.replace('sample', 'control')}{img_file.suffix}",
                        image_dir / f"{name.replace('Sample', 'Control')}{img_file.suffix}"
                    ]
                    
                    for control_path in control_candidates:
                        if control_path.exists():
                            pairs.append((img_file, control_path))
                            break
        
        return pairs

    def generate_report(self, results: List[Dict], report_file: str = None) -> str:
        """
        Generate a summary statistics report from validation results.
        
        Args:
            results: List of validation results
            report_file: Optional file to save the report
            
        Returns:
            Report as string
        """
        if not results:
            return "No results to generate report from."
        
        total = len(results)
        true_positives = sum(1 for r in results if r['classification'] == 1)
        false_positives = sum(1 for r in results if r['classification'] == 0)
        
        confidence_counts = {
            'high': sum(1 for r in results if r['confidence'] == 'high'),
            'medium': sum(1 for r in results if r['confidence'] == 'medium'),
            'low': sum(1 for r in results if r['confidence'] == 'low')
        }
        
        report = f"""# Split Panel Bamsnap Deletion SV Validation Report

## Summary Statistics
- **Total Image Pairs Processed**: {total}
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

def main():
    """
    Main function for processing deletion SVs from sample/control image pairs
    """
    parser = argparse.ArgumentParser(description="Validate deletion structural variants from sample/control bamsnap image pairs")
    parser.add_argument("image", help="Directory containing sample/control image pairs")
    parser.add_argument("-t", "--train", default="trainingData/deletion", 
                       help="Training data directory (default: trainingData/deletion)")
    parser.add_argument("-o", "--output-dir", default=None, 
                       help="Output directory for results (default: auto-generated with timestamp)")
    parser.add_argument("--output-prefix", default="split_deletion_results", 
                       help="Prefix for output files (default: split_deletion_results)")
    parser.add_argument("-m", "--model", choices=["gpt-4o", "gpt-4o-mini"], default="gpt-4o",
                       help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("-p", "--parallel", action="store_true", 
                       help="Enable parallel processing (use with caution for API limits)")
    parser.add_argument("-w", "--workers", type=int, default=2, 
                       help="Number of parallel workers (default: 2)")
    parser.add_argument("--no-training", action="store_true", 
                       help="Skip loading training data")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize validator
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return 1
    
    validator = SplitBamsnapSVValidator(api_key, model=args.model)
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"split_results_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file paths
    output_file = output_dir / f"{args.output_prefix}.json"
    csv_file = output_dir / f"{args.output_prefix}.csv"
    report_file = output_dir / f"{args.output_prefix}_summary.md"
    
    print(f"=== Processing Split Panel Deletion SVs ===")
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
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    if not results:
        print("No image pairs were processed. Check your image directory and naming convention.")
        return 1
    
    print(f"\nProcessed {len(results)} image pairs")
    
    # Generate summary report
    print(f"\n=== Generating Summary Report ===")
    report = validator.generate_report(results, str(report_file))
    print("Summary report generated!")
    
    # Print quick stats
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