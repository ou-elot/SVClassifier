import openai
import base64
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse

# Global API key configuration
OPENAI_API_KEY = "ilikehotdogs" #enter actual api key here from openai

@dataclass
class SVResult:
    """Result from SV analysis"""
    is_real_sv: bool
    sv_type: str
    confidence: float
    explanation: str
    supporting_evidence: List[str]
    visual_features: Dict[str, str]

class SVTypeSpecificAnalyzer:
    """
    SV Type-Specific Analyzer that uses different training data and criteria
    for each structural variant type
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the SV analyzer
        
        Args:
            model: GPT model to use
        """
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.sv_type_training = {}  # Store training data by SV type
        self.sv_type_criteria = self._get_sv_type_criteria()
        
    def _get_sv_type_criteria(self) -> Dict[str, Dict]:
        """
        Define specific visual criteria for each SV type
        """
        return {
            "DEL": {
                "primary_features": [
                    "Sharp coverage drop in deleted region",
                    "Outward-facing discordant read pairs (-->  <--)",
                    "Split reads at breakpoints with soft-clipped sequences",
                    "Well-defined breakpoint boundaries"
                ],
                "coverage_pattern": "Significant decrease (50%+ drop)",
                "read_pair_orientation": "Outward-facing at breakpoints",
                "false_positive_indicators": [
                    "Gradual coverage decline",
                    "Low mapping quality reads",
                    "Repetitive sequence context",
                    "Scattered discordant pairs"
                ]
            },
            "DUP": {
                "primary_features": [
                    "Coverage increase in duplicated region (1.5x+ baseline)",
                    "Inward-facing discordant read pairs (<--  -->)",
                    "Junction reads at duplication boundaries",
                    "Consistent increased depth across region"
                ],
                "coverage_pattern": "Significant increase (150%+ of baseline)",
                "read_pair_orientation": "Inward-facing at breakpoints",
                "false_positive_indicators": [
                    "Irregular coverage pattern",
                    "Mixed read orientations",
                    "GC bias artifacts",
                    "Segmental duplication regions"
                ]
            },
            "INV": {
                "primary_features": [
                    "Read pairs with same orientation (-->  --> or <--  <--)",
                    "Split reads at both breakpoints",
                    "No coverage change (balanced rearrangement)",
                    "Clear inversion junction signatures"
                ],
                "coverage_pattern": "No significant change",
                "read_pair_orientation": "Same orientation within inversion",
                "false_positive_indicators": [
                    "Coverage fluctuations",
                    "Mixed orientations",
                    "Poor breakpoint definition",
                    "Low complexity regions"
                ]
            },
            "TRA": {
                "primary_features": [
                    "Discordant pairs mapping to different chromosomes",
                    "Split reads showing inter-chromosomal junctions",
                    "High mapping quality for supporting reads",
                    "Clustered breakpoint evidence"
                ],
                "coverage_pattern": "No change for balanced translocations",
                "read_pair_orientation": "Cross-chromosomal mapping",
                "false_positive_indicators": [
                    "Low mapping quality",
                    "Repetitive elements",
                    "Scattered evidence",
                    "Common fragile sites"
                ]
            },
            "INS": {
                "primary_features": [
                    "Split reads with inserted sequences",
                    "No coverage change at insertion site",
                    "Soft-clipped reads containing insert sequence",
                    "Discordant pairs if large insertion"
                ],
                "coverage_pattern": "No significant change",
                "read_pair_orientation": "Normal for small, discordant for large",
                "false_positive_indicators": [
                    "Poor sequence quality",
                    "Homopolymer regions",
                    "Sequencing artifacts",
                    "Low complexity inserts"
                ]
            }
        }
    
    def load_sv_type_training(self, training_dir: str) -> None:
        """
        Load training data organized by SV type, including both images and annotations
        
        Args:
            training_dir: Directory containing SV-type-specific subdirectories (del, dup, inv, tra, ins)
        """
        training_path = Path(training_dir)
        
        if not training_path.exists():
            print(f"Warning: Training directory {training_dir} not found")
            return
        
        # Look for SV-type-specific subdirectories matching your structure
        sv_type_mapping = {
            "del": "DEL",
            "dup": "DUP", 
            "inv": "INV",
            "tra": "TRA",
            "ins": "INS"
        }
        
        for folder_name, sv_type in sv_type_mapping.items():
            sv_type_dir = training_path / folder_name
            if sv_type_dir.exists() and sv_type_dir.is_dir():
                self.sv_type_training[sv_type] = []
                
                # Load all JSON files directly from the SV type folder
                json_files = list(sv_type_dir.glob("*.json"))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            training_data = json.load(f)
                        
                        # Look for corresponding image file
                        image_filename = training_data.get("filename", "")
                        if image_filename:
                            # Try to find the image in the same directory
                            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
                            image_found = False
                            
                            for ext in image_extensions:
                                # Try exact filename first
                                image_path = sv_type_dir / image_filename
                                if image_path.exists():
                                    training_data["image_path"] = str(image_path)
                                    image_found = True
                                    break
                                
                                # Try filename without extension + this extension
                                base_name = Path(image_filename).stem
                                image_path = sv_type_dir / f"{base_name}{ext}"
                                if image_path.exists():
                                    training_data["image_path"] = str(image_path)
                                    image_found = True
                                    break
                            
                            if image_found:
                                self.sv_type_training[sv_type].append(training_data)
                                print(f"Loaded training pair: {json_file.name} + {Path(training_data['image_path']).name}")
                            else:
                                print(f"Warning: Image not found for {json_file.name} (looking for {image_filename})")
                        else:
                            print(f"Warning: No filename specified in {json_file.name}")
                            
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Warning: Could not load {json_file}: {e}")
                        continue
                
                if self.sv_type_training[sv_type]:
                    print(f"Loaded {len(self.sv_type_training[sv_type])} complete training examples for {sv_type}")
                else:
                    print(f"No complete training pairs found in {sv_type_dir}")
            else:
                print(f"SV type directory not found: {sv_type_dir}")
        
        # Summary
        total_examples = sum(len(examples) for examples in self.sv_type_training.values())
        if total_examples == 0:
            print("Warning: No complete training examples loaded. Ensure both JSON and image files are present.")
        else:
            print(f"Total complete training examples loaded: {total_examples}")
            for sv_type, examples in self.sv_type_training.items():
                if examples:
                    real_count = sum(1 for ex in examples if ex.get("is_real", False))
                    false_count = len(examples) - real_count
                    print(f"  {sv_type}: {len(examples)} examples ({real_count} real, {false_count} false positive)")
    
    def select_training_examples(self, examples: List[Dict], max_examples: int = None, use_all: bool = False) -> List[Dict]:
        """
        Select training examples for optimal learning
        
        Args:
            examples: List of training examples with images
            max_examples: Maximum number of examples to select (None = no limit)
            use_all: If True, use all available examples regardless of max_examples
            
        Returns:
            Selected subset of examples
        """
        if use_all or max_examples is None:
            print(f"Using all {len(examples)} training examples")
            return examples
            
        if len(examples) <= max_examples:
            return examples
        
        # Separate real and false examples
        real_examples = [ex for ex in examples if ex.get("is_real", False)]
        false_examples = [ex for ex in examples if not ex.get("is_real", True)]
        
        # Try to get balanced selection
        target_real = min(len(real_examples), max_examples // 2)
        target_false = min(len(false_examples), max_examples - target_real)
        
        # If we have more false than real, adjust
        if target_false < max_examples // 2 and len(real_examples) > target_real:
            target_real = min(len(real_examples), max_examples - target_false)
        
        selected = []
        
        # Select real examples (prioritize high confidence if available)
        real_sorted = sorted(real_examples, 
                           key=lambda x: x.get("confidence_level", 0.5), 
                           reverse=True)
        selected.extend(real_sorted[:target_real])
        
        # Select false examples (prioritize high confidence if available)
        false_sorted = sorted(false_examples, 
                            key=lambda x: x.get("confidence_level", 0.5), 
                            reverse=True)
        selected.extend(false_sorted[:target_false])
        
        print(f"Selected {len(selected)} examples: {len([ex for ex in selected if ex.get('is_real')])} real, {len([ex for ex in selected if not ex.get('is_real')])} false")
        
        return selected
    
    def create_sv_specific_prompt(self, suspected_sv_type: str, use_all_training: bool = False) -> Tuple[str, List[Dict]]:
        """
        Create analysis prompt with training images for ChatGPT Vision
        
        Args:
            suspected_sv_type: SV type to focus analysis on (required)
            use_all_training: If True, use all training examples instead of limiting
            
        Returns:
            Tuple of (prompt_text, list_of_content_items_with_images)
        """
        if not suspected_sv_type or suspected_sv_type not in self.sv_type_criteria:
            raise ValueError(f"Must specify valid SV type. Choose from: {list(self.sv_type_criteria.keys())}")
        
        content_items = []
        
        # Create focused prompt for specific SV type
        criteria = self.sv_type_criteria[suspected_sv_type]
        
        prompt_text = f"""
You are an expert in {suspected_sv_type} structural variant analysis. Analyze the TARGET image based on these training examples and criteria.

SPECIFIC CRITERIA FOR {suspected_sv_type}:

Primary Features to Look For:
{chr(10).join(['- ' + feature for feature in criteria['primary_features']])}

Expected Coverage Pattern: {criteria['coverage_pattern']}
Expected Read Pair Orientation: {criteria['read_pair_orientation']}

FALSE POSITIVE INDICATORS:
{chr(10).join(['- ' + indicator for indicator in criteria['false_positive_indicators']])}

TRAINING EXAMPLES FOR {suspected_sv_type}:
"""
        
        # Add training examples with images
        available_examples = self.sv_type_training.get(suspected_sv_type, [])
        if not available_examples:
            raise ValueError(f"No training examples found for {suspected_sv_type}. Check your training data.")
            
        if use_all_training:
            selected_examples = self.select_training_examples(available_examples, use_all=True)
        else:
            selected_examples = self.select_training_examples(available_examples, max_examples=6)
        
        for i, example in enumerate(selected_examples):
            if "image_path" in example:
                prompt_text += f"""
TRAINING EXAMPLE {i+1}: {example.get('filename', 'N/A')}
- Real SV: {example.get('is_real', 'Unknown')}
- Type: {example.get('sv_type', 'Unknown')}
- Explanation: {example.get('explanation', 'No explanation')}
- Key features: {example.get('visual_features', {})}
---"""
                
                # Add the training image
                try:
                    base64_image = self.encode_image(example["image_path"])
                    content_items.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not load training image {example['image_path']}: {e}")

        prompt_text += """

Now analyze the TARGET image below and respond in JSON format:
{
    "is_real_sv": boolean,
    "sv_type": "DEL|DUP|INV|TRA|INS|COMPLEX|UNKNOWN",
    "confidence": float (0-1),
    "explanation": "detailed reasoning comparing to training examples",
    "supporting_evidence": ["list", "of", "key", "evidence"],
    "visual_features": {
        "coverage_pattern": "description",
        "read_pair_orientation": "description",
        "breakpoint_quality": "sharp|fuzzy|unclear",
        "mapping_quality": "high|medium|low"
    }
}

TARGET IMAGE TO ANALYZE:
"""
        
        return prompt_text, content_items
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_sv_image(self, image_path: str, suspected_sv_type: str, use_all_training: bool = False) -> SVResult:
        """
        Analyze SV image with training images and annotations
        
        Args:
            image_path: Path to bamsnap image
            suspected_sv_type: SV type to focus analysis on (required)
            use_all_training: If True, use all training examples instead of limiting
            
        Returns:
            SVResult with analysis
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not suspected_sv_type:
            raise ValueError("SV type must be specified. Choose from: DEL, DUP, INV, TRA, INS")
        
        # Create prompt with training examples and images
        prompt_text, training_content_items = self.create_sv_specific_prompt(suspected_sv_type, use_all_training)
        
        # Encode the target image to analyze
        base64_image = self.encode_image(image_path)
        
        # Build the complete content list: prompt + training images + target image
        content_list = [{"type": "text", "text": prompt_text}]
        content_list.extend(training_content_items)  # Add training images
        content_list.append({  # Add target image at the end
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })
        
        try:
            num_training_images = len(training_content_items)
            print(f"Sending {num_training_images} training images + 1 target image to ChatGPT Vision...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content_list
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            
            # Extract JSON
            try:
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = result_text[start_idx:end_idx]
                    result_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError):
                # Fallback parsing
                result_data = {
                    "is_real_sv": "real" in result_text.lower(),
                    "sv_type": suspected_sv_type or "UNKNOWN",
                    "confidence": 0.5,
                    "explanation": result_text,
                    "supporting_evidence": [],
                    "visual_features": {}
                }
            
            return SVResult(
                is_real_sv=result_data.get("is_real_sv", False),
                sv_type=result_data.get("sv_type", "UNKNOWN"),
                confidence=result_data.get("confidence", 0.5),
                explanation=result_data.get("explanation", ""),
                supporting_evidence=result_data.get("supporting_evidence", []),
                visual_features=result_data.get("visual_features", {})
            )
            
        except Exception as e:
            raise Exception(f"Error analyzing image {image_path}: {str(e)}")
    
    def analyze_batch_folder(self, images_dir: str, output_dir: str = None, suspected_sv_type: str = None, use_all_training: bool = False) -> List[Dict]:
        """
        Analyze all images in a folder (batch processing)
        
        Args:
            images_dir: Directory containing bamsnap images to analyze
            output_dir: Directory to save individual result files (optional)
            suspected_sv_type: SV type focus for all images (required)
            use_all_training: If True, use all training examples
            
        Returns:
            List of analysis results
        """
        if not suspected_sv_type:
            raise ValueError("SV type must be specified for batch analysis. Choose from: DEL, DUP, INV, TRA, INS")
        images_path = Path(images_dir)
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            return []
        
        print(f"Found {len(image_files)} images to analyze")
        if use_all_training:
            print("Using ALL training examples for maximum accuracy")
        
        results = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\nAnalyzing {i}/{len(image_files)}: {img_file.name}")
            
            try:
                # Analyze the image
                result = self.analyze_sv_image(str(img_file), suspected_sv_type, use_all_training)
                
                # Create result dictionary
                result_dict = {
                    "filename": img_file.name,
                    "filepath": str(img_file),
                    "is_real_sv": result.is_real_sv,
                    "sv_type": result.sv_type,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "supporting_evidence": result.supporting_evidence,
                    "visual_features": result.visual_features,
                    "analysis_focus": suspected_sv_type,
                    "used_all_training": use_all_training
                }
                
                results.append(result_dict)
                
                # Print summary
                status = "REAL SV" if result.is_real_sv else "FALSE POSITIVE"
                print(f"  Result: {status} - {result.sv_type} (confidence: {result.confidence:.2f})")
                
                # Save individual result file if output directory specified
                if output_dir:
                    result_filename = f"{img_file.stem}_result.json"
                    result_path = output_path / result_filename
                    with open(result_path, 'w') as f:
                        json.dump(result_dict, f, indent=2)
                    print(f"  Saved: {result_path}")
                
            except Exception as e:
                print(f"  Error analyzing {img_file.name}: {str(e)}")
                error_result = {
                    "filename": img_file.name,
                    "filepath": str(img_file),
                    "error": str(e),
                    "analysis_focus": suspected_sv_type,
                    "used_all_training": use_all_training
                }
                results.append(error_result)
        
        # Print batch summary
        print(f"\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        
        successful_results = [r for r in results if "error" not in r]
        error_count = len(results) - len(successful_results)
        
        print(f"Total images processed: {len(image_files)}")
        print(f"Successful analyses: {len(successful_results)}")
        print(f"Errors: {error_count}")
        
        if successful_results:
            real_svs = sum(1 for r in successful_results if r.get("is_real_sv", False))
            false_positives = len(successful_results) - real_svs
            
            print(f"\nClassification Results:")
            print(f"  Real SVs: {real_svs}")
            print(f"  False Positives: {false_positives}")
            
            # SV type breakdown
            sv_types = {}
            for r in successful_results:
                if r.get("is_real_sv", False):
                    sv_type = r.get("sv_type", "UNKNOWN")
                    sv_types[sv_type] = sv_types.get(sv_type, 0) + 1
            
            if sv_types:
                print(f"\nSV Type Breakdown (Real SVs only):")
                for sv_type, count in sorted(sv_types.items()):
                    print(f"  {sv_type}: {count}")
            
            # Confidence statistics
            confidences = [r.get("confidence", 0) for r in successful_results]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                high_conf = sum(1 for c in confidences if c >= 0.8)
                low_conf = sum(1 for c in confidences if c < 0.5)
                
                print(f"\nConfidence Statistics:")
                print(f"  Average confidence: {avg_confidence:.2f}")
                print(f"  High confidence (≥0.8): {high_conf}")
                print(f"  Low confidence (<0.5): {low_conf}")
        
        return results

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="SV Type-Specific Analysis")
    parser.add_argument("--image", help="Single image to analyze")
    parser.add_argument("--batch", help="Directory containing multiple images to analyze")
    parser.add_argument("--training", required=True, help="Training data directory")
    parser.add_argument("--sv-type", choices=["DEL", "DUP", "INV", "TRA", "INS"], 
                       required=True, help="SV type to focus analysis on (required)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--output-dir", help="Directory to save individual result files (for batch)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--use-all-training", action="store_true",
                       help="Use ALL training examples instead of limiting (higher cost but potentially better accuracy)")
    
    args = parser.parse_args()
    
    # Initialize analyzer (no API key needed - uses global variable)
    analyzer = SVTypeSpecificAnalyzer(args.model)
    
    # Load training data
    print(f"Loading SV-type-specific training data from: {args.training}")
    analyzer.load_sv_type_training(args.training)
    
    if args.use_all_training:
        print("WARNING: Using ALL training examples. This will be more expensive but potentially more accurate.")
    
    print(f"Analyzing for SV type: {args.sv_type}")
    
    # Single image analysis
    if args.image:
        print(f"Analyzing image: {args.image}")
        result = analyzer.analyze_sv_image(args.image, args.sv_type, args.use_all_training)
        
        print(f"\nAnalysis Results (focused on {args.sv_type}):")
        print(f"Real SV: {result.is_real_sv}")
        print(f"SV Type: {result.sv_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation}")
        print(f"Visual Features: {result.visual_features}")
        
        # Save single result
        if args.output:
            result_dict = {
                "filename": os.path.basename(args.image),
                "focus": args.sv_type,
                "is_real_sv": result.is_real_sv,
                "sv_type": result.sv_type,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "supporting_evidence": result.supporting_evidence,
                "visual_features": result.visual_features,
                "used_all_training": args.use_all_training
            }
            
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Results saved to: {args.output}")
    
    # Batch analysis
    elif args.batch:
        print(f"Running batch analysis on folder: {args.batch}")
        results = analyzer.analyze_batch_folder(
            args.batch, 
            args.output_dir, 
            args.sv_type, 
            args.use_all_training
        )
        
        # Save batch summary
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nBatch summary saved to: {args.output}")
    
    else:
        print("Please specify either --image or --batch")
        parser.print_help()

if __name__ == "__main__":
    main()