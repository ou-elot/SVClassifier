import json
import openai
import csv
from pathlib import Path
from typing import List, Dict
import argparse

# Use same API key as classifier
OPENAI_API_KEY = "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"

class SVResultsAnalyzer:
    """
    Analyze SV classifier JSON results using ChatGPT for insights and summaries
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
    
    def load_json_results(self, results_path: str) -> List[Dict]:
        """Load JSON results from file or directory"""
        results_path = Path(results_path)
        
        if results_path.is_file():
            # Single JSON file
            with open(results_path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        
        elif results_path.is_dir():
            # Directory of JSON files
            results = []
            for json_file in results_path.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            results.extend(data)
                        else:
                            results.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
            return results
        
        else:
            raise FileNotFoundError(f"Results path not found: {results_path}")
    
    def analyze_batch_summary(self, results: List[Dict]) -> str:
        """Generate comprehensive batch analysis summary"""
        
        # Prepare data summary for ChatGPT
        total_results = len(results)
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return "No successful results to analyze."
        
        # Create summary statistics
        real_svs = sum(1 for r in successful_results if r.get("is_real_sv", False))
        false_positives = len(successful_results) - real_svs
        
        # Confidence statistics
        confidences = [r.get("confidence", 0) for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        low_confidence = [r for r in successful_results if r.get("confidence", 0) < 0.5]
        high_confidence = [r for r in successful_results if r.get("confidence", 0) >= 0.8]
        
        # SV type breakdown
        sv_types = {}
        for r in successful_results:
            if r.get("is_real_sv", False):
                sv_type = r.get("sv_type", "UNKNOWN")
                sv_types[sv_type] = sv_types.get(sv_type, 0) + 1
        
        # Prepare sample results for analysis
        sample_real = [r for r in successful_results if r.get("is_real_sv", False)][:5]
        sample_false = [r for r in successful_results if not r.get("is_real_sv", True)][:5]
        sample_low_conf = low_confidence[:3]
        
        prompt = f"""
You are an expert in structural variant analysis. Analyze this batch of SV classification results and provide insights.

DATASET SUMMARY:
- Total images processed: {total_results}
- Successful analyses: {len(successful_results)}
- Real SVs identified: {real_svs}
- False positives: {false_positives}
- Average confidence: {avg_confidence:.2f}
- Low confidence results (<0.5): {len(low_confidence)}
- High confidence results (≥0.8): {len(high_confidence)}

SV TYPE BREAKDOWN (Real SVs only):
{json.dumps(sv_types, indent=2)}

SAMPLE REAL SV RESULTS:
{json.dumps(sample_real, indent=2)}

SAMPLE FALSE POSITIVE RESULTS:
{json.dumps(sample_false, indent=2)}

SAMPLE LOW CONFIDENCE RESULTS:
{json.dumps(sample_low_conf, indent=2)}

Please provide a comprehensive analysis including:

1. OVERALL ASSESSMENT:
   - Quality of the batch results
   - Distribution of SV types and what this suggests
   - Confidence level patterns

2. QUALITY CONTROL INSIGHTS:
   - Which results need manual review?
   - Common patterns in false positives
   - What might be causing low confidence calls?

3. BIOLOGICAL INSIGHTS:
   - Are the SV patterns consistent with expectations?
   - Any unusual findings or concerning patterns?
   - Clinical relevance of identified SVs

4. RECOMMENDATIONS:
   - Should any results be flagged for follow-up?
   - What training data improvements are needed?
   - Any systematic issues to address?

5. SUMMARY STATISTICS:
   - Key metrics and takeaways
   - Success rate assessment
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def analyze_specific_question(self, results: List[Dict], question: str) -> str:
        """Ask specific questions about the results"""
        
        # Limit results for context (ChatGPT has token limits)
        sample_results = results[:50] if len(results) > 50 else results
        
        prompt = f"""
You are an expert in structural variant analysis. Answer the following question about these SV classification results:

QUESTION: {question}

RESULTS DATA:
{json.dumps(sample_results, indent=2)}

Please provide a detailed, specific answer based on the data provided.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing question: {str(e)}"
    
    def quality_control_review(self, results: List[Dict]) -> str:
        """Focus on quality control and potential issues"""
        
        successful_results = [r for r in results if "error" not in r]
        
        # Identify problematic results
        low_confidence = [r for r in successful_results if r.get("confidence", 0) < 0.5]
        uncertain_calls = [r for r in successful_results if 0.3 <= r.get("confidence", 0) <= 0.7]
        
        prompt = f"""
You are conducting quality control review of SV classification results. Focus on identifying potential issues and recommendations.

LOW CONFIDENCE RESULTS (<0.5 confidence):
{json.dumps(low_confidence[:10], indent=2)}

UNCERTAIN CALLS (0.3-0.7 confidence):
{json.dumps(uncertain_calls[:10], indent=2)}

Please provide:

1. IMMEDIATE ACTIONS NEEDED:
   - Which specific results need manual review?
   - Any obvious misclassifications?

2. SYSTEMATIC ISSUES:
   - Common patterns in low-confidence calls
   - Recurring problems in explanations
   - Potential training data gaps

3. CONFIDENCE ASSESSMENT:
   - Are the confidence scores reliable?
   - What factors correlate with low confidence?

4. PRIORITIZATION:
   - Which results should be reviewed first?
   - Risk assessment for missed SVs vs false positives

5. TRAINING IMPROVEMENTS:
   - What additional training examples are needed?
   - Which SV types need better representation?
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error in quality control review: {str(e)}"
    
    def export_to_csv(self, results: List[Dict], output_file: str) -> None:
        """
        Export results to CSV format: image_name, sv_type, is_real, reason
        
        Args:
            results: List of classification results
            output_file: Path to save CSV file
        """
        successful_results = [r for r in results if "error" not in r]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['image_name', 'sv_type', 'is_real', 'confidence', 'reason'])
            
            # Write data rows
            for result in successful_results:
                image_name = result.get('filename', 'unknown')
                sv_type = result.get('sv_type', 'UNKNOWN')
                is_real = 'TRUE' if result.get('is_real_sv', False) else 'FALSE'
                confidence = f"{result.get('confidence', 0):.2f}"
                
                # Get reason from explanation (truncate if too long)
                explanation = result.get('explanation', 'No explanation provided')
                # Clean up the reason - take first sentence or first 100 chars
                reason = explanation.split('.')[0].strip()
                if len(reason) > 100:
                    reason = reason[:97] + "..."
                
                writer.writerow([image_name, sv_type, is_real, confidence, reason])
        
        print(f"CSV exported to: {output_file}")
        print(f"Exported {len(successful_results)} results")
    
    def print_csv_summary(self, results: List[Dict]) -> None:
        """
        Print results in CSV format to console
        """
        successful_results = [r for r in results if "error" not in r]
        
        print("\n" + "="*100)
        print("CSV FORMAT SUMMARY")
        print("="*100)
        print(f"{'IMAGE_NAME':<25} {'SV_TYPE':<8} {'IS_REAL':<8} {'CONFIDENCE':<10} {'REASON':<50}")
        print("-" * 100)
        
        for result in successful_results:
            image_name = result.get('filename', 'unknown')[:24]  # Truncate long names
            sv_type = result.get('sv_type', 'UNKNOWN')
            is_real = 'TRUE' if result.get('is_real_sv', False) else 'FALSE'
            confidence = f"{result.get('confidence', 0):.2f}"
            
            # Get reason from explanation (truncate for display)
            explanation = result.get('explanation', 'No explanation')
            reason = explanation.split('.')[0].strip()
            if len(reason) > 47:
                reason = reason[:44] + "..."
            
            print(f"{image_name:<25} {sv_type:<8} {is_real:<8} {confidence:<10} {reason:<50}")
    
    def create_summary_statistics(self, results: List[Dict]) -> Dict:
        """Create summary statistics for the batch"""
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return {}
        
        # Basic stats
        total = len(successful_results)
        real_svs = sum(1 for r in successful_results if r.get("is_real_sv", False))
        false_positives = total - real_svs
        
        # SV type breakdown
        sv_types = {}
        for r in successful_results:
            if r.get("is_real_sv", False):
                sv_type = r.get("sv_type", "UNKNOWN")
                sv_types[sv_type] = sv_types.get(sv_type, 0) + 1
        
        # Confidence stats
        confidences = [r.get("confidence", 0) for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_conf = sum(1 for c in confidences if c >= 0.8)
        medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.5)
        
        return {
            'total_images': total,
            'real_svs': real_svs,
            'false_positives': false_positives,
            'real_sv_percentage': (real_svs / total * 100) if total > 0 else 0,
            'sv_type_breakdown': sv_types,
            'avg_confidence': avg_confidence,
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf
        }

def main():
    """Command line interface for analyzing SV results"""
    parser = argparse.ArgumentParser(description="Analyze SV classification results with ChatGPT")
    parser.add_argument("--results", required=True, help="JSON results file or directory")
    parser.add_argument("--analysis-type", choices=["summary", "quality", "question"], 
                       default="summary", help="Type of analysis to perform")
    parser.add_argument("--question", help="Specific question to ask about the results")
    parser.add_argument("--output", help="Save analysis to file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SVResultsAnalyzer()
    
    # Load results
    print(f"Loading results from: {args.results}")
    try:
        results = analyzer.load_json_results(args.results)
        print(f"Loaded {len(results)} results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Perform analysis
    if args.analysis_type == "summary":
        print("Generating comprehensive batch analysis...")
        analysis = analyzer.analyze_batch_summary(results)
        
    elif args.analysis_type == "quality":
        print("Performing quality control review...")
        analysis = analyzer.quality_control_review(results)
        
    elif args.analysis_type == "question":
        if not args.question:
            print("Please provide a question with --question")
            return
        print(f"Analyzing question: {args.question}")
        analysis = analyzer.analyze_specific_question(results, args.question)
    
    # Output results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(analysis)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(analysis)
        print(f"\nAnalysis saved to: {args.output}")

if __name__ == "__main__":
    main()