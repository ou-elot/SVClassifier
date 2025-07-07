# Structural Variant (SV) Image Classifier

A ChatGPT Vision API-based tool for analyzing bamsnap IGV images to classify structural variants as real or false positives for specific SV types.

## Overview

This tool uses OpenAI's GPT-4o Vision model to analyze bamsnap-generated IGV images and:
- Classify SVs as **real** or **false positive** for a specific SV type
- Identify and validate SV types: **DEL**, **DUP**, **INV**, **TRA**, **INS**
- Provide confidence scores and detailed explanations
- Support supervised learning with custom training data
- Export results to CSV format for further analysis

## Features

- **SV Type-Specific Analysis**: Focused analysis for each structural variant type (DEL, DUP, INV, TRA, INS)
- **Supervised Learning**: Train with your own bamsnap images and expert annotations
- **Batch Processing**: Analyze multiple images at once
- **Detailed Explanations**: Get reasoning behind each classification
- **CSV Export**: Clean summary format for spreadsheets and databases
- **All Training Examples**: Option to use complete training sets for maximum accuracy

## Requirements

- Python 3.7+
- OpenAI API key with Vision access
- Dependencies: `openai>=1.0.0`, `Pillow>=9.0.0`

## Installation

1. **Install dependencies:**
   ```bash
   pip install openai>=1.0.0 Pillow>=9.0.0
   ```

2. **Download the classifier files:**
   - `classify.py` - Main SV classifier
   - `json_analyzer.py` - Results analyzer and CSV exporter

3. **Set up training data** (see structure below)

## Quick Start

### 1. Prepare Training Data

Create folders for each SV type with images and annotations:

```
trainingData/
├── del/
│   ├── deletion_real_1.png
│   ├── deletion_real_1.json      # Your annotation
│   ├── deletion_fake_1.png
│   ├── deletion_fake_1.json
│   └── ...
├── dup/
│   ├── duplication_real_1.png
│   ├── duplication_real_1.json
│   └── ...
├── inv/
├── tra/
└── ins/
```

### 2. Create Annotation Files

For each image, create a matching JSON file:

```json
{
  "filename": "deletion_real_1.png",
  "is_real": true,
  "sv_type": "DEL",
  "explanation": "Clear 5kb deletion with 60% coverage drop. Sharp boundaries at chr1:12345 and chr1:17345. Outward-facing discordant read pairs clustered at breakpoints.",
  "visual_features": {
    "coverage_pattern": "60% decrease",
    "breakpoint_precision": "sharp",
    "read_pair_orientation": "outward-facing"
  },
  "supporting_evidence": [
    "Sharp coverage boundaries",
    "Clustered discordant pairs",
    "High mapping quality reads"
  ]
}
```

### 3. Analyze Images

**SV type must always be specified:**

```bash
# Single image analysis
python classify.py --image test.png --training ./trainingData --sv-type DEL

# Batch processing
python classify.py --batch ./images --training ./trainingData --sv-type DEL --output-dir ./results
```

## Usage Examples

### Single Image Analysis
```bash
python classify.py \
    --image bamsnap_candidate.png \
    --training ./trainingData \
    --sv-type DEL \
    --output result.json
```

### Batch Processing
```bash
# Analyze folder of deletion candidates
python classify.py \
    --batch ./deletion_candidates \
    --training ./trainingData \
    --sv-type DEL \
    --output-dir ./results \
    --output batch_summary.json
```

### Using All Training Examples (Higher Accuracy)
```bash
python classify.py \
    --batch ./images \
    --training ./trainingData \
    --sv-type DEL \
    --use-all-training \
    --output-dir ./results
```

### Results Analysis and CSV Export
```bash
# Analyze results and export to CSV
python json_analyzer.py \
    --results batch_summary.json \
    --csv-output final_results.csv \
    --print-csv
```

## Command Line Arguments

### classify.py
- `--image`: Path to single bamsnap image to analyze
- `--batch`: Directory containing multiple images to analyze
- `--training`: Path to training data directory (required)
- `--sv-type`: SV type to analyze (required: `DEL`, `DUP`, `INV`, `TRA`, `INS`)
- `--output`: Save results to JSON file
- `--output-dir`: Directory for individual result files (batch mode)
- `--use-all-training`: Use all training examples (higher cost, better accuracy)
- `--model`: OpenAI model (`gpt-4o` or `gpt-4o-mini`)

### json_analyzer.py
- `--results`: JSON results file or directory (required)
- `--analysis-type`: Analysis type (`summary`, `quality`, `question`, `csv`)
- `--question`: Specific question about results
- `--output`: Save detailed analysis to file
- `--csv-output`: Export to CSV file
- `--print-csv`: Print CSV format to console

## SV Type Characteristics

### Deletions (DEL)
- **Key Features**: Sharp coverage drop (50%+), outward-facing read pairs (`-->  <--`)
- **False Positives**: Gradual decline, low mapping quality, repetitive regions

### Duplications (DUP)
- **Key Features**: Coverage increase (150%+), inward-facing read pairs (`<--  -->`)
- **False Positives**: GC bias, irregular patterns, mixed orientations

### Inversions (INV)
- **Key Features**: Same-orientation pairs (`-->  -->` or `<--  <--`), no coverage change
- **False Positives**: Mixed orientations, coverage fluctuations

### Translocations (TRA)
- **Key Features**: Inter-chromosomal mapping, junction sequences, no coverage change
- **False Positives**: Low mapping quality, repetitive breakpoints

### Insertions (INS)
- **Key Features**: Split reads with inserted sequence, soft-clipping, no coverage change
- **False Positives**: Homopolymers, poor sequence quality

## Output Formats

### JSON Results (from classify.py)
```json
{
  "filename": "test.png",
  "is_real_sv": true,
  "sv_type": "DEL",
  "confidence": 0.89,
  "explanation": "Clear deletion with sharp coverage drop...",
  "supporting_evidence": ["Coverage drop", "Outward pairs"],
  "visual_features": {
    "coverage_pattern": "50% decrease",
    "breakpoint_quality": "sharp"
  }
}
```

### CSV Export (from json_analyzer.py)
```csv
image_name,sv_type,is_real,confidence,reason
deletion_1.png,DEL,TRUE,0.89,"Clear deletion with sharp coverage drop"
duplication_2.png,DUP,FALSE,0.23,"Coverage increase appears to be GC bias"
```

## Training Data Guidelines

### Quality Training Examples
- Include both **real SVs** and **false positives**
- Provide **detailed explanations** of visual features
- Document **edge cases** and challenging examples
- Be **consistent** in annotation criteria

### File Structure Requirements
```
trainingData/
├── [sv_type]/
│   ├── image1.png
│   ├── image1.json
│   ├── image2.png
│   └── image2.json
```

### Annotation Template
```json
{
  "filename": "image.png",
  "is_real": true/false,
  "sv_type": "DEL/DUP/INV/TRA/INS",
  "explanation": "Detailed description of what you observe",
  "visual_features": {
    "coverage_pattern": "specific pattern description",
    "breakpoint_precision": "sharp/fuzzy/unclear",
    "read_pair_orientation": "specific orientation",
    "mapping_quality": "high/medium/low"
  },
  "supporting_evidence": ["list", "of", "key", "features"]
}
```
## Complete Workflow

### 1. Prepare Training Data
```bash
# Organize your bamsnap images and create annotations
trainingData/del/deletion_1.png + deletion_1.json
trainingData/dup/duplication_1.png + duplication_1.json
```

### 2. Classify Images
```bash
# Analyze your target images
python classify.py --batch ./target_images --training ./trainingData --sv-type DEL --output-dir ./results
```

### 3. Analyze Results
```bash
# Get insights and CSV export
python json_analyzer.py --results ./results --csv-output final_summary.csv
```

### File Structure
```
project/
├── classify.py              # Main classifier
├── json_analyzer.py         # Results analyzer
├── requirements.txt         # Dependencies
├── trainingData/           # Your training data
│   ├── del/
│   ├── dup/
│   ├── inv/
│   ├── tra/
│   └── ins/
├── target_images/          # Images to analyze
└── results/               # Analysis results
    ├── image1_result.json
    ├── image2_result.json
    └── batch_summary.json
```

## License

This tool is for research use. OpenAI API usage subject to OpenAI's terms of service.
