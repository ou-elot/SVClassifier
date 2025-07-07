# Structural Variant (SV) Image Classifier

A ChatGPT Vision API-based tool for analyzing bamsnap IGV images to classify structural variants as real or false positives and identify SV types.

## Overview

This tool uses OpenAI's GPT-4o Vision model to analyze bamsnap-generated IGV images and:
- Classify SVs as **real** or **false positive**
- Identify SV type: **DEL**, **DUP**, **INV**, **TRA**, **INS**
- Provide confidence scores and detailed explanations
- Support supervised learning with custom training data

## Features

- **SV Type-Specific Analysis**: Tailored analysis for each structural variant type
- **Supervised Learning**: Train with your own bamsnap images and expert annotations
- **Comparative Analysis**: Compare results across different SV type focuses
- **Batch Processing**: Analyze multiple images at once
- **Detailed Explanations**: Get reasoning behind each classification

## Requirements

- Python 3.7+
- OpenAI API key with Vision access
- Dependencies: `openai`, `pillow`

## Installation

1. **Clone/download the project files**
2. **Install dependencies:**
   ```bash
   pip install openai pillow
   ```
3. **Set up API key:** Your OpenAI API key is already configured in the code

## Quick Start

### 1. Prepare Training Data

Create a folder with your bamsnap images and annotations:

```
my_training/
├── deletion_real.png
├── deletion_real.json      # Your annotation
├── deletion_fake.png
├── deletion_fake.json
├── duplication_real.png
├── duplication_real.json
└── ...
```

### 2. Create Annotation Files

For each image, create a matching JSON file:

```json
{
  "filename": "deletion_real.png",
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

```bash
# General analysis
python classifier.py --image test.png --training ./my_training

# Focus on specific SV type
python classifier.py --image test.png --training ./my_training --sv-type DEL

# Compare across all SV types
python classifier.py --image test.png --training ./my_training --compare-types
```

## Usage Examples

### Single Image Analysis
```bash
python classifier.py \
    --image bamsnap_candidate.png \
    --training ./my_training \
    --output result.json
```

### SV Type-Focused Analysis
```bash
# Focus on deletions
python classifier.py \
    --image suspected_deletion.png \
    --training ./my_training \
    --sv-type DEL \
    --output deletion_result.json

# Focus on duplications
python classifier.py \
    --image suspected_dup.png \
    --training ./my_training \
    --sv-type DUP
```

### Comparative Analysis
```bash
python classifier.py \
    --image uncertain_sv.png \
    --training ./my_training \
    --compare-types \
    --output comparison.json
```

### Batch Processing
```bash
# Process all images in a directory
for img in bamsnap_images/*.png; do
    python classifier.py \
        --image "$img" \
        --training ./my_training \
        --output "results/$(basename "$img" .png)_result.json"
done
```

## Command Line Arguments

- `--image`: Path to bamsnap image to analyze
- `--training`: Path to training data directory (required)
- `--sv-type`: Focus on specific SV type (`DEL`, `DUP`, `INV`, `TRA`, `INS`)
- `--compare-types`: Compare results across all SV types
- `--output`: Save results to JSON file
- `--model`: OpenAI model to use (`gpt-4o` or `gpt-4o-mini`)

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

## Output Format

### Single Analysis
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

### Comparative Analysis
```json
{
  "GENERAL": {
    "is_real_sv": true,
    "sv_type": "DEL",
    "confidence": 0.85
  },
  "DEL": {
    "is_real_sv": true,
    "sv_type": "DEL",
    "confidence": 0.92
  },
  "DUP": {
    "is_real_sv": false,
    "sv_type": "DUP",
    "confidence": 0.15
  }
}
```

## Training Data Guidelines

### Quality Training Examples
- Include both **real SVs** and **false positives**
- Provide **detailed explanations** of visual features
- Document **edge cases** and challenging examples
- Be **consistent** in annotation criteria

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

## Cost Information

- **gpt-4o**: ~$0.01-0.02 per image (higher accuracy)