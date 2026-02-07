# AI Content Detection, Plagiarism Check & Humanization Suite

Three powerful Python scripts to analyze and improve your review paper using Hugging Face models.

## 📋 All Scripts

### 1. AI Content Detector (`ai_content_detector.py`)
- Detects AI-generated text using transformer models
- Analyzes text in chunks to handle large documents
- Provides overall AI probability score
- Highlights sections with highest AI probability

### 2. Plagiarism Detector (`plagiarism_detector.py`)
- Detects internal duplication (self-plagiarism)
- Uses semantic similarity analysis
- Identifies similar sentence pairs
- Can be extended with web search APIs

### 3. **AI Content Humanizer (`ai_humanizer.py`)** ⭐ NEW
- **Detects AI-generated content**
- **Automatically paraphrases flagged sections**
- **Generates multiple humanized alternatives**
- **Creates side-by-side comparison document**
- **Provides best humanized version of entire paper**

## 🚀 Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Basic Usage

**Check for AI content and humanize it:**
```bash
python ai_humanizer.py your_paper.pdf
```

This will create:
- `humanized_document.docx` - Word doc with original vs humanized text
- `humanization_report.txt` - Detailed text report

**Just check for AI content:**
```bash
python ai_content_detector.py your_paper.pdf
```

**Check for plagiarism:**
```bash
python plagiarism_detector.py your_paper.pdf
```

## 📖 Detailed Usage

### AI Content Humanizer (Recommended)

**Basic usage:**
```bash
python ai_humanizer.py your_paper.pdf
```

**Adjust sensitivity threshold:**
```bash
# More sensitive (detect more AI content)
python ai_humanizer.py your_paper.pdf --threshold 0.5

# Less sensitive (only flag high-confidence AI)
python ai_humanizer.py your_paper.pdf --threshold 0.8
```

**Custom output files:**
```bash
python ai_humanizer.py your_paper.pdf \
    --output-docx my_humanized_paper.docx \
    --output-txt my_report.txt
```

**Use different models:**
```bash
python ai_humanizer.py your_paper.pdf \
    --detector "Hello-SimpleAI/chatgpt-detector-roberta" \
    --paraphraser "Vamsi/T5_Paraphrase_Paws"
```

**What it generates:**

1. **Word Document** (`humanized_document.docx`):
   - Summary statistics
   - Color-coded comparisons (green = good, red = needs work)
   - Original text with AI probability scores
   - 3 humanized alternatives for each flagged section
   - Complete humanized version of your paper

2. **Text Report** (`humanization_report.txt`):
   - Detailed statistics
   - All flagged sections with alternatives
   - Full humanized text for easy copy-paste

### AI Content Detector Only

```bash
python ai_content_detector.py your_paper.pdf --output ai_report.txt
```

### Plagiarism Detector Only

```bash
python plagiarism_detector.py your_paper.pdf --output plag_report.txt
```

## 🎯 How It Works

### The Humanization Process

1. **Extract Text**: Reads your PDF and extracts all text
2. **Split into Sentences**: Breaks down into individual sentences
3. **Detect AI Content**: Each sentence gets an AI probability score
4. **Flag High-Probability Sentences**: Sentences above threshold are flagged
5. **Generate Paraphrases**: Creates 3 humanized versions for each flagged sentence
6. **Re-check AI Score**: Verifies the paraphrases are more human-like
7. **Select Best Version**: Chooses the version with lowest AI probability
8. **Create Reports**: Generates Word doc and text file with results

### Models Used

**For AI Detection:**
- `roberta-base-openai-detector` (default) - Detects GPT-style text
- `Hello-SimpleAI/chatgpt-detector-roberta` - Specialized for ChatGPT

**For Humanization/Paraphrasing:**
- `humarin/chatgpt_paraphraser_on_T5_base` (default) - T5-based paraphraser
- `Vamsi/T5_Paraphrase_Paws` - Alternative paraphraser
- `ramsrigouthamg/t5-large-paraphraser-diverse-high-quality` - High quality option

## 📊 Understanding Results

### AI Probability Scores

- **0-50%** (Green): Likely human-written ✓
- **50-70%** (Yellow): Borderline, review recommended ⚠️
- **70-100%** (Red): High likelihood of AI generation ⚠️⚠️

### Threshold Settings

- **0.5**: Very sensitive - flags more content, may have false positives
- **0.7** (default): Balanced - good for most use cases
- **0.8**: Conservative - only flags high-confidence AI content

### Improvement Scores

The humanizer shows "improvement" percentages:
- **Positive improvement**: Successfully reduced AI detection
- **Negative improvement**: Paraphrase still flagged as AI (choose different option)

## 💡 Best Practices

### Before Running Scripts

1. **Remove non-original content:**
   - Delete references section
   - Remove acknowledgments
   - Remove quotes/citations (add them back later)
   - Keep only your written content

2. **Split large papers:**
   ```bash
   # For papers >30 pages, process by chapter
   python ai_humanizer.py chapter1.pdf --output-docx chapter1_humanized.docx
   python ai_humanizer.py chapter2.pdf --output-docx chapter2_humanized.docx
   ```

### Reviewing Humanized Output

1. **Check the Word document first** - easier to compare
2. **Review each flagged section** - don't blindly accept paraphrases
3. **Choose the option that:**
   - Has lowest AI probability (green)
   - Maintains your original meaning
   - Sounds natural to you
4. **Manually refine** - use humanized versions as starting points
5. **Add your own touch** - personalize with examples, analysis

### Combining All Three Scripts

```bash
# 1. Check original AI content
python ai_content_detector.py paper.pdf --output original_ai_check.txt

# 2. Check plagiarism
python plagiarism_detector.py paper.pdf --output plagiarism_check.txt

# 3. Humanize AI content
python ai_humanizer.py paper.pdf --threshold 0.7

# 4. Re-check the humanized version
# (convert humanized text back to PDF first, or use the text output)
```

## ⚙️ Advanced Options

### Memory Optimization

If you get out-of-memory errors:

```python
# Edit ai_humanizer.py and reduce batch size
# Around line 150-160, change num_return_sequences:
num_return_sequences=1,  # Instead of 3
```

Or process smaller chunks:
```bash
# Split your PDF into smaller files first
pdftk original.pdf cat 1-10 output part1.pdf
pdftk original.pdf cat 11-20 output part2.pdf
```

### Model Alternatives

**Faster processing (smaller models):**
```bash
python ai_humanizer.py paper.pdf \
    --paraphraser "Vamsi/T5_Paraphrase_Paws"
```

**Higher quality (larger models, slower):**
```bash
python ai_humanizer.py paper.pdf \
    --paraphraser "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
```

**Different AI detector:**
```bash
python ai_humanizer.py paper.pdf \
    --detector "andreas122001/roberta-mixed-detector"
```

## 🔧 Troubleshooting

### Common Issues

**1. "Out of memory" error:**
```bash
# Use CPU instead of GPU (slower but uses less memory)
export CUDA_VISIBLE_DEVICES=""
python ai_humanizer.py paper.pdf
```

**2. Paraphrases don't sound natural:**
- Try a different paraphrasing model
- Lower the threshold to flag less content
- Manually edit the paraphrases

**3. Models downloading slowly:**
```bash
# Set cache directory to faster drive
export HF_HOME=/path/to/fast/drive/hf_cache
```

**4. PDF text extraction issues:**
- Ensure PDF contains actual text (not scanned images)
- Try converting PDF to text first: `pdftotext yourpaper.pdf`

**5. Script takes too long:**
- Process smaller sections of your paper
- Use faster/smaller models
- Reduce number of paraphrase variants (edit script)

## 📁 Output Files Explained

### From ai_humanizer.py

**humanized_document.docx:**
- Professional Word document
- Color-coded for easy review
- Side-by-side original vs humanized
- Multiple options for each flagged section
- Complete humanized draft at the end

**humanization_report.txt:**
- Plain text version
- Good for quick review
- Easy to copy-paste sections
- Full statistics

### From ai_content_detector.py

**ai_detection_report.txt:**
- Overall AI probability
- Top sections with highest AI scores
- Detailed analysis of all chunks

### From plagiarism_detector.py

**plagiarism_report.txt:**
- Internal duplication statistics
- Similar sentence pairs
- Recommendations

## 🎓 Example Workflow

**Complete paper improvement workflow:**

```bash
# Step 1: Initial assessment
python ai_content_detector.py review_paper.pdf
python plagiarism_detector.py review_paper.pdf

# Step 2: Humanize AI content
python ai_humanizer.py review_paper.pdf --threshold 0.7

# Step 3: Review the Word document
# Open humanized_document.docx
# For each flagged section:
#   - Read original
#   - Review 3 humanized options
#   - Pick best one or manually refine
#   - Copy to your actual paper

# Step 4: Verify improvements
# Convert your edited version back to PDF
python ai_content_detector.py review_paper_v2.pdf

# Step 5: Final polish
# Manually review and add your personal touches
```

## ⚠️ Important Disclaimers

### What This Tool IS:
- A helper to identify and rephrase AI-detected content
- A tool to improve writing and reduce AI signatures
- A starting point for manual refinement

### What This Tool IS NOT:
- A guarantee that content won't be detected as AI
- A replacement for original writing
- A way to use AI-written content unethically

### Ethical Use:
- **DO**: Use to improve your own drafts and reduce AI signatures
- **DO**: Manually review and refine all paraphrased content
- **DO**: Add your own analysis, examples, and insights
- **DON'T**: Submit AI-written content as your own work
- **DON'T**: Rely solely on paraphrasing without adding original thought
- **DON'T**: Use this to bypass academic integrity policies

### Accuracy Notes:
- AI detection is probabilistic, not definitive
- False positives can occur with formal academic writing
- Paraphrasing doesn't make content original if it wasn't yours
- Always add your own critical analysis and insights

## 📚 Recommended Models

### For Academic Papers

**Best overall:**
```bash
python ai_humanizer.py paper.pdf \
    --detector "roberta-base-openai-detector" \
    --paraphraser "humarin/chatgpt_paraphraser_on_T5_base" \
    --threshold 0.7
```

**For highly technical content:**
```bash
python ai_humanizer.py paper.pdf \
    --threshold 0.8  # Higher threshold to avoid false positives
```

**For maximum quality (slower):**
```bash
python ai_humanizer.py paper.pdf \
    --paraphraser "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
```

## 🆘 Getting Help

If you encounter issues:

1. Check error message carefully
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Try with a smaller test PDF first
4. Check available disk space (models need ~2-3GB)
5. Try different models if current ones don't work
6. Check GPU memory if using CUDA

## 📄 License

These scripts are provided for academic self-improvement. Use responsibly and ethically.

---

**Remember**: These tools help you improve your writing, but the best humanization is your own original thinking, analysis, and voice! 🎓
