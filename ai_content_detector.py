#!/usr/bin/env python3
"""
AI Content Detector for Academic Papers
Uses Hugging Face models to detect AI-generated text in PDF documents
"""

import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple
import argparse

class AIContentDetector:
    def __init__(self, model_name: str = "roberta-base-openai-detector"):
        """
        Initialize the AI content detector
        
        Args:
            model_name: Hugging Face model for AI detection
        """
        print("Loading AI detection model...")
        try:
            # Using roberta-base-openai-detector (good for detecting GPT-generated text)
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative model...")
            # Fallback to another detector
            self.tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
            self.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
            self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        print(f"Extracting text from {pdf_path}...")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text()
                    if (page_num + 1) % 5 == 0:
                        print(f"Processed {page_num + 1} pages...")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        
        print(f"Extracted {len(text.split())} words")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for analysis
        
        Args:
            text: Input text
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        # Split by sentences (rough approximation)
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def detect_ai_content(self, text: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Detect AI-generated content in text
        
        Returns:
            Tuple of (overall_ai_probability, list of (chunk, probability))
        """
        chunks = self.chunk_text(text)
        results = []
        
        print("\nAnalyzing chunks for AI content...")
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                # Assuming label 1 is "AI-generated" (check model card for actual labels)
                ai_probability = predictions[0][1].item()
            
            results.append((chunk[:200] + "...", ai_probability))
            
            if (i + 1) % 10 == 0:
                print(f"Analyzed {i + 1}/{len(chunks)} chunks...")
        
        # Calculate overall AI probability (weighted average)
        if results:
            overall_probability = np.mean([prob for _, prob in results])
        else:
            overall_probability = 0.0
        
        return overall_probability, results
    
    def generate_report(self, pdf_path: str, output_file: str = "ai_detection_report.txt"):
        """Generate a detailed report of AI content detection"""
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("No text extracted from PDF!")
            return
        
        overall_prob, chunk_results = self.detect_ai_content(text)
        
        # Sort chunks by AI probability (highest first)
        chunk_results.sort(key=lambda x: x[1], reverse=True)
        
        # Generate report
        print("\n" + "="*80)
        print("AI CONTENT DETECTION REPORT")
        print("="*80)
        print(f"\nDocument: {pdf_path}")
        print(f"Total words: ~{len(text.split())}")
        print(f"Overall AI Content Probability: {overall_prob*100:.2f}%")
        
        if overall_prob > 0.8:
            verdict = "HIGH likelihood of AI-generated content"
        elif overall_prob > 0.5:
            verdict = "MODERATE likelihood of AI-generated content"
        elif overall_prob > 0.3:
            verdict = "LOW-MODERATE likelihood of AI-generated content"
        else:
            verdict = "LOW likelihood of AI-generated content"
        
        print(f"Verdict: {verdict}")
        
        # Write detailed report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AI CONTENT DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Document: {pdf_path}\n")
            f.write(f"Total words: ~{len(text.split())}\n")
            f.write(f"Overall AI Content Probability: {overall_prob*100:.2f}%\n")
            f.write(f"Verdict: {verdict}\n\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("TOP 10 SECTIONS WITH HIGHEST AI PROBABILITY\n")
            f.write("-"*80 + "\n\n")
            
            for i, (chunk, prob) in enumerate(chunk_results[:10], 1):
                f.write(f"{i}. AI Probability: {prob*100:.2f}%\n")
                f.write(f"Text preview: {chunk}\n\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("ALL SECTIONS ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            for i, (chunk, prob) in enumerate(chunk_results, 1):
                f.write(f"Section {i}: {prob*100:.2f}% AI probability\n")
        
        print(f"\nDetailed report saved to: {output_file}")
        print("\nTop 5 sections with highest AI probability:")
        for i, (chunk, prob) in enumerate(chunk_results[:5], 1):
            print(f"\n{i}. AI Probability: {prob*100:.2f}%")
            print(f"   Preview: {chunk[:150]}...")


def main():
    parser = argparse.ArgumentParser(description='Detect AI-generated content in PDF documents')
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file to analyze')
    parser.add_argument('--output', type=str, default='ai_detection_report.txt',
                       help='Output report file name (default: ai_detection_report.txt)')
    parser.add_argument('--model', type=str, default='roberta-base-openai-detector',
                       help='Hugging Face model to use for detection')
    
    args = parser.parse_args()
    
    detector = AIContentDetector(model_name=args.model)
    detector.generate_report(args.pdf_path, args.output)


if __name__ == "__main__":
    main()
