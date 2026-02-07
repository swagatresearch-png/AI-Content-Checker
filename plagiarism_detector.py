#!/usr/bin/env python3
"""
Plagiarism Detector for Academic Papers
Uses semantic similarity and web search to detect potential plagiarism in PDF documents
"""

import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict
import argparse
import requests
import time
from bs4 import BeautifulSoup
import re

class PlagiarismDetector:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the plagiarism detector
        
        Args:
            model_name: Hugging Face model for sentence embeddings
        """
        print("Loading semantic similarity model...")
        try:
            # Using sentence-transformers for semantic similarity
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
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
                    text += page.extract_text() + "\n"
                    if (page_num + 1) % 5 == 0:
                        print(f"Processed {page_num + 1} pages...")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        
        print(f"Extracted {len(text.split())} words")
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Basic sentence splitting
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        return sentences
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding using the model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def search_web_for_sentence(self, sentence: str, max_results: int = 3) -> List[Dict]:
        """
        Search the web for similar content (simplified version)
        Note: This is a basic implementation. For production, use proper APIs like Google Custom Search
        """
        # Clean the sentence for search
        search_query = sentence[:200]  # Limit query length
        
        results = []
        
        # Using a simple approach - in production, use proper search APIs
        # This is just a placeholder that shows the structure
        print(f"  Searching web for: {search_query[:80]}...")
        
        # NOTE: For actual plagiarism detection, you should:
        # 1. Use Google Custom Search API
        # 2. Use academic databases APIs (Google Scholar, PubMed, etc.)
        # 3. Use specialized plagiarism detection APIs
        
        # Placeholder for search results
        results.append({
            'title': 'Web search requires API setup',
            'url': 'N/A',
            'snippet': 'To enable web search, configure Google Custom Search API or similar service',
            'similarity': 0.0
        })
        
        return results
    
    def check_self_plagiarism(self, sentences: List[str], threshold: float = 0.85) -> List[Tuple]:
        """
        Check for repeated/duplicated content within the document (self-plagiarism)
        """
        print("\nChecking for self-plagiarism (internal duplication)...")
        duplicates = []
        
        # Get embeddings for all sentences
        print("Computing sentence embeddings...")
        embeddings = []
        for i, sentence in enumerate(sentences):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(sentences)} sentences...")
            embeddings.append(self.get_embedding(sentence))
        
        embeddings = np.vstack(embeddings)
        
        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find highly similar sentence pairs
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    duplicates.append((
                        i, j,
                        sentences[i],
                        sentences[j],
                        similarity
                    ))
        
        duplicates.sort(key=lambda x: x[4], reverse=True)
        return duplicates
    
    def analyze_document(self, text: str, check_web: bool = False) -> Dict:
        """Analyze document for plagiarism"""
        sentences = self.split_into_sentences(text)
        print(f"\nAnalyzing {len(sentences)} sentences...")
        
        results = {
            'total_sentences': len(sentences),
            'self_plagiarism': [],
            'web_matches': [],
            'statistics': {}
        }
        
        # Check self-plagiarism
        duplicates = self.check_self_plagiarism(sentences, threshold=0.85)
        results['self_plagiarism'] = duplicates
        
        # Check web plagiarism (optional, requires API setup)
        if check_web:
            print("\nChecking web sources (limited functionality without API)...")
            # Sample a few sentences to check
            sample_size = min(20, len(sentences))
            sampled_sentences = np.random.choice(sentences, sample_size, replace=False)
            
            for sentence in sampled_sentences:
                if len(sentence.split()) > 10:  # Only check substantial sentences
                    web_results = self.search_web_for_sentence(sentence)
                    if web_results:
                        results['web_matches'].append({
                            'sentence': sentence,
                            'matches': web_results
                        })
                    time.sleep(1)  # Rate limiting
        
        # Calculate statistics
        results['statistics'] = {
            'duplicate_pairs': len(duplicates),
            'high_similarity_rate': len(duplicates) / max(len(sentences), 1) * 100,
            'unique_sentences': len(sentences) - len(set(idx for dup in duplicates for idx in [dup[0], dup[1]]))
        }
        
        return results
    
    def generate_report(self, pdf_path: str, output_file: str = "plagiarism_report.txt", 
                       check_web: bool = False):
        """Generate a detailed plagiarism detection report"""
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("No text extracted from PDF!")
            return
        
        results = self.analyze_document(text, check_web)
        
        # Generate report
        print("\n" + "="*80)
        print("PLAGIARISM DETECTION REPORT")
        print("="*80)
        print(f"\nDocument: {pdf_path}")
        print(f"Total sentences analyzed: {results['total_sentences']}")
        print(f"Duplicate/similar pairs found: {results['statistics']['duplicate_pairs']}")
        print(f"High similarity rate: {results['statistics']['high_similarity_rate']:.2f}%")
        
        # Determine verdict
        dup_rate = results['statistics']['high_similarity_rate']
        if dup_rate > 10:
            verdict = "HIGH risk - Significant internal duplication detected"
        elif dup_rate > 5:
            verdict = "MODERATE risk - Some internal duplication found"
        elif dup_rate > 2:
            verdict = "LOW-MODERATE risk - Minor duplication present"
        else:
            verdict = "LOW risk - Minimal duplication detected"
        
        print(f"Verdict: {verdict}")
        
        # Write detailed report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PLAGIARISM DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Document: {pdf_path}\n")
            f.write(f"Total sentences analyzed: {results['total_sentences']}\n")
            f.write(f"Duplicate/similar pairs found: {results['statistics']['duplicate_pairs']}\n")
            f.write(f"High similarity rate: {results['statistics']['high_similarity_rate']:.2f}%\n")
            f.write(f"Verdict: {verdict}\n\n")
            
            if results['self_plagiarism']:
                f.write("\n" + "-"*80 + "\n")
                f.write("INTERNAL DUPLICATION ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                f.write("Top 20 most similar sentence pairs:\n\n")
                
                for i, (idx1, idx2, sent1, sent2, similarity) in enumerate(results['self_plagiarism'][:20], 1):
                    f.write(f"{i}. Similarity: {similarity*100:.2f}%\n")
                    f.write(f"   Sentence {idx1}: {sent1[:200]}...\n")
                    f.write(f"   Sentence {idx2}: {sent2[:200]}...\n\n")
            
            if results['web_matches']:
                f.write("\n" + "-"*80 + "\n")
                f.write("WEB SOURCE ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                for match in results['web_matches']:
                    f.write(f"Sentence: {match['sentence'][:200]}...\n")
                    f.write("Potential matches:\n")
                    for web_result in match['matches']:
                        f.write(f"  - {web_result['title']}\n")
                        f.write(f"    URL: {web_result['url']}\n")
                        f.write(f"    Snippet: {web_result['snippet']}\n\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n\n")
            
            if dup_rate > 5:
                f.write("- Review highlighted duplicate sections and ensure proper paraphrasing\n")
                f.write("- Check if repeated content is necessary or can be referenced instead\n")
            
            f.write("- For comprehensive plagiarism checking, use:\n")
            f.write("  * Turnitin (institutional access)\n")
            f.write("  * iThenticate (for journal submissions)\n")
            f.write("  * Grammarly Premium (includes plagiarism checker)\n")
            f.write("  * Google Scholar search for specific phrases\n")
            f.write("\n- This tool checks internal duplication. External plagiarism requires\n")
            f.write("  access to academic databases and search APIs.\n")
        
        print(f"\nDetailed report saved to: {output_file}")
        
        if results['self_plagiarism']:
            print("\nTop 5 most similar sentence pairs:")
            for i, (idx1, idx2, sent1, sent2, similarity) in enumerate(results['self_plagiarism'][:5], 1):
                print(f"\n{i}. Similarity: {similarity*100:.2f}%")
                print(f"   Sentence {idx1}: {sent1[:150]}...")
                print(f"   Sentence {idx2}: {sent2[:150]}...")


def main():
    parser = argparse.ArgumentParser(description='Detect plagiarism in PDF documents')
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file to analyze')
    parser.add_argument('--output', type=str, default='plagiarism_report.txt',
                       help='Output report file name (default: plagiarism_report.txt)')
    parser.add_argument('--check-web', action='store_true',
                       help='Enable web search (requires API setup)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Hugging Face model for embeddings')
    
    args = parser.parse_args()
    
    detector = PlagiarismDetector(model_name=args.model)
    detector.generate_report(args.pdf_path, args.output, args.check_web)


if __name__ == "__main__":
    main()
