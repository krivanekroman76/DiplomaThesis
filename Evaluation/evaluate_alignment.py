"""
================================================================================
LYRICS ALIGNMENT & EVALUATION TOOL (Dual-Extraction & 5-Tier Normalization)
================================================================================
AUTHOR: Bc. Roman Křivánek
REQUIRED: A transcription text file (containing 'Timestamps:') and a ground truth
          lyrics text file.

USAGE GUIDE:
    python Evaluation/evaluate_alignment.py --transcription path/to/transcription.txt --lyrics path/to/lyrics.txt
================================================================================
"""

import argparse
import re
import unicodedata
from pathlib import Path
import difflib
import jiwer

class AlignmentEvaluator:
    def __init__(self, transcription_path: str, lyrics_path: str):
        self.transcription_path = Path(transcription_path)
        self.lyrics_path = Path(lyrics_path)

    # --- 5 CLEANING TIERS ---
    def tier_1_minimalist(self, text: str) -> str:
        """TIER 1: Just lowercase. Keeps all raw punctuation and accents."""
        text = text.lower()
        return re.sub(r'\s+', ' ', text).strip()

    def tier_2_punctuation_strip(self, text: str) -> str:
        """TIER 2: Lowercase, removes basic punctuation, KEEPS accents, apostrophes, hyphens."""
        text = text.lower()
        text = re.sub(r'[^\w\s\'-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_3_boundary_split(self, text: str) -> str:
        """TIER 3: Keeps accents, but turns apostrophes and hyphens into spaces (j'aime -> j aime)."""
        text = text.lower()
        text = text.replace("'", " ").replace("-", " ")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_4_acoustic_robust(self, text: str) -> str:
        """TIER 4: Removes accents/diacritics AND turns apostrophes/hyphens into spaces."""
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.lower().replace("'", " ").replace("-", " ")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_5_aggressive_squash(self, text: str) -> str:
        """TIER 5: Removes accents, and REMOVES apostrophes/hyphens entirely (j'aime -> jaime)."""
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.lower().replace("'", "").replace("-", "")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    # --- EXTRACTION METHODS ---
    def extract_continuous_text(self, raw_text: str) -> str:
        if "Transcription (Model:" in raw_text and "Timestamps:" in raw_text:
            parts = raw_text.split("\n\nTimestamps:")[0]
            content = parts.split("):\n", 1)[-1]
            return content.strip()
        return raw_text.strip()

    def extract_timestamped_text(self, raw_text: str) -> str:
        if "Timestamps:" in raw_text:
            timestamp_section = raw_text.split("Timestamps:")[1].strip()
            lines = timestamp_section.split('\n')
            extracted_lines = []
            for line in lines:
                cleaned_line = re.sub(r'^\d+\.\d+s\s*-\s*\d+\.\d+s:\s*', '', line.strip())
                if cleaned_line:
                    extracted_lines.append(cleaned_line)
            return " ".join(extracted_lines)
        return raw_text.strip()

    # --- VISUALIZATION ---
    def generate_alignment_visualization(self, ref_text: str, hyp_text: str, chunk_size: int = 6):
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        aligned_ref, aligned_hyp, aligned_ops = [], [], []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for r, h in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                    aligned_ref.append(r); aligned_hyp.append(h); aligned_ops.append(" ")
            elif tag == 'replace':
                r_chunk, h_chunk = ref_words[i1:i2], hyp_words[j1:j2]
                for r, h in zip(r_chunk, h_chunk):
                    aligned_ref.append(r); aligned_hyp.append(h); aligned_ops.append("SUB")
                if len(r_chunk) > len(h_chunk):
                    for r in r_chunk[len(h_chunk):]:
                        aligned_ref.append(r); aligned_hyp.append("*" * len(r)); aligned_ops.append("DEL")
                elif len(h_chunk) > len(r_chunk):
                    for h in h_chunk[len(r_chunk):]:
                        aligned_ref.append("*" * len(h)); aligned_hyp.append(h); aligned_ops.append("INS")
            elif tag == 'delete':
                for r in ref_words[i1:i2]:
                    aligned_ref.append(r); aligned_hyp.append("*" * len(r)); aligned_ops.append("DEL")
            elif tag == 'insert':
                for h in hyp_words[j1:j2]:
                    aligned_ref.append("*" * len(h)); aligned_hyp.append(h); aligned_ops.append("INS")

        for i in range(0, len(aligned_ops), chunk_size):
            r_chunk = aligned_ref[i:i+chunk_size]
            h_chunk = aligned_hyp[i:i+chunk_size]
            o_chunk = aligned_ops[i:i+chunk_size]
            
            widths = [max(len(r), len(h), len(o)) for r, h, o in zip(r_chunk, h_chunk, o_chunk)]
            
            ref_row = "REF: " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(r_chunk))
            hyp_row = "HYP: " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(h_chunk))
            op_row  = "OP:  " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(o_chunk))
            
            print(ref_row)
            print(hyp_row)
            print(op_row)
            print("-" * 80)

    def evaluate(self):
        print("="*80)
        print(" 5-TIER TRANSCRIPTION EVALUATION (Acoustic vs Linguistic Analysis)")
        print("="*80)

        try:
            raw_transcription = self.transcription_path.read_text(encoding="utf-8")
            raw_lyrics = self.lyrics_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            print(f"[!] Error reading files: {e}")
            return

        hyp_cont_raw = self.extract_continuous_text(raw_transcription)
        hyp_time_raw = self.extract_timestamped_text(raw_transcription)
        
        # Calculate Metrics across all 5 tiers
        tiers = [
            ("1. Minimalist (All Punct/Accents)", self.tier_1_minimalist),
            ("2. Strip Punct (Keep Apostrophes)", self.tier_2_punctuation_strip),
            ("3. Boundary Split (j'aime -> j aime)", self.tier_3_boundary_split),
            ("4. Acoustic Robust (No Accents/Split)", self.tier_4_acoustic_robust),
            ("5. Aggressive Squash (j'aime->jaime)", self.tier_5_aggressive_squash)
        ]

        print(f"\n[WER COMPARISON MATRIX]")
        print(f"{'Cleaning Tier':<40} | {'Continuous Text':<18} | {'Timestamped Text':<18}")
        print("-" * 82)

        robust_metrics = None # Store Tier 4 for the error breakdown

        for tier_name, clean_func in tiers:
            ref_clean = clean_func(raw_lyrics)
            hyp_cont_clean = clean_func(hyp_cont_raw)
            hyp_time_clean = clean_func(hyp_time_raw)

            if not ref_clean:
                continue

            meas_cont = jiwer.compute_measures(ref_clean, hyp_cont_clean)
            meas_time = jiwer.compute_measures(ref_clean, hyp_time_clean)

            if "Acoustic Robust" in tier_name:
                robust_metrics = (meas_cont, meas_time)

            print(f"{tier_name:<40} | {meas_cont['wer']*100:>6.2f}%            | {meas_time['wer']*100:>6.2f}%")
        print("-" * 82)
        
        if robust_metrics:
            meas_cont_R, meas_time_R = robust_metrics
            print("\n[ERROR BREAKDOWN (Using Tier 4: Acoustic Robust Baseline)]")
            print(f"{'Metric':<15} | {'Continuous Text':<20} | {'Timestamped Text':<20}")
            print("-" * 62)
            print(f"{'Substitutions':<15} | {meas_cont_R['substitutions']:<20} | {meas_time_R['substitutions']:<20}")
            print(f"{'Deletions':<15} | {meas_cont_R['deletions']:<20} | {meas_time_R['deletions']:<20}")
            print(f"{'Insertions':<15} | {meas_cont_R['insertions']:<20} | {meas_time_R['insertions']:<20}")

        print("\n" + "="*80)
        print(" VISUAL ALIGNMENT (Timestamped Text - Tier 2: Strip Punct)")
        print(" REF = Ground Truth Lyrics | HYP = Model Output")
        print("="*80)
        
        # Visualize using Tier 2 so you can see if apostrophes are the cause of Substitutions/Insertions
        self.generate_alignment_visualization(
            self.tier_2_punctuation_strip(raw_lyrics), 
            self.tier_2_punctuation_strip(hyp_time_raw)
        )
            
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transcription accuracy against ground truth lyrics.")
    parser.add_argument("--transcription", type=str, required=True, help="Path to the model output text file.")
    parser.add_argument("--lyrics", type=str, required=True, help="Path to the ground truth lyrics text file.")
    
    args = parser.parse_args()
    
    evaluator = AlignmentEvaluator(args.transcription, args.lyrics)
    evaluator.evaluate()