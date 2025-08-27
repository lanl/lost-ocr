"""
Levenshtein Distance Analysis by Document Degradation Level

This module compares OCR quality between different systems (LLaMA vs Nougat) by calculating
Levenshtein distance between their outputs. It analyzes how document distortion affects
OCR accuracy across different degradation levels, excluding slideshow pages for cleaner results.

Key Metrics:
- Levenshtein distance: Character-level edit distance between OCR outputs
- Degradation levels: 0 (pristine) to 3 (high distortion)
- Filtering: Excludes slideshow pages that may skew results

Author: Alex Most
Date: 2025
Paper: Lost in OCR Translation (arXiv:2505.05666)
"""

import os
import pandas as pd
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Directory structure for different distortion levels
root_dirs = ["easy", "medium", "hard"]

# CSV files containing metadata
distortion_csv = "pdf_labels.csv"  # Contains distortion scores for each page
slides_csv = "Expanded_Pages.csv"  # Contains page metadata including slideshow flags

# ============================================================================
# STEP 1: LOAD DISTORTION SCORES
# ============================================================================

print("📊 Loading distortion scores from pdf_labels.csv")

# Read the CSV containing distortion labels
df = pd.read_csv(distortion_csv)

# Strip whitespace from column names to avoid matching issues
df.columns = df.columns.str.strip()

# Identify relevant columns by position
# These column indices are specific to the dataset structure
filename_col = df.columns[2]    # Column C: Filename
distortion_col = df.columns[3]  # Column D: Distortion score
page_id_col = df.columns[7]     # Column H: Page identifier

# Remove duplicate page entries, keeping first occurrence
# This ensures each page has only one distortion score
df_unique = df.drop_duplicates(subset=page_id_col)

# Create dictionary mapping page_id -> distortion_score
# Only include pages with valid (non-null) values
distortion_scores = {
    row[page_id_col].strip(): row[distortion_col]
    for _, row in df_unique.iterrows()
    if pd.notnull(row[page_id_col]) and pd.notnull(row[distortion_col])
}

print(f"✅ Loaded distortion scores for {len(distortion_scores)} pages")

# ============================================================================
# STEP 2: LOAD SLIDESHOW FILTER
# ============================================================================

print("🎯 Loading slideshow filter from expanded_pages.csv")

# Read the CSV containing page metadata
slides_df = pd.read_csv(slides_csv)

# Strip whitespace from column names
slides_df.columns = slides_df.columns.str.strip()

# Identify relevant columns
title_col = slides_df.columns[0]      # Column A: Document title
page_num_col = slides_df.columns[2]   # Column C: Page number
slide_flag_col = slides_df.columns[10] # Column K: Slideshow flag (0=normal, 1=slideshow)

# Create set of allowed pages (non-slideshow pages only)
# Format: "document_title-page_number"
# This filtering is important because slideshow pages often have different
# characteristics that could skew OCR comparison results
allowed_pages = {
    f"{row[title_col].strip()}-{int(row[page_num_col])}"
    for _, row in slides_df.iterrows()
    if pd.notnull(row[title_col]) and 
       pd.notnull(row[page_num_col]) and 
       row[slide_flag_col] == 0  # Only include non-slideshow pages
}

print(f"✅ Identified {len(allowed_pages)} non-slideshow pages for analysis")

# ============================================================================
# STEP 3: COMPARISON LOOP - CALCULATE LEVENSHTEIN DISTANCES
# ============================================================================

print("🔍 Scanning directories and computing Levenshtein distances")

# Dictionary to store distances by degradation level
dist_by_level = defaultdict(list)

def get_level(score):
    """
    Map continuous distortion score to discrete level.
    
    This function categorizes documents into 4 degradation levels based on
    their distortion scores, allowing for grouped analysis of OCR performance.
    
    Args:
        score (float): Distortion score from pdf_labels.csv
    
    Returns:
        int: Degradation level (0-3)
        
    Level Mapping:
        - Level 0: score < 1 (pristine/minimal distortion)
        - Level 1: 1 ≤ score < 2 (low distortion)
        - Level 2: 2 ≤ score < 3 (medium distortion)
        - Level 3: score ≥ 3 (high distortion)
    """
    if score < 1: 
        return 0  # Pristine
    elif score < 2: 
        return 1  # Low distortion
    elif score < 3: 
        return 2  # Medium distortion
    else: 
        return 3  # High distortion

# Tracking counters for reporting
missing_pages = 0    # Pages where one OCR system is missing output
compared_pages = 0   # Successfully compared page pairs

# Iterate through each difficulty directory (easy, medium, hard)
for base_dir in root_dirs:
    # Check each document folder within the difficulty level
    for doc_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, doc_folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Define paths to OCR output directories
        llama_dir = os.path.join(folder_path, "llama_txt")    # LLaMA OCR outputs
        nougat_dir = os.path.join(folder_path, "mmd")         # Nougat OCR outputs
        
        # Skip if either OCR directory is missing
        if not os.path.isdir(llama_dir) or not os.path.isdir(nougat_dir):
            continue

        # Process each LLaMA OCR output file
        for fname in os.listdir(llama_dir):
            if not fname.endswith(".txt"):
                continue

            # Extract page ID from filename
            page_id = fname.replace(".txt", "")
            
            # Skip if page doesn't have distortion score or is a slideshow page
            if page_id not in distortion_scores or page_id not in allowed_pages:
                continue

            # Construct paths to both OCR outputs
            # Note: Nougat uses .pdf.mmd extension pattern
            mmd_name = f"{page_id}.pdf.mmd"
            llama_path = os.path.join(llama_dir, fname)
            nougat_path = os.path.join(nougat_dir, mmd_name)

            # Check if corresponding Nougat output exists
            if not os.path.exists(nougat_path):
                missing_pages += 1
                continue

            # Read both OCR outputs with error handling
            try:
                with open(llama_path, 'r', encoding='utf-8') as f1:
                    llama_text = f1.read().strip()
                with open(nougat_path, 'r', encoding='utf-8') as f2:
                    nougat_text = f2.read().strip()
            except Exception as e:
                print(f"⚠️  Skipping {page_id} due to read error: {e}")
                continue

            # Calculate Levenshtein distance between OCR outputs
            # This measures the minimum number of single-character edits
            # (insertions, deletions, substitutions) needed to transform
            # one text into the other
            dist = levenshtein_distance(llama_text, nougat_text)
            
            # Categorize by degradation level and store distance
            level = get_level(distortion_scores[page_id])
            dist_by_level[level].append(dist)
            compared_pages += 1

# ============================================================================
# STEP 4: GENERATE REPORT
# ============================================================================

print("\n📈 Levenshtein Distance Averages by Degradation Level (slides excluded)")
print("=" * 60)

# Calculate and display statistics for each degradation level
for level in sorted(dist_by_level.keys()):
    distances = dist_by_level[level]
    
    # Calculate average Levenshtein distance for this level
    avg = sum(distances) / len(distances) if distances else 0
    
    # Define level descriptions for clarity
    level_names = {
        0: "Pristine (< 1.0)",
        1: "Low (1.0-2.0)",
        2: "Medium (2.0-3.0)",
        3: "High (≥ 3.0)"
    }
    
    print(f"\n📊 Level {level} - {level_names[level]}")
    print(f"   Pages analyzed: {len(distances)}")
    print(f"   Average Levenshtein distance: {avg:.2f}")
    
    # Additional statistics for more insight
    if distances:
        print(f"   Min distance: {min(distances)}")
        print(f"   Max distance: {max(distances)}")
        print(f"   Median distance: {sorted(distances)[len(distances)//2]}")

# Summary statistics
print("\n" + "=" * 60)
print("📊 SUMMARY STATISTICS")
print("=" * 60)
print(f"✅ Total pages compared: {compared_pages}")
print(f"⚠️  Pages skipped due to missing files: {missing_pages}")

# Calculate overall average across all levels
all_distances = []
for distances in dist_by_level.values():
    all_distances.extend(distances)

if all_distances:
    overall_avg = sum(all_distances) / len(all_distances)
    print(f"📈 Overall average Levenshtein distance: {overall_avg:.2f}")

# Provide interpretation of results
print("\n💡 INTERPRETATION GUIDE:")
print("Lower Levenshtein distance = Better OCR agreement")
print("Higher distance at higher degradation levels indicates OCR quality degradation")
print("Excluding slideshows provides cleaner comparison of standard document pages")
