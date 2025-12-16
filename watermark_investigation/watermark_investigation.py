#!/usr/bin/env python3
"""
Watermark Investigation Script
Analyzes AI-edited images to find evidence of embedded watermarks.
"""

import json
import os
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import hashlib

# Base path for images
BASE_PATH = "/Users/aloshdenny/Downloads"

def load_image_pair(input_path, output_path):
    """Load an original and AI-edited image pair."""
    input_full = os.path.join(BASE_PATH, input_path)
    output_full = os.path.join(BASE_PATH, output_path)
    
    if not os.path.exists(input_full) or not os.path.exists(output_full):
        return None, None
    
    original = cv2.imread(input_full)
    edited = cv2.imread(output_full)
    return original, edited

def analyze_frequency_domain(img1, img2, name=""):
    """Analyze frequency domain differences - watermarks often hide in high frequencies."""
    if img1 is None or img2 is None:
        return {}
    
    results = {}
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize to same dimensions if needed
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Compute FFT
    f1 = np.fft.fft2(gray1.astype(float))
    f2 = np.fft.fft2(gray2.astype(float))
    
    # Shift zero frequency to center
    fshift1 = np.fft.fftshift(f1)
    fshift2 = np.fft.fftshift(f2)
    
    # Get magnitude spectrum
    mag1 = np.log(np.abs(fshift1) + 1)
    mag2 = np.log(np.abs(fshift2) + 1)
    
    # Compare high frequency components
    diff_mag = np.abs(mag2 - mag1)
    
    results['high_freq_diff_mean'] = float(np.mean(diff_mag))
    results['high_freq_diff_std'] = float(np.std(diff_mag))
    results['high_freq_diff_max'] = float(np.max(diff_mag))
    
    return results

def analyze_lsb_pattern(img, name=""):
    """Analyze Least Significant Bit patterns - common watermark hiding technique."""
    if img is None:
        return {}
    
    results = {}
    
    # Extract LSB for each channel
    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        channel = img[:, :, i]
        lsb = channel & 1  # Extract LSB
        
        # Check for patterns in LSB
        lsb_mean = np.mean(lsb)
        lsb_std = np.std(lsb)
        
        # In a natural image, LSB should be ~0.5 mean with high variance
        # Watermarked images might show deviations
        results[f'{channel_name}_lsb_mean'] = float(lsb_mean)
        results[f'{channel_name}_lsb_std'] = float(lsb_std)
        
        # Check for structured patterns using autocorrelation
        lsb_flat = lsb.flatten()[:10000]  # Sample
        autocorr = np.correlate(lsb_flat - 0.5, lsb_flat - 0.5, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Check for periodic patterns
        results[f'{channel_name}_lsb_autocorr_peak'] = float(np.max(autocorr[1:min(100, len(autocorr))]))
    
    return results

def analyze_dct_coefficients(img, name=""):
    """Analyze DCT coefficients - JPEG-based watermarks often modify DCT."""
    if img is None:
        return {}
    
    results = {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply DCT in 8x8 blocks (like JPEG)
    h, w = gray.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    gray = gray[:h, :w]
    
    dct_coeffs = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = gray[i:i+8, j:j+8].astype(float)
            dct_block = cv2.dct(block)
            dct_coeffs.append(dct_block)
    
    dct_coeffs = np.array(dct_coeffs)
    
    # Analyze specific DCT positions often used for watermarking
    # Middle frequencies are common targets
    mid_freq_positions = [(1,2), (2,1), (2,2), (3,1), (1,3)]
    
    for pos in mid_freq_positions:
        coeff_values = dct_coeffs[:, pos[0], pos[1]]
        results[f'dct_{pos[0]}_{pos[1]}_mean'] = float(np.mean(coeff_values))
        results[f'dct_{pos[0]}_{pos[1]}_std'] = float(np.std(coeff_values))
        
        # Check for quantization patterns (sign of modification)
        hist, _ = np.histogram(coeff_values, bins=50)
        entropy = -np.sum((hist/hist.sum() + 1e-10) * np.log2(hist/hist.sum() + 1e-10))
        results[f'dct_{pos[0]}_{pos[1]}_entropy'] = float(entropy)
    
    return results

def analyze_color_histogram_anomalies(img1, img2, name=""):
    """Check for systematic color modifications that might indicate watermarking."""
    if img1 is None or img2 is None:
        return {}
    
    results = {}
    
    # Resize if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256]).flatten()
        
        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Chi-square distance
        chi_sq = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        results[f'{channel_name}_hist_chi_sq'] = float(chi_sq)
        
        # Earth mover's distance approximation
        emd = np.sum(np.abs(np.cumsum(hist1) - np.cumsum(hist2)))
        results[f'{channel_name}_hist_emd'] = float(emd)
    
    return results

def check_metadata_watermarks(filepath):
    """Check EXIF and other metadata for watermark signatures."""
    results = {}
    full_path = os.path.join(BASE_PATH, filepath)
    
    if not os.path.exists(full_path):
        return results
    
    try:
        with Image.open(full_path) as img:
            # Get EXIF data
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if exif:
                results['has_exif'] = True
                results['exif_tags'] = list(exif.keys())
            else:
                results['has_exif'] = False
            
            # Get other info
            results['format'] = img.format
            results['mode'] = img.mode
            results['size'] = img.size
            
            # Check for ICC profile (can contain watermark)
            if 'icc_profile' in img.info:
                results['has_icc_profile'] = True
                results['icc_profile_size'] = len(img.info['icc_profile'])
            else:
                results['has_icc_profile'] = False
                
    except Exception as e:
        results['error'] = str(e)
    
    return results

def analyze_pixel_value_distribution(img, name=""):
    """Analyze pixel value distribution for anomalies."""
    if img is None:
        return {}
    
    results = {}
    
    # Check for unusual value concentrations
    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        channel = img[:, :, i].flatten()
        
        # Check LSB distribution
        lsb = channel % 2
        results[f'{channel_name}_lsb_ratio'] = float(np.mean(lsb))
        
        # Check for values that are multiples of specific numbers
        # Some watermarks use quantization
        for q in [2, 4, 8]:
            mod_vals = channel % q
            hist = np.bincount(mod_vals, minlength=q)
            uniformity = np.std(hist) / np.mean(hist)
            results[f'{channel_name}_mod{q}_uniformity'] = float(uniformity)
    
    return results

def compare_spatial_differences(img1, img2, name=""):
    """Analyze spatial differences between original and edited."""
    if img1 is None or img2 is None:
        return {}
    
    results = {}
    
    # Resize if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute difference
    diff = cv2.absdiff(img1, img2)
    
    # Analyze difference patterns
    results['diff_mean'] = float(np.mean(diff))
    results['diff_std'] = float(np.std(diff))
    results['diff_max'] = float(np.max(diff))
    
    # Check if differences are localized (edit) or global (watermark)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Divide into regions and check variance
    h, w = gray_diff.shape
    region_means = []
    for i in range(4):
        for j in range(4):
            region = gray_diff[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            region_means.append(np.mean(region))
    
    results['diff_region_variance'] = float(np.var(region_means))
    results['diff_region_mean'] = float(np.mean(region_means))
    
    return results

def detect_repeated_patterns(img, name=""):
    """Detect repeated patterns that might indicate watermarks."""
    if img is None:
        return {}
    
    results = {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use template matching with parts of the image
    # Check corners and edges for repeated patterns
    h, w = gray.shape
    
    # Extract corner template
    template_size = min(64, h//4, w//4)
    corners = [
        gray[:template_size, :template_size],  # Top-left
        gray[:template_size, -template_size:],  # Top-right
        gray[-template_size:, :template_size],  # Bottom-left
        gray[-template_size:, -template_size:]  # Bottom-right
    ]
    
    corner_names = ['TL', 'TR', 'BL', 'BR']
    
    for corner, corner_name in zip(corners, corner_names):
        # Check if this corner pattern appears elsewhere
        result = cv2.matchTemplate(gray, corner, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        locations = np.where(result >= threshold)
        results[f'{corner_name}_pattern_matches'] = len(locations[0])
    
    return results

def main():
    """Main investigation function."""
    print("=" * 80)
    print("WATERMARK INVESTIGATION REPORT")
    print("=" * 80)
    
    # Load pairs from JSONL
    pairs = []
    with open('/Users/aloshdenny/vscode/pairs.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Analyze first 20 pairs for investigation
                break
            pairs.append(json.loads(line))
    
    print(f"\nAnalyzing {len(pairs)} image pairs...\n")
    
    # Aggregate results
    all_original_results = []
    all_edited_results = []
    freq_differences = []
    
    for idx, pair in enumerate(pairs):
        input_path = pair['input_images'][0]
        output_path = pair['output_images'][0]
        
        original, edited = load_image_pair(input_path, output_path)
        
        if original is None or edited is None:
            print(f"Pair {idx}: Could not load images")
            continue
        
        print(f"\nPair {idx}: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        print("-" * 60)
        
        # Analyze frequency domain
        freq_results = analyze_frequency_domain(original, edited)
        if freq_results:
            freq_differences.append(freq_results)
            print(f"  Frequency Domain Difference: mean={freq_results['high_freq_diff_mean']:.4f}, max={freq_results['high_freq_diff_max']:.4f}")
        
        # Analyze LSB patterns
        lsb_orig = analyze_lsb_pattern(original, "original")
        lsb_edit = analyze_lsb_pattern(edited, "edited")
        
        print(f"  Original LSB means: R={lsb_orig.get('Red_lsb_mean', 0):.4f}, G={lsb_orig.get('Green_lsb_mean', 0):.4f}, B={lsb_orig.get('Blue_lsb_mean', 0):.4f}")
        print(f"  Edited LSB means:   R={lsb_edit.get('Red_lsb_mean', 0):.4f}, G={lsb_edit.get('Green_lsb_mean', 0):.4f}, B={lsb_edit.get('Blue_lsb_mean', 0):.4f}")
        
        # Analyze DCT coefficients
        dct_orig = analyze_dct_coefficients(original)
        dct_edit = analyze_dct_coefficients(edited)
        all_original_results.append({'lsb': lsb_orig, 'dct': dct_orig})
        all_edited_results.append({'lsb': lsb_edit, 'dct': dct_edit})
        
        # Check metadata
        meta_orig = check_metadata_watermarks(input_path)
        meta_edit = check_metadata_watermarks(output_path)
        
        if meta_edit.get('has_icc_profile'):
            print(f"  ⚠️  Edited image has ICC profile (size: {meta_edit.get('icc_profile_size')} bytes)")
        
        # Analyze spatial differences
        spatial = compare_spatial_differences(original, edited)
        print(f"  Spatial Difference: mean={spatial.get('diff_mean', 0):.2f}, region_variance={spatial.get('diff_region_variance', 0):.4f}")
        
        # Pixel distribution analysis
        pixel_orig = analyze_pixel_value_distribution(original)
        pixel_edit = analyze_pixel_value_distribution(edited)
        
        # Check for LSB anomalies (should be ~0.5 for natural images)
        lsb_anomaly = False
        for channel in ['Red', 'Green', 'Blue']:
            orig_lsb = pixel_orig.get(f'{channel}_lsb_ratio', 0.5)
            edit_lsb = pixel_edit.get(f'{channel}_lsb_ratio', 0.5)
            if abs(edit_lsb - 0.5) > 0.02:  # Deviation threshold
                lsb_anomaly = True
                print(f"  ⚠️  LSB anomaly in {channel} channel: {edit_lsb:.4f} (expected ~0.5)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY FINDINGS")
    print("=" * 80)
    
    if freq_differences:
        avg_freq_diff = np.mean([f['high_freq_diff_mean'] for f in freq_differences])
        print(f"\n1. FREQUENCY DOMAIN ANALYSIS:")
        print(f"   Average high-frequency difference: {avg_freq_diff:.4f}")
        print(f"   → Non-zero differences in frequency domain suggest spectral modifications")
    
    if all_edited_results:
        print(f"\n2. LSB (LEAST SIGNIFICANT BIT) ANALYSIS:")
        orig_lsb_means = [r['lsb'].get('Red_lsb_mean', 0.5) for r in all_original_results]
        edit_lsb_means = [r['lsb'].get('Red_lsb_mean', 0.5) for r in all_edited_results]
        print(f"   Original images avg LSB mean: {np.mean(orig_lsb_means):.4f}")
        print(f"   Edited images avg LSB mean: {np.mean(edit_lsb_means):.4f}")
        lsb_shift = abs(np.mean(edit_lsb_means) - np.mean(orig_lsb_means))
        if lsb_shift > 0.01:
            print(f"   ⚠️  EVIDENCE: Systematic LSB shift of {lsb_shift:.4f} detected!")
            print(f"   → This suggests LSB-based watermarking or steganographic modification")
    
    print(f"\n3. DCT COEFFICIENT ANALYSIS:")
    print(f"   DCT modifications in mid-frequency coefficients can indicate JPEG-domain watermarking")
    
    print(f"\n4. METADATA ANALYSIS:")
    print(f"   Checked for EXIF tags, ICC profiles that might carry watermark data")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FURTHER INVESTIGATION")
    print("=" * 80)
    print("""
    1. Use specialized watermark detection tools (e.g., StirTrace, GIMP analysis)
    2. Analyze bit planes visually (especially LSB plane)
    3. Check for invisible/robust watermarks using correlation attacks
    4. Examine JPEG quantization tables for modifications
    5. Use blind watermark detection algorithms
    6. Check for neural network watermarks (adversarial patterns)
    """)

if __name__ == "__main__":
    main()
