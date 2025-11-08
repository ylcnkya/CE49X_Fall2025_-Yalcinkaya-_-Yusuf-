"""
Lab 4: Statistical Analysis - Descriptive Statistics and Probability Distributions
Reads datasets from the same directory as the script.
Saves PNG visualizations and a summary report in the script directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, binom, poisson, expon, uniform
import math

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def load_data(file_name):
    """Load dataset from the same directory as the script. Returns DataFrame or raises error."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df

def calculate_descriptive_stats(series):
    """Calculate descriptive statistics for a pandas Series."""
    s = series.dropna()
    stats_dict = {}
    stats_dict['count'] = int(s.count())
    stats_dict['mean'] = float(s.mean())
    stats_dict['median'] = float(s.median())
    # mode: return first mode if multiple
    modes = s.mode()
    stats_dict['mode'] = float(modes.iloc[0]) if not modes.empty else None
    stats_dict['std'] = float(s.std(ddof=1))
    stats_dict['var'] = float(s.var(ddof=1))
    stats_dict['min'] = float(s.min())
    stats_dict['max'] = float(s.max())
    stats_dict['range'] = float(s.max() - s.min())
    stats_dict['q1'] = float(s.quantile(0.25))
    stats_dict['q3'] = float(s.quantile(0.75))
    stats_dict['iqr'] = float(s.quantile(0.75) - s.quantile(0.25))
    stats_dict['skewness'] = float(stats.skew(s, bias=False))
    stats_dict['kurtosis'] = float(stats.kurtosis(s, fisher=True, bias=False))
    stats_dict['percentiles'] = {int(p): float(s.quantile(p/100.0)) for p in [5,25,50,75,95]}
    return stats_dict

def plot_histogram_with_normal(series, title, filename):
    """Histogram + fitted normal overlay, marks mean/median/mode and ±1/2/3 sigma regions."""
    s = series.dropna()
    mu, sigma = s.mean(), s.std(ddof=1)
    plt.figure(figsize=(10,6))
    n, bins, patches = plt.hist(s, bins=20, density=True, alpha=0.6, edgecolor='black')
    # Normal curve using sample mu/sigma
    x = np.linspace(s.min(), s.max(), 200)
    plt.plot(x, norm.pdf(x, mu, sigma), lw=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
    # Markers
    plt.axvline(mu, color='k', linestyle='--', label=f'Mean = {mu:.2f}')
    plt.axvline(np.median(s), color='g', linestyle='-.', label=f'Median = {np.median(s):.2f}')
    modes = s.mode()
    if not modes.empty:
        plt.axvline(modes.iloc[0], color='r', linestyle=':', label=f'Mode = {modes.iloc[0]:.2f}')
    # ± sigma regions shading
    for k in [1,2,3]:
        plt.axvline(mu + k*sigma, color='grey', alpha=0.6 if k==1 else 0.2, linestyle='--')
        plt.axvline(mu - k*sigma, color='grey', alpha=0.6 if k==1 else 0.2, linestyle='--')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Strength (MPa)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    # Legend'i sağ üstte, grafik dışına taşı
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    return out

def plot_boxplot_by_group(df, value_col, group_col, filename):
    """Create an improved boxplot comparing groups with better visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    groups = []
    labels = []
    group_stats = {}
    
    for name, group in df.groupby(group_col):
        values = group[value_col].dropna().values
        if len(values) > 0:
            groups.append(values)
            labels.append(str(name))
            # Calculate statistics for annotation
            group_stats[str(name)] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values, ddof=1),
                'n': len(values)
            }
    
    if len(groups) == 0:
        plt.close()
        return None
    
    # Create boxplot with better styling
    box_plot = ax.boxplot(groups, labels=labels, patch_artist=True, 
                          medianprops=dict(color='black', linewidth=2.5),
                          boxprops=dict(linewidth=2),
                          whiskerprops=dict(linewidth=2),
                          capprops=dict(linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', 
                                        markersize=6, alpha=0.6, markeredgecolor='darkred'))
    
    # Color each box differently with a professional color palette
    # Use a softer, more professional color scheme
    if len(groups) <= 6:
        # Use distinct colors for small number of groups
        color_list = ['#4A90E2', '#50C878', '#FF6B6B', '#FFA07A', '#9370DB', '#20B2AA']
        colors = color_list[:len(groups)]
    else:
        # Use a colormap for many groups
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(groups)))
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    
    # Add mean markers
    means = [group_stats[label]['mean'] for label in labels]
    ax.plot(range(1, len(labels) + 1), means, 'D', color='blue', 
           markersize=10, label='Mean', zorder=3, markeredgecolor='darkblue', markeredgewidth=1.5)
    
    # Improve labels and title
    ax.set_xlabel(group_col.replace('_', ' ').title(), fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(f'Comparison of {value_col.replace("_", " ").title()} by {group_col.replace("_", " ").title()}', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels if they're long
    if any(len(label) > 10 for label in labels):
        plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Add statistics text box (more compact)
    if len(labels) <= 5:  # Only show stats box if not too many groups
        stats_text = 'Statistics:\n'
        for label in labels:
            stats = group_stats[label]
            # Truncate label if too long
            display_label = label[:15] + '...' if len(label) > 15 else label
            stats_text += f'{display_label}:\n'
            stats_text += f'  n={stats["n"]}, μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}\n'
        
        # Place statistics box in upper left
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8.5,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85, edgecolor='navy', linewidth=1.5),
               family='monospace')
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out

def plot_probability_distributions(filename):
    """Create a composite figure demonstrating discrete and continuous distributions."""
    fig, axs = plt.subplots(2,3, figsize=(14,8))
    axs = axs.flatten()
    # Bernoulli p=0.1 PMF
    p = 0.1
    probs = [1-p, p]
    axs[0].bar([0,1], probs, edgecolor='black')
    axs[0].set_title('Bernoulli PMF (p=0.1)')
    axs[0].set_xticks([0,1])
    # Binomial n=100 p=0.05 PMF around mean
    n, p = 100, 0.05
    k = np.arange(0, 16)
    axs[1].bar(k, binom.pmf(k, n, p), edgecolor='black')
    axs[1].set_title(f'Binomial PMF (n={n}, p={p})')
    # Poisson lambda=10
    lam = 10
    k = np.arange(0, 25)
    axs[2].bar(k, poisson.pmf(k, lam), edgecolor='black')
    axs[2].set_title(f'Poisson PMF (λ={lam})')
    # Uniform continuous PDF on [0,1]
    x = np.linspace(0,1,200)
    axs[3].plot(x, uniform.pdf(x, 0, 1))
    axs[3].set_title('Uniform(0,1) PDF')
    # Normal PDF example
    x = np.linspace(-4,4,400)
    axs[4].plot(x, norm.pdf(x, 0, 1))
    axs[4].set_title('Standard Normal PDF')
    # Exponential PDF mean=1000 -> scale=1000
    x = np.linspace(0, 5000, 400)
    axs[5].plot(x, expon.pdf(x, scale=1000))
    axs[5].set_title('Exponential PDF (mean=1000)')
    for ax in axs:
        ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out)
    plt.close()
    return out

def fit_normal_and_compare(series, filename):
    """Fit normal distribution to the series, plot overlay and generate synthetic sample comparison."""
    s = series.dropna()
    mu_hat, sigma_hat = s.mean(), s.std(ddof=1)
    x = np.linspace(s.min(), s.max(), 300)
    plt.figure(figsize=(10,6))
    plt.hist(s, bins=20, density=True, alpha=0.6, edgecolor='black', label='Data histogram')
    plt.plot(x, norm.pdf(x, mu_hat, sigma_hat), lw=2, label=f'Fitted Normal (μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
    # synthetic
    synt = np.random.normal(loc=mu_hat, scale=sigma_hat, size=len(s))
    plt.hist(synt, bins=20, density=True, alpha=0.4, edgecolor='none', label='Synthetic (fitted)')
    plt.title('Data vs Fitted Normal Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Strength (MPa)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    # Legend'i sağ üstte, grafik dışına taşı
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True, fancybox=True, shadow=True)
    out = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    # Return fitted params and synthetic sample stats
    return {'mu_hat': float(mu_hat), 'sigma_hat': float(sigma_hat),
            'synt_mean': float(np.mean(synt)), 'synt_std': float(np.std(synt, ddof=1)), 'file': out}

def calculate_probability_binomial(n, p, k):
    """Return P(X=k) and P(X<=k) for Binomial(n,p)."""
    pmf_k = binom.pmf(k, n, p)
    cdf_k = binom.cdf(k, n, p)
    return float(pmf_k), float(cdf_k)

def calculate_probability_poisson(lam, k):
    """Return P(X=k) and P(X>k) for Poisson(lambda)."""
    pmf_k = poisson.pmf(k, lam)
    cdf_k = poisson.cdf(k, lam)
    return float(pmf_k), float(1 - cdf_k)

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """Return P(X <= x_upper) - P(X <= x_lower) for Normal(mean,std). Use None for open bounds."""
    if x_lower is None and x_upper is None:
        return None
    if x_lower is None:
        return float(norm.cdf(x_upper, loc=mean, scale=std))
    if x_upper is None:
        return float(1 - norm.cdf(x_lower, loc=mean, scale=std))
    return float(norm.cdf(x_upper, loc=mean, scale=std) - norm.cdf(x_lower, loc=mean, scale=std))

def calculate_probability_exponential(mean, x):
    """Return P(X <= x) for Exponential(mean)."""
    # scale = mean
    return float(expon.cdf(x, scale=mean))

def apply_bayes_theorem(prior, sensitivity, specificity):
    """Calculate posterior P(Damage | Positive) using Bayes' theorem."""
    # prior = P(Damage)
    # sensitivity = P(Pos | Damage)
    # specificity = P(Neg | NoDamage)
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
    posterior = (sensitivity * prior) / p_pos if p_pos > 0 else None
    return float(posterior), float(p_pos)

def plot_probability_tree(prior, sensitivity, specificity, filename):
    """Draw a professional probability tree diagram with clear structure and colors."""
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Calculate all probabilities
    p_damage = prior
    p_no_damage = 1 - prior
    p_pos_given_damage = sensitivity
    p_neg_given_damage = 1 - sensitivity
    p_pos_given_no_damage = 1 - specificity  # False positive rate
    p_neg_given_no_damage = specificity
    
    # Joint probabilities
    p_damage_and_pos = p_damage * p_pos_given_damage
    p_damage_and_neg = p_damage * p_neg_given_damage
    p_no_damage_and_pos = p_no_damage * p_pos_given_no_damage
    p_no_damage_and_neg = p_no_damage * p_neg_given_no_damage
    
    # Total probability of positive test
    p_pos_total = p_damage_and_pos + p_no_damage_and_pos
    p_neg_total = p_damage_and_neg + p_no_damage_and_neg
    
    # Posterior probability (Bayes result)
    posterior = p_damage_and_pos / p_pos_total if p_pos_total > 0 else 0
    
    # Define node positions (using better coordinate system)
    # Root node (center-left)
    root_x, root_y = 1.5, 5
    
    # Level 1: Damage states
    level1_x = 3.5
    damage_y = 7
    no_damage_y = 3
    
    # Level 2: Test results
    level2_x = 6.5
    damage_pos_y = 7.8
    damage_neg_y = 6.2
    no_damage_pos_y = 3.8
    no_damage_neg_y = 2.2
    
    # Final results
    result_x = 8.5
    
    # Draw root node (softer blue)
    root_rect = plt.Rectangle((root_x-0.6, root_y-0.5), 1.2, 1, 
                              facecolor='#6C7A89', edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(root_rect)
    ax.text(root_x, root_y, 'Population\n(Start)', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white', zorder=4)
    
    # Draw Level 1: Damage branches (softer colors)
    # Damage branch
    ax.plot([root_x+0.6, level1_x-0.5], [root_y+0.2, damage_y-0.3], 
            'k-', linewidth=2.5, zorder=1)
    damage_rect = plt.Rectangle((level1_x-0.7, damage_y-0.5), 1.4, 1, 
                                facecolor='#A8A8A8', edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(damage_rect)
    ax.text(level1_x, damage_y, f'Damage\nP = {p_damage:.3f}', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            color='black', zorder=4)
    # Probability label on branch (simpler, no colored background)
    ax.text((root_x+level1_x)/2, (root_y+damage_y)/2+0.3, f'P(Damage) = {p_damage:.3f}',
            fontsize=10, style='italic', color='black', weight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # No Damage branch
    ax.plot([root_x+0.6, level1_x-0.5], [root_y-0.2, no_damage_y+0.3], 
            'k-', linewidth=2.5, zorder=1)
    no_damage_rect = plt.Rectangle((level1_x-0.7, no_damage_y-0.5), 1.4, 1, 
                                   facecolor='#D3D3D3', edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(no_damage_rect)
    ax.text(level1_x, no_damage_y, f'No Damage\nP = {p_no_damage:.3f}', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            color='black', zorder=4)
    # Probability label on branch
    ax.text((root_x+level1_x)/2, (root_y+no_damage_y)/2-0.3, f'P(NoDamage) = {p_no_damage:.3f}',
            fontsize=10, style='italic', color='black', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # Draw Level 2: Test results from Damage (neutral colors)
    # Damage -> Positive
    ax.plot([level1_x+0.7, level2_x-0.5], [damage_y+0.2, damage_pos_y-0.3], 
            'k-', linewidth=2, zorder=1)
    pos_damage_rect = plt.Rectangle((level2_x-0.6, damage_pos_y-0.4), 1.2, 0.8, 
                                    facecolor='#E8E8E8', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(pos_damage_rect)
    ax.text(level2_x, damage_pos_y, f'Positive\nP = {p_damage_and_pos:.4f}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='black', zorder=4)
    ax.text((level1_x+level2_x)/2, damage_pos_y+0.35, f'P(Pos|Damage) = {p_pos_given_damage:.2f}',
            fontsize=9, style='italic', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # Damage -> Negative
    ax.plot([level1_x+0.7, level2_x-0.5], [damage_y-0.2, damage_neg_y+0.3], 
            'k--', linewidth=2, zorder=1)
    neg_damage_rect = plt.Rectangle((level2_x-0.6, damage_neg_y-0.4), 1.2, 0.8, 
                                    facecolor='#E8E8E8', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(neg_damage_rect)
    ax.text(level2_x, damage_neg_y, f'Negative\nP = {p_damage_and_neg:.4f}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='black', zorder=4)
    ax.text((level1_x+level2_x)/2, damage_neg_y-0.35, f'P(Neg|Damage) = {p_neg_given_damage:.2f}',
            fontsize=9, style='italic', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # Draw Level 2: Test results from No Damage (neutral colors)
    # No Damage -> Positive (False Positive)
    ax.plot([level1_x+0.7, level2_x-0.5], [no_damage_y+0.2, no_damage_pos_y-0.3], 
            'k-', linewidth=2, zorder=1)
    pos_no_damage_rect = plt.Rectangle((level2_x-0.6, no_damage_pos_y-0.4), 1.2, 0.8, 
                                       facecolor='#F0F0F0', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(pos_no_damage_rect)
    ax.text(level2_x, no_damage_pos_y, f'False Positive\nP = {p_no_damage_and_pos:.4f}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='black', zorder=4)
    ax.text((level1_x+level2_x)/2, no_damage_pos_y+0.35, f'P(Pos|NoDamage) = {p_pos_given_no_damage:.2f}',
            fontsize=9, style='italic', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # No Damage -> Negative
    ax.plot([level1_x+0.7, level2_x-0.5], [no_damage_y-0.2, no_damage_neg_y+0.3], 
            'k--', linewidth=2, zorder=1)
    neg_no_damage_rect = plt.Rectangle((level2_x-0.6, no_damage_neg_y-0.4), 1.2, 0.8, 
                                       facecolor='#F0F0F0', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(neg_no_damage_rect)
    ax.text(level2_x, no_damage_neg_y, f'Negative\nP = {p_no_damage_and_neg:.4f}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='black', zorder=4)
    ax.text((level1_x+level2_x)/2, no_damage_neg_y-0.35, f'P(Neg|NoDamage) = {p_neg_given_no_damage:.2f}',
            fontsize=9, style='italic', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5)
    
    # Draw Bayes Result Box (for Positive test result) - softer color
    result_y = 5
    # Arrows from both positive results to Bayes box (gray instead of purple)
    ax.plot([level2_x+0.6, result_x-0.5], [damage_pos_y-0.1, result_y+0.3], 
            'gray', linewidth=2, linestyle='--', alpha=0.7, zorder=1)
    ax.plot([level2_x+0.6, result_x-0.5], [no_damage_pos_y+0.1, result_y-0.3], 
            'gray', linewidth=2, linestyle='--', alpha=0.7, zorder=1)
    
    result_rect = plt.Rectangle((result_x-0.8, result_y-1), 1.6, 2, 
                               facecolor='#FFFACD', edgecolor='black', linewidth=2.5, zorder=3)
    ax.add_patch(result_rect)
    result_text = (f'Bayes Result\n(If Test = Positive)\n\n'
                   f'P(Positive) = {p_pos_total:.4f}\n\n'
                   f'P(Damage|Positive) = {posterior:.4f}\n'
                   f'= {posterior*100:.2f}%')
    ax.text(result_x, result_y, result_text, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='black', zorder=4)
    
    # Add title (simpler, no colored background)
    ax.text(5, 9.5, 'Probability Tree Diagram - Damage Detection (Bayes Theorem)', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.95))
    
    # Add parameters box - make it wider to fit text properly
    param_lines = [
        'Parameters:',
        f'Prior P(Damage) = {prior:.3f}',
        f'Sensitivity = {sensitivity:.3f}',
        f'Specificity = {specificity:.3f}'
    ]
    param_text = '\n'.join(param_lines)
    # Calculate text width to size box properly
    param_rect_width = 2.8
    param_rect_height = 1.8
    param_rect = plt.Rectangle((0.15, 0.15), param_rect_width, param_rect_height, 
                               facecolor='white', edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(param_rect)
    ax.text(0.15 + param_rect_width/2, 0.15 + param_rect_height - 0.1, param_text, 
            ha='center', va='top', fontsize=10, fontweight='bold', zorder=4)
    
    # Add summary statistics (softer color)
    summary_text = (f'Summary:\n'
                    f'P(Positive) = {p_pos_total:.4f}\n'
                    f'P(Negative) = {p_neg_total:.4f}\n'
                    f'P(Damage|Positive) = {posterior:.4f}')
    summary_rect = plt.Rectangle((7.4, 0.15), 2.4, 1.3, 
                                facecolor='#F5F5F5', edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(summary_rect)
    ax.text(8.6, 0.8, summary_text, ha='center', va='center', 
            fontsize=10, fontweight='bold', zorder=4)
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return out

def create_statistical_report(concrete_stats, material_group_stats, fitting_info, bayes_info, out_file='lab4_statistical_report.txt'):
    """Create a textual report summarizing results."""
    lines = []
    lines.append('Lab 4: Statistical Analysis - Summary Report\n')
    lines.append('=' * 60 + '\n')
    lines.append('\n1) Concrete strength descriptive statistics:\n')
    lines.append('-' * 60 + '\n')
    for k, v in concrete_stats.items():
        if k == 'percentiles':
            lines.append(f'  Percentiles: {v}\n')
        else:
            lines.append(f'  {k}: {v}\n')
    lines.append('\n2) Material group statistics (mean, std):\n')
    lines.append('-' * 60 + '\n')
    for mat, stats in material_group_stats.items():
        lines.append(f'  {mat}: mean={stats["mean"]:.2f}, std={stats["std"]:.2f}, n={stats["count"]}\n')
    lines.append('\n3) Fitted normal distribution for concrete:\n')
    lines.append('-' * 60 + '\n')
    lines.append(f'  Fitted mu = {fitting_info["mu_hat"]:.2f}, sigma = {fitting_info["sigma_hat"]:.2f}\n')
    lines.append(f'  Synthetic sample: mean={fitting_info["synt_mean"]:.2f}, std={fitting_info["synt_std"]:.2f}\n')
    lines.append('\n4) Bayes theorem application:\n')
    lines.append('-' * 60 + '\n')
    lines.append(f'  Prior (Damage) = {bayes_info["prior"]:.3f}\n')
    lines.append(f'  Sensitivity = {bayes_info["sensitivity"]:.3f}, Specificity = {bayes_info["specificity"]:.3f}\n')
    lines.append(f'  P(Positive) = {bayes_info["p_pos"]:.3f}\n')
    lines.append(f'  Posterior P(Damage|Positive) = {bayes_info["posterior"]:.3f}\n')
    lines.append('\n5) Key findings and engineering implications:\n')
    lines.append('-' * 60 + '\n')
    # Simple interpretation heuristics
    # skewness/kurtosis based insights
    skew = concrete_stats.get('skewness', 0)
    kurt = concrete_stats.get('kurtosis', 0)
    if abs(skew) < 0.5:
        lines.append('  - Concrete strength approximately symmetric (low skewness).\n')
    elif skew > 0:
        lines.append('  - Concrete strength is right-skewed (long tail to higher strengths).\n')
    else:
        lines.append('  - Concrete strength is left-skewed (tail to lower strengths).\n')
    if kurt > 0:
        lines.append('  - Distribution is leptokurtic (heavy tails) which may indicate more outliers.\n')
    else:
        lines.append('  - Distribution is platykurtic (lighter tails).\n')
    lines.append('\nRecommendations:\n')
    lines.append('-' * 60 + '\n')
    lines.append('  - If distribution deviates from normal, prefer median for central tendency in design safety margins.\n')
    lines.append('  - Bayes result shows that even with a good test, low prior prevalence leads to lower positive predictive value.\n')
    report_path = os.path.join(OUT_DIR, out_file)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return report_path

def main():
    # Load datasets
    concrete = load_data('concrete_strength.csv')
    materials = load_data('material_properties.csv')
    loads = load_data('structural_loads.csv')

    # Assume concrete strength column name contains 'strength' or 'strength_mpa'
    candidate_cols = [c for c in concrete.columns if 'strength' in c.lower()]
    if not candidate_cols:
        raise ValueError('Could not find a strength column in concrete dataset. Columns: ' + ','.join(concrete.columns))
    strength_col = candidate_cols[0]
    concrete_series = concrete[strength_col]

    # Descriptive stats for concrete
    concrete_stats = calculate_descriptive_stats(concrete_series)

    # Plots for concrete
    hfile = plot_histogram_with_normal(concrete_series, 'Concrete Strength Distribution', 'concrete_strength_distribution.png')

    # Fit normal and synthetic compare
    fitting_info = fit_normal_and_compare(concrete_series, 'distribution_fitting.png')

    # Material comparison
    # Assume material_properties has columns: material_type, strength or similar
    mat_val_cols = [c for c in materials.columns if ('strength' in c.lower() or 'yield' in c.lower())]
    if not mat_val_cols:
        raise ValueError('Could not find a strength column in material_properties dataset. Columns: ' + ','.join(materials.columns))
    mat_val = mat_val_cols[0]
    mat_group_col = None
    for c in materials.columns:
        if 'material' in c.lower() or 'type' in c.lower():
            mat_group_col = c
            break
    if mat_group_col is None:
        raise ValueError('Could not find a material type column in material_properties dataset. Columns: ' + ','.join(materials.columns))
    # Group statistics
    material_group_stats = {}
    for name, grp in materials.groupby(mat_group_col):
        vals = grp[mat_val].dropna()
        material_group_stats[str(name)] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)), 'count': int(vals.count())}
    boxfile = plot_boxplot_by_group(materials, mat_val, mat_group_col, 'material_comparison_boxplot.png')

    # Probability distributions demonstration
    probfile = plot_probability_distributions('probability_distributions.png')

    # Tasks: Binomial, Poisson, Normal, Exponential scenario calculations
    # Binomial: 100 components, p=0.05
    binom_pmf_3, binom_cdf_3 = calculate_probability_binomial(100, 0.05, 3)
    binom_cdf_5 = binom.cdf(5, 100, 0.05)

    # Poisson: lambda=10 trucks/hour
    pois_pmf_8, pois_gt_15 = calculate_probability_poisson(10, 8)
    # Normal scenario: steel yield mean 250 std 15
    normal_pct_above_280 = calculate_probability_normal(250, 15, 280, None)
    normal_95th = norm.ppf(0.95, loc=250, scale=15)

    # Exponential: mean=1000
    exp_prob_before_500 = calculate_probability_exponential(1000, 500)
    exp_prob_survive_1500 = 1 - calculate_probability_exponential(1000, 1500)

    # Bayes scenario
    prior = 0.05
    sensitivity = 0.95
    specificity = 0.90
    posterior, p_pos = apply_bayes_theorem(prior, sensitivity, specificity)
    treefile = plot_probability_tree(prior, sensitivity, specificity, 'probability_tree.png')

    # Create dashboard-like summary (a simple text-based summary plotted)
    plt.figure(figsize=(8,6))
    plt.axis('off')
    txt = [
        f'Concrete mean={concrete_stats["mean"]:.2f}, median={concrete_stats["median"]:.2f}, std={concrete_stats["std"]:.2f}',
        f'Materials: ' + ', '.join([f'{k}:mean={v["mean"]:.1f},std={v["std"]:.1f}' for k,v in material_group_stats.items()]),
        f'Binomial P(X=3)={binom_pmf_3:.4f}, P(X<=5)={binom_cdf_5:.4f}',
        f'Poisson P(X=8)={pois_pmf_8:.4f}, P(X>15)={pois_gt_15:.4f}',
        f'Normal >280MPa = {normal_pct_above_280:.4f}, 95th={normal_95th:.2f}',
        f'Exponential P(fail<500)={exp_prob_before_500:.4f}, survive>1500={exp_prob_survive_1500:.4f}',
        f'Bayes P(Damage|Positive)={posterior:.4f} (P+={p_pos:.4f})'
    ]
    for i,l in enumerate(txt):
        plt.text(0.01, 0.95 - i*0.12, l, fontsize=10)
    plt.title('Statistical Summary Dashboard')
    dashfile = os.path.join(OUT_DIR, 'statistical_summary_dashboard.png')
    plt.savefig(dashfile, bbox_inches='tight')
    plt.close()

    # Create report
    bayes_info = {'prior': prior, 'sensitivity': sensitivity, 'specificity': specificity, 'posterior': posterior, 'p_pos': p_pos}
    report_path = create_statistical_report(concrete_stats, material_group_stats, fitting_info, {'prior':prior,'sensitivity':sensitivity,'specificity':specificity,'posterior':posterior,'p_pos':p_pos})

    # Print console summary
    print('\\n=== Lab 4 Summary (Console Output) ===\\n')
    print('Concrete descriptive stats:')
    for k,v in concrete_stats.items():
        if k == 'percentiles':
            print(f'  {k}: {v}')
        else:
            print(f'  {k}: {v}')
    print('\\nMaterial group summary:')
    for k,v in material_group_stats.items():
        print(f'  {k}: mean={v["mean"]:.2f}, std={v["std"]:.2f}, n={v["count"]}')
    print('\\nProbability calculations:')
    print(f'  Binomial P(X=3) = {binom_pmf_3:.6f}, P(X<=5) = {binom_cdf_5:.6f}')
    print(f'  Poisson P(X=8) = {pois_pmf_8:.6f}, P(X>15) = {pois_gt_15:.6f}')
    print(f'  Normal (>280) = {normal_pct_above_280:.6f}, 95th percentile = {normal_95th:.2f}')
    print(f'  Exponential P(<500) = {exp_prob_before_500:.6f}, survive>1500 = {exp_prob_survive_1500:.6f}')
    print(f'  Bayes P(Damage|Positive) = {posterior:.6f} (P(Positive)={p_pos:.6f})')

    # Print file locations
    files = {
        'concrete_plot': hfile,
        'material_boxplot': boxfile,
        'prob_distribution_demo': probfile,
        'distribution_fitting': fitting_info['file'],
        'dashboard': dashfile,
        'probability_tree': treefile,
        'report': report_path
    }
    print('\\nGenerated files:')
    for k,f in files.items():
        print(f'  {k}: {f}')

    return files

if __name__ == "__main__":
    main()
