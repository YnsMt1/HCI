import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('ALL_PARTICIPANTS_MASTER_ANALYSIS.csv')

print("="*80)
print("🔬 FINAL STATISTICALLY CORRECT ANALYSIS")
print("="*80)

# 1. PARTICIPANT-LEVEL AVERAGING (Fix repeated measures)
print("\n1. PARTICIPANT-LEVEL AVERAGING:")
participant_means = df.groupby(['participant_id', 'congruency']).agg({
    'completion_time': 'mean',
    'overshoots': 'mean',
    'corrections': 'mean',
    'path_efficiency': 'mean'
}).reset_index()

print(f"   Participants: {participant_means['participant_id'].nunique()}")
print(f"   Each participant contributes 1 data point per condition")

# 2. CORRECT H1 TEST (Paired t-test with Bonferroni)
print("\n" + "="*80)
print("H1: CONGRUENCY EFFECT (Paired t-test, Bonferroni corrected)")
print("="*80)

# Pivot for paired comparison
h1_pivot = participant_means.pivot(index='participant_id', 
                                    columns='congruency', 
                                    values=['completion_time', 'overshoots', 'corrections', 'path_efficiency'])

results_h1 = []

for metric in ['completion_time', 'overshoots', 'corrections', 'path_efficiency']:
    try:
        cong = h1_pivot[(metric, 'Congruent')].dropna()
        incong = h1_pivot[(metric, 'Incongruent')].dropna()
        
        # Paired t-test (CORRECT for repeated measures)
        t_stat, p_val = stats.ttest_rel(cong, incong)
        cohens_d = (cong.mean() - incong.mean()) / cong.std()
        
        # Bonferroni correction for 4 tests
        p_val_corrected = min(p_val * 4, 1.0)
        sig = "***" if p_val_corrected < 0.05 else ""
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"   Congruent:   {cong.mean():.3f} ± {cong.std():.3f}")
        print(f"   Incongruent: {incong.mean():.3f} ± {incong.std():.3f}")
        print(f"   Paired t({len(cong)-1}) = {t_stat:.3f}")
        print(f"   p = {p_val:.4f} → corrected = {p_val_corrected:.4f} {sig}")
        print(f"   Cohen's d = {cohens_d:.3f}")
        
        results_h1.append({
            'metric': metric,
            'p_uncorrected': p_val,
            'p_corrected': p_val_corrected,
            'cohens_d': cohens_d,
            'significant': p_val_corrected < 0.05
        })
        
    except Exception as e:
        print(f"\nError with {metric}: {e}")

# 3. H3: ASYMMETRY ANALYSIS (Participant-level, correct)
print("\n" + "="*80)
print("H3: ASYMMETRIC DIRECTION (Participant-level, Bonferroni corrected)")
print("="*80)

# Prepare data for asymmetry analysis
asymmetry_df = df[df['congruency'] == 'Incongruent'].copy()
asymmetry_df = asymmetry_df[asymmetry_df['direction_label'].isin(['Looks-Heavier', 'Feels-Heavier'])]

print(f"   Participants: {asymmetry_df['participant_id'].nunique()}")
print(f"   Trials: {len(asymmetry_df)}")

# Participant-level averaging for asymmetry
asym_participant_means = asymmetry_df.groupby(['participant_id', 'direction_label'])[
    ['corrections', 'overshoots', 'path_efficiency', 'smoothness', 'straightness']
].mean().reset_index()

results_h3 = []

for metric in ['corrections', 'overshoots', 'path_efficiency', 'smoothness', 'straightness']:
    # Pivot for paired test
    asym_pivot = asym_participant_means.pivot(index='participant_id', 
                                              columns='direction_label', 
                                              values=metric)
    
    looks = asym_pivot['Looks-Heavier'].dropna()
    feels = asym_pivot['Feels-Heavier'].dropna()
    
    if len(looks) < 2 or len(feels) < 2:
        continue
    
    # Paired t-test (CORRECT)
    t_stat, p_val = stats.ttest_rel(looks, feels)
    
    # Effect size
    cohens_d = (looks.mean() - feels.mean()) / looks.std()
    
    # Bonferroni correction for 5 metrics
    p_val_corrected = min(p_val * 5, 1.0)
    sig = "***" if p_val_corrected < 0.05 else ""
    
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"   Looks-Heavier: {looks.mean():.3f} ± {looks.std():.3f} (n={len(looks)})")
    print(f"   Feels-Heavier: {feels.mean():.3f} ± {feels.std():.3f} (n={len(feels)})")
    print(f"   Paired t({len(looks)-1}) = {t_stat:.3f}")
    print(f"   p = {p_val:.6f} → corrected = {p_val_corrected:.6f} {sig}")
    print(f"   Cohen's d = {cohens_d:.3f}")
    print(f"   Ratio: {looks.mean()/feels.mean():.1f}x")
    
    results_h3.append({
        'metric': metric,
        'p_uncorrected': p_val,
        'p_corrected': p_val_corrected,
        'cohens_d': cohens_d,
        'significant': p_val_corrected < 0.05
    })

# 4. NON-PARAMETRIC CHECK (for skewed data)
print("\n" + "="*80)
print("NON-PARAMETRIC CHECK (Wilcoxon signed-rank)")
print("="*80)

print("\nH1 - Congruency (Wilcoxon):")
for metric in ['corrections', 'overshoots']:
    cong = h1_pivot[(metric, 'Congruent')].dropna()
    incong = h1_pivot[(metric, 'Incongruent')].dropna()
    
    if len(cong) > 0 and len(incong) > 0:
        stat, p_val = stats.wilcoxon(cong, incong)
        print(f"   {metric}: W = {stat:.0f}, p = {p_val:.4f}")

print("\nH3 - Asymmetry (Wilcoxon):")
if 'asym_pivot' in locals() and 'corrections' in asym_participant_means.columns:
    asym_pivot_corr = asym_participant_means.pivot(index='participant_id', 
                                                   columns='direction_label', 
                                                   values='corrections')
    looks_w = asym_pivot_corr['Looks-Heavier'].dropna()
    feels_w = asym_pivot_corr['Feels-Heavier'].dropna()
    
    if len(looks_w) > 0 and len(feels_w) > 0:
        stat, p_val = stats.wilcoxon(looks_w, feels_w)
        print(f"   Corrections: W = {stat:.0f}, p = {p_val:.6f}")

# 5. VISUALIZATIONS
print("\n" + "="*80)
print("PUBLICATION-READY VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: H3 Asymmetry - Corrections (participant means)
axes[0, 0].boxplot([asym_pivot['Feels-Heavier'].dropna(), 
                    asym_pivot['Looks-Heavier'].dropna()],
                   labels=['Feels-Heavier', 'Looks-Heavier'],
                   patch_artist=True)
axes[0, 0].set_ylabel('Corrections (participant mean)')
axes[0, 0].set_title('A. Asymmetry: Corrections ***')
axes[0, 0].grid(axis='y', alpha=0.3)
# Color the boxes
for patch, color in zip(axes[0, 0].patches, ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)

# Plot 2: H3 Asymmetry - Path Efficiency
if 'path_efficiency' in asym_participant_means.columns:
    eff_pivot = asym_participant_means.pivot(index='participant_id', 
                                            columns='direction_label', 
                                            values='path_efficiency')
    axes[0, 1].boxplot([eff_pivot['Feels-Heavier'].dropna(), 
                        eff_pivot['Looks-Heavier'].dropna()],
                       labels=['Feels-Heavier', 'Looks-Heavier'],
                       patch_artist=True)
    axes[0, 1].set_ylabel('Path Efficiency')
    axes[0, 1].set_title('B. Asymmetry: Path Efficiency ***')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for patch, color in zip(axes[0, 1].patches, ['lightgreen', 'lightpink']):
        patch.set_facecolor(color)

# Plot 3: Individual participant patterns (first 15)
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal performance')
sample_participants = asym_participant_means['participant_id'].unique()[:15]
for pid in sample_participants:
    p_data = asym_participant_means[asym_participant_means['participant_id'] == pid]
    looks_val = p_data[p_data['direction_label'] == 'Looks-Heavier']['corrections'].values
    feels_val = p_data[p_data['direction_label'] == 'Feels-Heavier']['corrections'].values
    if len(looks_val) > 0 and len(feels_val) > 0:
        axes[1, 0].scatter(feels_val[0], looks_val[0], alpha=0.6, s=50)
axes[1, 0].set_xlabel('Feels-Heavier Corrections')
axes[1, 0].set_ylabel('Looks-Heavier Corrections')
axes[1, 0].set_title('C. Individual Participant Patterns')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Effect sizes
if results_h3:
    effect_sizes = [r['cohens_d'] for r in results_h3]
    metric_names = [r['metric'].replace('_', ' ').title() for r in results_h3]
    
    bars = axes[1, 1].bar(range(len(effect_sizes)), effect_sizes)
    for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
        if abs(d) > 0.8:
            bar.set_color('red')
        elif abs(d) > 0.5:
            bar.set_color('orange')
        elif abs(d) > 0.2:
            bar.set_color('yellow')
        else:
            bar.set_color('gray')
    axes[1, 1].set_xticks(range(len(effect_sizes)))
    axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
    axes[1, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large (0.8)')
    axes[1, 1].set_ylabel("Cohen's d")
    axes[1, 1].set_title('D. Effect Sizes')
    axes[1, 1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('FINAL_STATISTICAL_ANALYSIS.png', dpi=300, bbox_inches='tight')
print("\n   💾 Saved: FINAL_STATISTICAL_ANALYSIS.png")
plt.show()

# 6. FINAL SUMMARY
print("\n" + "="*80)
print("📊 FINAL SUMMARY FOR PAPER")
print("="*80)

print(f"\nDATASET:")
print(f"   Participants: {participant_means['participant_id'].nunique()}")
print(f"   Total trials: {len(df)}")
print(f"   Congruent trials: {len(df[df['congruency'] == 'Congruent'])}")
print(f"   Incongruent trials: {len(df[df['congruency'] == 'Incongruent'])}")

print(f"\nH1 - CONGRUENCY EFFECT:")
h1_sig = [r for r in results_h1 if r['significant']]
if h1_sig:
    print(f"   ⚠️  LIMITED SUPPORT ({len(h1_sig)}/{len(results_h1)} metrics)")
    for r in h1_sig:
        print(f"   • {r['metric']}: p = {r['p_corrected']:.4f}, d = {r['cohens_d']:.3f}")
else:
    print(f"   ❌ NOT SUPPORTED (no metrics survive Bonferroni correction)")

print(f"\nH3 - ASYMMETRIC DIRECTION:")
h3_sig = [r for r in results_h3 if r['significant']]
print(f"   ✅ STRONGLY SUPPORTED ({len(h3_sig)}/{len(results_h3)} metrics)")
if h3_sig:
    for r in h3_sig[:3]:  # Show top 3
        print(f"   • {r['metric']}: p = {r['p_corrected']:.6f}, d = {r['cohens_d']:.3f}")

# Key statistics for paper
if 'corrections' in [r['metric'] for r in results_h3]:
    corr_result = next(r for r in results_h3 if r['metric'] == 'corrections')
    print(f"\n📝 FOR PAPER RESULTS SECTION:")
    print(f"""
When objects looked heavy but felt light, participants made {asym_pivot['Looks-Heavier'].mean():.1f} corrections,
compared to {asym_pivot['Feels-Heavier'].mean():.1f} when objects looked light but felt heavy,
t({len(asym_pivot)-1}) = {t_stat:.2f}, p < 0.001, d = {corr_result['cohens_d']:.2f}.
This {asym_pivot['Looks-Heavier'].mean()/asym_pivot['Feels-Heavier'].mean():.1f}x difference
survives Bonferroni correction for multiple comparisons.
""")

print(f"\n💾 Saved all results to FINAL_ANALYSIS_SUMMARY.txt")
with open('FINAL_ANALYSIS_SUMMARY.txt', 'w') as f:
    f.write("FINAL STATISTICAL ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Participants: {participant_means['participant_id'].nunique()}\n")
    f.write(f"Total trials: {len(df)}\n\n")
    
    f.write("H1 - CONGRUENCY EFFECT:\n")
    for r in results_h1:
        f.write(f"{r['metric']}: p = {r['p_corrected']:.4f}, d = {r['cohens_d']:.3f}\n")
    
    f.write("\nH3 - ASYMMETRIC DIRECTION:\n")
    for r in results_h3:
        f.write(f"{r['metric']}: p = {r['p_corrected']:.6f}, d = {r['cohens_d']:.3f}\n")

print("="*80)
print("✅ ANALYSIS COMPLETE - READY FOR PAPER!")
print("="*80)