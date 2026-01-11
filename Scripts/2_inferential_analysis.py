# 2_inferential_analysis.py
# Purpose: Proper inferential statistics with participant-level aggregation
# This is the CORRECT way to test your hypotheses

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 INFERENTIAL ANALYSIS (Participant-Level)")
print("="*80)
print("✅ Uses proper repeated measures statistics")
print("   This is the version for your paper!")
print("="*80)

def format_p(p):
    """Format p-value for reporting"""
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f} ns"

def cohens_d(a, b):
    """Calculate Cohen's d for paired samples"""
    diff = a - b
    return diff.mean() / diff.std()

def load_data():
    """Load the master dataset created by 1_descriptive_analysis.py"""
    try:
        df = pd.read_csv('ALL_TRIALS_MASTER.csv')
        print(f"\n✅ Loaded {len(df)} trials from {df['participant_id'].nunique()} participants")
        return df
    except FileNotFoundError:
        print("\n❌ ALL_TRIALS_MASTER.csv not found!")
        print("   Run 1_descriptive_analysis.py first")
        return None

def omnibus_anova(df):
    """Omnibus 3×3 Repeated Measures ANOVA"""
    print("\n" + "="*80)
    print("📊 OMNIBUS: 3×3 Repeated Measures ANOVA")
    print("="*80)
    print("Tests overall effect across all 9 conditions")
    
    metrics = ['completion_time', 'corrections', 'overshoots', 'path_efficiency']
    
    results = []
    
    for metric in metrics:
        # Participant-level means for each condition
        pivot = df.pivot_table(
            values=metric,
            index='participant_id',
            columns='condition_code',
            aggfunc='mean'
        )
        
        # Only use participants with data in all conditions
        pivot = pivot.dropna()
        
        if len(pivot) < 3:
            continue
        
        # Convert to list of arrays for each condition
        conditions = ['LL', 'LM', 'LH', 'ML', 'MM', 'MH', 'HL', 'HM', 'HH']
        groups = [pivot[c].values for c in conditions if c in pivot.columns]
        
        if len(groups) < 9:
            print(f"\n⚠️ {metric}: Missing conditions, skipping")
            continue
        
        # Repeated measures ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared approximation)
        grand_mean = np.mean([g.mean() for g in groups])
        ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
        ss_total = sum([sum((g - grand_mean)**2) for g in groups])
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        print(f"\n{metric.upper()}:")
        print(f"   F(8, {len(pivot)-1}) = {f_stat:.3f}")
        print(f"   {format_p(p_value)}")
        print(f"   η² = {eta_sq:.3f}")
        
        results.append({
            'metric': metric,
            'f_stat': f_stat,
            'p_value': p_value,
            'eta_squared': eta_sq
        })
    
    return results

def test_h1_congruence(df):
    """H1: Congruent vs Incongruent (Participant-Level)"""
    print("\n" + "="*80)
    print("🔬 H1: CONGRUENCE EFFECT")
    print("="*80)
    print("Prediction: Congruent conditions yield better performance")
    
    metrics = ['completion_time', 'corrections', 'overshoots', 'path_efficiency']
    
    results = []
    
    for metric in metrics:
        # Aggregate to participant level
        participant_means = df.groupby(['participant_id', 'congruency'])[metric].mean().unstack()
        
        if 'Congruent' not in participant_means or 'Incongruent' not in participant_means:
            continue
        
        # Only participants with both conditions
        complete = participant_means.dropna()
        
        if len(complete) < 3:
            continue
        
        cong = complete['Congruent']
        incong = complete['Incongruent']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(cong, incong)
        
        # Effect size
        d = cohens_d(cong, incong)
        
        print(f"\n{metric.upper()}:")
        print(f"   Congruent:   M = {cong.mean():.3f}, SD = {cong.std():.3f}")
        print(f"   Incongruent: M = {incong.mean():.3f}, SD = {incong.std():.3f}")
        print(f"   t({len(complete)-1}) = {t_stat:.3f}, {format_p(p_value)}")
        print(f"   Cohen's d = {d:.3f}")
        
        results.append({
            'metric': metric,
            'congruent_mean': cong.mean(),
            'incongruent_mean': incong.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'n': len(complete)
        })
    
    return results

def test_h2_asymmetry(df):
    """H2: Underestimation vs Overestimation (Participant-Level)"""
    print("\n" + "="*80)
    print("🔬 H2: ASYMMETRIC MISMATCH DIRECTION")
    print("="*80)
    print("Prediction: Underestimation (looks heavy/feels light) causes more errors")
    
    # Only incongruent trials
    incongruent = df[df['congruency'] == 'Incongruent']
    
    metrics = ['completion_time', 'corrections', 'overshoots', 'path_efficiency']
    
    results = []
    
    for metric in metrics:
        # Aggregate to participant level
        participant_means = incongruent.groupby(['participant_id', 'direction'])[metric].mean().unstack()
        
        if 'Underestimation' not in participant_means or 'Overestimation' not in participant_means:
            continue
        
        # Only participants with both directions
        complete = participant_means.dropna()
        
        if len(complete) < 3:
            continue
        
        under = complete['Underestimation']
        over = complete['Overestimation']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(under, over)
        
        # Effect size
        d = cohens_d(under, over)
        
        print(f"\n{metric.upper()}:")
        print(f"   Underestimation: M = {under.mean():.3f}, SD = {under.std():.3f}")
        print(f"   Overestimation:  M = {over.mean():.3f}, SD = {over.std():.3f}")
        print(f"   t({len(complete)-1}) = {t_stat:.3f}, {format_p(p_value)}")
        print(f"   Cohen's d = {d:.3f}")
        
        results.append({
            'metric': metric,
            'underestimation_mean': under.mean(),
            'overestimation_mean': over.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'n': len(complete)
        })
    
    return results

def test_mismatch_magnitude(df):
    """Exploratory: 1-step vs 2-step mismatch"""
    print("\n" + "="*80)
    print("📊 EXPLORATORY: MISMATCH MAGNITUDE")
    print("="*80)
    
    incongruent = df[df['congruency'] == 'Incongruent']
    
    metrics = ['completion_time', 'corrections', 'overshoots']
    
    for metric in metrics:
        participant_means = incongruent.groupby(['participant_id', 'mismatch_magnitude'])[metric].mean().unstack()
        
        if 1 not in participant_means or 2 not in participant_means:
            continue
        
        complete = participant_means.dropna()
        
        if len(complete) < 3:
            continue
        
        mag1 = complete[1]
        mag2 = complete[2]
        
        t_stat, p_value = stats.ttest_rel(mag1, mag2)
        d = cohens_d(mag1, mag2)
        
        print(f"\n{metric}:")
        print(f"   1-step: {mag1.mean():.3f} ± {mag1.std():.3f}")
        print(f"   2-step: {mag2.mean():.3f} ± {mag2.std():.3f}")
        print(f"   {format_p(p_value)}, d = {d:.3f}")

def create_participant_level_plots(df):
    """Visualize participant-level results"""
    print("\n🎨 Creating participant-level visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Participant-Level Analysis Results', fontsize=16, fontweight='bold')
    
    metrics = ['completion_time', 'corrections', 'overshoots']
    
    # H1: Congruence
    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]
        
        participant_means = df.groupby(['participant_id', 'congruency'])[metric].mean().unstack()
        complete = participant_means.dropna()
        
        # Paired plot
        for i, pid in enumerate(complete.index):
            ax.plot([0, 1], [complete.loc[pid, 'Congruent'], 
                             complete.loc[pid, 'Incongruent']], 
                    'o-', color='gray', alpha=0.3)
        
        # Means
        ax.plot([0, 1], [complete['Congruent'].mean(), 
                         complete['Incongruent'].mean()], 
                'o-', color='red', linewidth=3, markersize=10, 
                label='Group Mean')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Congruent', 'Incongruent'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'H1: {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # H2: Asymmetry
    incongruent = df[df['congruency'] == 'Incongruent']
    
    for idx, metric in enumerate(metrics):
        ax = axes[1, idx]
        
        participant_means = incongruent.groupby(['participant_id', 'direction'])[metric].mean().unstack()
        complete = participant_means.dropna()
        
        if 'Underestimation' not in complete or 'Overestimation' not in complete:
            continue
        
        # Paired plot
        for i, pid in enumerate(complete.index):
            ax.plot([0, 1], [complete.loc[pid, 'Underestimation'], 
                             complete.loc[pid, 'Overestimation']], 
                    'o-', color='gray', alpha=0.3)
        
        # Means
        ax.plot([0, 1], [complete['Underestimation'].mean(), 
                         complete['Overestimation'].mean()], 
                'o-', color='blue', linewidth=3, markersize=10, 
                label='Group Mean')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Under-\nestimation', 'Over-\nestimation'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'H2: {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('PARTICIPANT_LEVEL_RESULTS.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: PARTICIPANT_LEVEL_RESULTS.png")
    plt.show()

def generate_report(anova_results, h1_results, h2_results):
    """Generate final report"""
    print("\n" + "="*80)
    print("📝 FINAL STATISTICAL REPORT")
    print("="*80)
    
    print("\n📊 OMNIBUS ANOVA:")
    for r in anova_results:
        sig = "✅" if r['p_value'] < 0.05 else "⚪"
        print(f"   {sig} {r['metric']}: F = {r['f_stat']:.2f}, {format_p(r['p_value'])}, η² = {r['eta_squared']:.3f}")
    
    print("\n🔬 H1 - CONGRUENCE EFFECT:")
    h1_sig = [r for r in h1_results if r['p_value'] < 0.05]
    if h1_sig:
        print(f"   ✅ SUPPORTED ({len(h1_sig)}/{len(h1_results)} metrics significant)")
        for r in h1_sig:
            print(f"      {r['metric']}: t({r['n']-1}) = {r['t_stat']:.2f}, {format_p(r['p_value'])}, d = {r['cohens_d']:.2f}")
    else:
        print(f"   ❌ NOT SUPPORTED")
    
    print("\n🔬 H2 - ASYMMETRIC MISMATCH:")
    h2_sig = [r for r in h2_results if r['p_value'] < 0.05]
    if h2_sig:
        print(f"   ✅ SUPPORTED ({len(h2_sig)}/{len(h2_results)} metrics significant)")
        for r in h2_sig:
            print(f"      {r['metric']}: t({r['n']-1}) = {r['t_stat']:.2f}, {format_p(r['p_value'])}, d = {r['cohens_d']:.2f}")
    else:
        print(f"   ❌ NOT SUPPORTED")
    
    print("\n📝 FOR YOUR PAPER:")
    print(f"   'We conducted a {df['participant_id'].nunique()}-participant within-subjects")
    print(f"   experiment with {len(df)} total trials across 9 conditions (3×3 design).")
    print(f"   Data were analyzed using repeated-measures statistical tests with")
    print(f"   participant-level aggregation to account for repeated measures.'")

# MAIN
df = load_data()

if df is not None:
    anova_results = omnibus_anova(df)
    h1_results = test_h1_congruence(df)
    h2_results = test_h2_asymmetry(df)
    test_mismatch_magnitude(df)
    create_participant_level_plots(df)
    generate_report(anova_results, h1_results, h2_results)
    
    print("\n" + "="*80)
    print("✅ INFERENTIAL ANALYSIS COMPLETE")
    print("="*80)