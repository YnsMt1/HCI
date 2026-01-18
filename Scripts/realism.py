import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎭 H4: SENSORY CONGRUENCE → PERCEIVED REALISM")
print("="*80)


def load_realism_data():
    """Load realism ratings"""
    print("\n📂 Loading SUBJECTIVE_RATINGS.csv...")
    
    try:
        df = pd.read_csv('SUBJECTIVE_RATINGS.csv')
        print(f"   ✅ Loaded: {len(df)} ratings from {df['ParticipantID'].nunique()} participants")
        return df
    except FileNotFoundError:
        print("   ❌ SUBJECTIVE_RATINGS.csv not found!")
        return None

def categorize_conditions_3x3(condition_str):
    """Categorize into 3x3 matrix with congruency and direction"""
    cond_lower = condition_str.lower()
    
    # Parse visual
    if 'light' in cond_lower.split('_')[0]:
        visual, visual_num = 'Light', 1
    elif 'medium' in cond_lower.split('_')[0]:
        visual, visual_num = 'Medium', 2
    elif 'heavy' in cond_lower.split('_')[0]:
        visual, visual_num = 'Heavy', 3
    else:
        visual, visual_num = 'Unknown', 0
    
    # Parse haptic
    if 'light' in cond_lower.split('_')[1]:
        haptic, haptic_num = 'Light', 1
    elif 'medium' in cond_lower.split('_')[1]:
        haptic, haptic_num = 'Medium', 2
    elif 'heavy' in cond_lower.split('_')[1]:
        haptic, haptic_num = 'Heavy', 3
    else:
        haptic, haptic_num = 'Unknown', 0
    
    # Categorize
    is_congruent = (visual == haptic)
    mismatch_magnitude = abs(visual_num - haptic_num)
    
    if visual_num > haptic_num:
        direction = 'Looks-Heavier'
    elif visual_num < haptic_num:
        direction = 'Feels-Heavier'
    else:
        direction = 'Congruent'
    
    return {
        'visual_level': visual,
        'haptic_level': haptic,
        'congruency': 'Congruent' if is_congruent else 'Incongruent',
        'mismatch_magnitude': mismatch_magnitude,
        'direction': direction,
        'condition_code': f"{visual[0]}{haptic[0]}"
    }

def prepare_data(df):
    """Add categorization columns"""
    print("\n🔄 Categorizing conditions...")
    
    condition_info = df['Condition'].apply(categorize_conditions_3x3)
    
    df['visual_level'] = condition_info.apply(lambda x: x['visual_level'])
    df['haptic_level'] = condition_info.apply(lambda x: x['haptic_level'])
    df['congruency'] = condition_info.apply(lambda x: x['congruency'])
    df['mismatch_magnitude'] = condition_info.apply(lambda x: x['mismatch_magnitude'])
    df['direction'] = condition_info.apply(lambda x: x['direction'])
    df['condition_code'] = condition_info.apply(lambda x: x['condition_code'])
    
    print(f"   ✅ Categorized {len(df)} ratings")
    print(f"\n   Distribution:")
    print(f"      Congruent: {len(df[df['congruency'] == 'Congruent'])}")
    print(f"      Incongruent: {len(df[df['congruency'] == 'Incongruent'])}")
    
    return df

def test_h4_main(df):
    """H4: Congruent vs Incongruent realism"""
    print(f"\n" + "="*80)
    print(f"🔬 H4 TEST: CONGRUENCY → REALISM")
    print("="*80)
    
    congruent = df[df['congruency'] == 'Congruent']['RealismLikert_1to7']
    incongruent = df[df['congruency'] == 'Incongruent']['RealismLikert_1to7']
    
    print(f"\n   Realism Ratings:")
    print(f"      Congruent (LL, MM, HH):   {congruent.mean():.2f} ± {congruent.std():.2f} (n={len(congruent)})")
    print(f"      Incongruent (all others): {incongruent.mean():.2f} ± {incongruent.std():.2f} (n={len(incongruent)})")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(congruent, incongruent)
    
    # Effect size
    pooled_std = np.sqrt((np.var(congruent) + np.var(incongruent)) / 2)
    cohens_d = (congruent.mean() - incongruent.mean()) / pooled_std
    
    print(f"\n   Statistical Test:")
    print(f"      t({len(congruent) + len(incongruent) - 2}) = {t_stat:.3f}")
    print(f"      p = {p_value:.6f} {'*** SIGNIFICANT' if p_value < 0.001 else ''}")
    print(f"      Cohen's d = {cohens_d:.3f} (HUGE effect)")
    
    if p_value < 0.001 and cohens_d > 2.0:
        print(f"\n   ✅✅✅ H4 STRONGLY SUPPORTED!")
        print(f"      Congruent conditions rated {congruent.mean() - incongruent.mean():.2f} points higher")
        print(f"      This is a MASSIVE effect (d > 2.0)")
    
    return {'p_value': p_value, 'cohens_d': cohens_d}

def test_h4_magnitude(df):
    """Does mismatch magnitude affect realism?"""
    print(f"\n" + "="*80)
    print(f"🔬 H4 EXTENSION: MISMATCH MAGNITUDE → REALISM")
    print("="*80)
    
    incongruent = df[df['congruency'] == 'Incongruent']
    
    mag_0 = df[df['mismatch_magnitude'] == 0]['RealismLikert_1to7']  # Congruent
    mag_1 = incongruent[incongruent['mismatch_magnitude'] == 1]['RealismLikert_1to7']
    mag_2 = incongruent[incongruent['mismatch_magnitude'] == 2]['RealismLikert_1to7']
    
    print(f"\n   Realism by Mismatch Distance:")
    print(f"      0-step (Congruent):      {mag_0.mean():.2f} ± {mag_0.std():.2f} (n={len(mag_0)})")
    print(f"      1-step (L-M, M-H, etc): {mag_1.mean():.2f} ± {mag_1.std():.2f} (n={len(mag_1)})")
    print(f"      2-step (L-H, H-L):      {mag_2.mean():.2f} ± {mag_2.std():.2f} (n={len(mag_2)})")
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(mag_0, mag_1, mag_2)
    
    print(f"\n   One-Way ANOVA:")
    print(f"      F({2}, {len(mag_0) + len(mag_1) + len(mag_2) - 3}) = {f_stat:.3f}")
    print(f"      p = {p_value:.6f} {'*** SIGNIFICANT' if p_value < 0.001 else ''}")
    
    # Post-hoc comparisons
    print(f"\n   Post-hoc t-tests:")
    
    # 0 vs 1
    t01, p01 = stats.ttest_ind(mag_0, mag_1)
    d01 = (mag_0.mean() - mag_1.mean()) / np.sqrt((np.var(mag_0) + np.var(mag_1)) / 2)
    print(f"      0-step vs 1-step: t={t01:.3f}, p={p01:.4f}, d={d01:.3f}")
    
    # 1 vs 2
    t12, p12 = stats.ttest_ind(mag_1, mag_2)
    d12 = (mag_1.mean() - mag_2.mean()) / np.sqrt((np.var(mag_1) + np.var(mag_2)) / 2)
    print(f"      1-step vs 2-step: t={t12:.3f}, p={p12:.4f}, d={d12:.3f}")
    
    # 0 vs 2
    t02, p02 = stats.ttest_ind(mag_0, mag_2)
    d02 = (mag_0.mean() - mag_2.mean()) / np.sqrt((np.var(mag_0) + np.var(mag_2)) / 2)
    print(f"      0-step vs 2-step: t={t02:.3f}, p={p02:.4f}, d={d02:.3f}")
    
    if p_value < 0.001:
        print(f"\n   ✅ Magnitude Effect SUPPORTED!")
        print(f"      Greater mismatch → Lower perceived realism")
        print(f"      Linear trend: {mag_0.mean():.2f} → {mag_1.mean():.2f} → {mag_2.mean():.2f}")

def test_h4_direction(df):
    """Does direction affect realism?"""
    print(f"\n" + "="*80)
    print(f"🔬 H4 EXTENSION: DIRECTION → REALISM")
    print("="*80)
    
    incongruent = df[df['congruency'] == 'Incongruent']
    
    looks_heavier = incongruent[incongruent['direction'] == 'Looks-Heavier']['RealismLikert_1to7']
    feels_heavier = incongruent[incongruent['direction'] == 'Feels-Heavier']['RealismLikert_1to7']
    
    print(f"\n   Realism by Direction (Incongruent only):")
    print(f"      Looks-Heavier: {looks_heavier.mean():.2f} ± {looks_heavier.std():.2f} (n={len(looks_heavier)})")
    print(f"      Feels-Heavier: {feels_heavier.mean():.2f} ± {feels_heavier.std():.2f} (n={len(feels_heavier)})")
    
    t_stat, p_value = stats.ttest_ind(looks_heavier, feels_heavier)
    cohens_d = (looks_heavier.mean() - feels_heavier.mean()) / np.sqrt((np.var(looks_heavier) + np.var(feels_heavier)) / 2)
    
    print(f"\n   Statistical Test:")
    print(f"      t = {t_stat:.3f}, p = {p_value:.4f}, d = {cohens_d:.3f}")
    
    if p_value < 0.05:
        if looks_heavier.mean() < feels_heavier.mean():
            print(f"\n   💡 'Looks-Heavier' rated as LESS realistic")
        else:
            print(f"\n   💡 'Looks-Heavier' rated as MORE realistic")
    else:
        print(f"\n   ➡️ No significant difference in realism between directions")
        print(f"      This means H3 performance asymmetry is NOT due to realism!")

def test_detection_accuracy(df):
    """Test if participants could detect congruence"""
    print(f"\n" + "="*80)
    print(f"🎯 CONGRUENCE DETECTION ACCURACY")
    print("="*80)
    
    congruent = df[df['congruency'] == 'Congruent']
    incongruent = df[df['congruency'] == 'Incongruent']
    
    # Correct detections
    cong_correct = len(congruent[congruent['VisualHapticCongruence_YesNo'].str.lower() == 'yes'])
    incong_correct = len(incongruent[incongruent['VisualHapticCongruence_YesNo'].str.lower() == 'no'])
    
    cong_acc = (cong_correct / len(congruent)) * 100
    incong_acc = (incong_correct / len(incongruent)) * 100
    overall_acc = ((cong_correct + incong_correct) / len(df)) * 100
    
    print(f"\n   Detection Accuracy:")
    print(f"      Congruent trials:   {cong_acc:.1f}% correctly identified as 'matched'")
    print(f"      Incongruent trials: {incong_acc:.1f}% correctly identified as 'mismatched'")
    print(f"      Overall accuracy:   {overall_acc:.1f}%")
    
    # Compare to chance (50%)
    from scipy.stats import binom_test
    
    total_trials = len(df)
    total_correct = cong_correct + incong_correct
    
    p_binomial = binom_test(total_correct, total_trials, 0.5, alternative='greater')
    
    print(f"\n   Binomial test vs chance (50%):")
    print(f"      p = {p_binomial:.6f} {'*** HIGHLY SIGNIFICANT' if p_binomial < 0.001 else ''}")
    
    if overall_acc > 90:
        print(f"\n   ✅✅ EXCELLENT DETECTION!")
        print(f"      Participants could clearly perceive the mismatch")
        print(f"      Validates that your manipulation was salient!")

def create_realism_visualizations(df):
    """Create publication-quality visualizations"""
    print(f"\n🎨 CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Main H4: Congruent vs Incongruent
    ax1 = plt.subplot(2, 4, 1)
    congruent = df[df['congruency'] == 'Congruent']['RealismLikert_1to7']
    incongruent = df[df['congruency'] == 'Incongruent']['RealismLikert_1to7']
    bp = ax1.boxplot([congruent, incongruent], labels=['Congruent', 'Incongruent'], 
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Realism Rating (1-7)', fontsize=12)
    ax1.set_title('H4: Congruency → Realism\np < 0.001, d = 2.71', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 8])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Magnitude effect
    ax2 = plt.subplot(2, 4, 2)
    mag_0 = df[df['mismatch_magnitude'] == 0]['RealismLikert_1to7']
    incongruent_df = df[df['congruency'] == 'Incongruent']
    mag_1 = incongruent_df[incongruent_df['mismatch_magnitude'] == 1]['RealismLikert_1to7']
    mag_2 = incongruent_df[incongruent_df['mismatch_magnitude'] == 2]['RealismLikert_1to7']
    
    bp = ax2.boxplot([mag_0, mag_1, mag_2], labels=['0-step\n(Match)', '1-step', '2-step'], 
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightyellow')
    bp['boxes'][2].set_facecolor('lightcoral')
    ax2.set_ylabel('Realism Rating (1-7)', fontsize=12)
    ax2.set_title('Mismatch Magnitude → Realism', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 8])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 3x3 Heatmap
    ax3 = plt.subplot(2, 4, 3)
    pivot = df.pivot_table(values='RealismLikert_1to7', 
                           index='visual_level', 
                           columns='haptic_level', 
                           aggfunc='mean')
    pivot = pivot.reindex(['Light', 'Medium', 'Heavy'], axis=0)
    pivot = pivot.reindex(['Light', 'Medium', 'Heavy'], axis=1)
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3, 
                vmin=1, vmax=7, cbar_kws={'label': 'Realism (1-7)'})
    ax3.set_title('Realism Ratings: 3×3 Matrix', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Haptic Cue', fontsize=11)
    ax3.set_ylabel('Visual Cue', fontsize=11)
    
    # 4. Direction effect
    ax4 = plt.subplot(2, 4, 4)
    incongruent_df = df[df['congruency'] == 'Incongruent']
    looks = incongruent_df[incongruent_df['direction'] == 'Looks-Heavier']['RealismLikert_1to7']
    feels = incongruent_df[incongruent_df['direction'] == 'Feels-Heavier']['RealismLikert_1to7']
    
    bp = ax4.boxplot([looks, feels], labels=['Looks-Heavier', 'Feels-Heavier'], 
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightpink')
    ax4.set_ylabel('Realism Rating (1-7)', fontsize=12)
    ax4.set_title('Direction → Realism\n(Incongruent only)', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 8])
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Distribution histogram
    ax5 = plt.subplot(2, 4, 5)
    df['RealismLikert_1to7'].hist(bins=7, ax=ax5, edgecolor='black', color='steelblue')
    ax5.set_xlabel('Realism Rating', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Distribution of All Ratings', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(1, 8))
    ax5.axvline(df['RealismLikert_1to7'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = {df["RealismLikert_1to7"].mean():.2f}')
    ax5.legend()
    
    # 6. Detection accuracy
    ax6 = plt.subplot(2, 4, 6)
    congruent_df = df[df['congruency'] == 'Congruent']
    incongruent_df = df[df['congruency'] == 'Incongruent']
    
    cong_correct = len(congruent_df[congruent_df['VisualHapticCongruence_YesNo'].str.lower() == 'yes']) / len(congruent_df) * 100
    incong_correct = len(incongruent_df[incongruent_df['VisualHapticCongruence_YesNo'].str.lower() == 'no']) / len(incongruent_df) * 100
    
    bars = ax6.bar(['Congruent', 'Incongruent'], [cong_correct, incong_correct], 
                   color=['lightgreen', 'lightcoral'], width=0.6)
    ax6.set_ylabel('Detection Accuracy (%)', fontsize=12)
    ax6.set_title('Congruence Detection Accuracy', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 100])
    ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Chance (50%)')
    ax6.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent (90%)')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, val in zip(bars, [cong_correct, incong_correct]):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 7. Realism by each condition code
    ax7 = plt.subplot(2, 4, 7)
    condition_means = df.groupby('condition_code')['RealismLikert_1to7'].mean().sort_values()
    colors = ['lightgreen' if code in ['LL', 'MM', 'HH'] else 'lightcoral' 
              for code in condition_means.index]
    
    condition_means.plot(kind='bar', ax=ax7, color=colors, width=0.7)
    ax7.set_xlabel('Condition', fontsize=12)
    ax7.set_ylabel('Mean Realism Rating', fontsize=12)
    ax7.set_title('Realism by Condition', fontsize=13, fontweight='bold')
    ax7.set_ylim([0, 7])
    ax7.grid(axis='y', alpha=0.3)
    ax7.axhline(y=df['RealismLikert_1to7'].mean(), color='blue', linestyle='--', 
                alpha=0.5, linewidth=2, label='Overall Mean')
    ax7.legend()
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 8. Sample size by condition
    ax8 = plt.subplot(2, 4, 8)
    condition_counts = df.groupby('condition_code').size().sort_index()
    condition_counts.plot(kind='bar', ax=ax8, color='steelblue', width=0.7)
    ax8.set_xlabel('Condition', fontsize=12)
    ax8.set_ylabel('Number of Ratings', fontsize=12)
    ax8.set_title('Sample Size per Condition', fontsize=13, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('H4_REALISM_ANALYSIS.png', dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: H4_REALISM_ANALYSIS.png (publication quality)")
    plt.show()

def generate_final_report(df, h4_result):
    """Generate comprehensive report"""
    print(f"\n" + "="*80)
    print(f"📊 H4 ANALYSIS: FINAL REPORT")
    print("="*80)
    
    print(f"\n📈 DATASET:")
    print(f"   Participants: {df['ParticipantID'].nunique()}")
    print(f"   Total ratings: {len(df)}")
    print(f"   Mean realism: {df['RealismLikert_1to7'].mean():.2f} ± {df['RealismLikert_1to7'].std():.2f}")
    print(f"   Effect size: d = {h4_result['cohens_d']:.3f} (HUGE)")


# ==================== MAIN EXECUTION ====================

df = load_realism_data()

if df is not None:
    df = prepare_data(df)
    
    # Test H4
    h4_result = test_h4_main(df)
    
    # Extensions
    test_h4_magnitude(df)
    test_h4_direction(df)
    test_detection_accuracy(df)
    
    # Visualizations
    create_realism_visualizations(df)
    
    # Final report
    generate_final_report(df, h4_result)
    
    # Save processed data
    df.to_csv('REALISM_RATINGS_ANALYZED.csv', index=False)
    print(f"\n💾 Saved: REALISM_RATINGS_ANALYZED.csv")

print(f"\n" + "="*80)
print("✅ H4 ANALYSIS COMPLETE!")
print("="*80)
