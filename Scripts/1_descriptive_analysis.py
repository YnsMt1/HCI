# 1_descriptive_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("📊 DESCRIPTIVE ANALYSIS (Trial-Level)")
print("="*80)
print("⚠️  Statistics here are DESCRIPTIVE ONLY")
print("    Use 2_inferential_analysis.py for hypothesis testing")
print("="*80)

def categorize_3x3_conditions(condition_str):
    """Categorize conditions into 3x3 matrix"""
    condition_lower = condition_str.lower()
    
    if 'light' in condition_lower.split('_')[0]:
        visual_level, visual_num = 'Light', 1
    elif 'medium' in condition_lower.split('_')[0]:
        visual_level, visual_num = 'Medium', 2
    elif 'heavy' in condition_lower.split('_')[0]:
        visual_level, visual_num = 'Heavy', 3
    else:
        visual_level, visual_num = 'Unknown', 0
    
    if 'light' in condition_lower.split('_')[1]:
        haptic_level, haptic_num = 'Light', 1
    elif 'medium' in condition_lower.split('_')[1]:
        haptic_level, haptic_num = 'Medium', 2
    elif 'heavy' in condition_lower.split('_')[1]:
        haptic_level, haptic_num = 'Heavy', 3
    else:
        haptic_level, haptic_num = 'Unknown', 0
    
    is_congruent = (visual_level == haptic_level)
    mismatch_magnitude = abs(visual_num - haptic_num)
    
    if visual_num > haptic_num:
        direction = 'Underestimation'  # Looks heavy, feels light
    elif visual_num < haptic_num:
        direction = 'Overestimation'  # Looks light, feels heavy
    else:
        direction = 'Congruent'
    
    return {
        'visual_level': visual_level,
        'haptic_level': haptic_level,
        'visual_num': visual_num,
        'haptic_num': haptic_num,
        'congruency': 'Congruent' if is_congruent else 'Incongruent',
        'mismatch_magnitude': mismatch_magnitude,
        'direction': direction,
        'condition_code': f"{visual_level[0]}{haptic_level[0]}"
    }

def load_all_participants():
    """Load all participant files"""
    files = sorted(glob.glob('P*.csv'))
    
    if not files:
        print("❌ No P*.csv files found!")
        return None
    
    print(f"\n📂 Found {len(files)} participant files")
    
    all_trials = []
    
    for filepath in files:
        participant_id = filepath.replace('.csv', '')
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"⚠️ Error loading {filepath}: {e}")
            continue
        
        trial_starts = df[df['Event'] == 'TRIAL_START'].index.tolist()
        trial_ends = df[df['Event'] == 'TRIAL_END'].index.tolist()
        
        for i, start_idx in enumerate(trial_starts):
            matching_ends = [end for end in trial_ends if end > start_idx]
            if not matching_ends:
                continue
            
            end_idx = matching_ends[0]
            trial_data = df.iloc[start_idx:end_idx+1].copy()
            
            moving_data = trial_data[trial_data['Event'] == 'MOVING']
            if len(moving_data) < 10:
                continue
            
            duration = trial_data.iloc[-1]['Time'] - trial_data.iloc[0]['Time']
            if duration < 0.5 or duration > 120:
                continue
            
            results_time = df[(df.index > end_idx) & (df['Event'] == 'RESULTS_Time')]
            results_efficiency = df[(df.index > end_idx) & (df['Event'] == 'RESULTS_Efficiency')]
            results_distance = df[(df.index > end_idx) & (df['Event'] == 'RESULTS_Distance')]
            
            if len(results_time) == 0:
                continue
            
            condition_info = categorize_3x3_conditions(trial_data.iloc[0]['Condition'])
            
            positions = moving_data[['PosX', 'PosY', 'PosZ']].values
            times = moving_data['Time'].values
            
            velocities = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]
            accelerations = np.diff(velocities, axis=0)
            smoothness = np.mean(np.linalg.norm(accelerations, axis=1))
            
            straight_distance = np.linalg.norm(positions[-1] - positions[0])
            actual_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            straightness = straight_distance / max(actual_distance, 0.001)
            
            speeds = np.linalg.norm(velocities, axis=1)
            
            trial_info = {
                'participant_id': participant_id,
                'trial_num': i + 1,
                'condition': trial_data.iloc[0]['Condition'],
                **condition_info,
                'completion_time': results_time.iloc[0]['PosX'],
                'path_efficiency': results_efficiency.iloc[0]['PosX'],
                'total_distance': results_distance.iloc[0]['PosX'],
                'overshoots': trial_data.iloc[-1]['Overshoots'],
                'corrections': trial_data.iloc[-1]['Corrections'],
                'smoothness': smoothness,
                'straightness': straightness,
                'avg_speed': np.mean(speeds),
            }
            
            all_trials.append(trial_info)
    
    trials_df = pd.DataFrame(all_trials)
    
    print(f"\n✅ Loaded {len(trials_df)} valid trials from {trials_df['participant_id'].nunique()} participants")
    print(f"   Avg trials/participant: {len(trials_df) / trials_df['participant_id'].nunique():.1f}")
    
    return trials_df

def descriptive_statistics(trials_df):
    """Calculate descriptive statistics"""
    print(f"\n" + "="*80)
    print("📊 DESCRIPTIVE STATISTICS")
    print("="*80)
    
    print("\n3×3 Matrix Trial Counts:")
    print(f"{'':>12} {'Light-H':>10} {'Medium-H':>10} {'Heavy-H':>10}")
    for v_level in ['Light', 'Medium', 'Heavy']:
        counts = []
        for h_level in ['Light', 'Medium', 'Heavy']:
            count = len(trials_df[(trials_df['visual_level'] == v_level) & 
                                  (trials_df['haptic_level'] == h_level)])
            counts.append(f"{count:>10}")
        print(f"{v_level:>10}-V {''.join(counts)}")
    
    print("\n\nCondition Means (Trial-Level):")
    summary = trials_df.groupby(['congruency', 'direction']).agg({
        'completion_time': ['mean', 'std', 'count'],
        'corrections': ['mean', 'std'],
        'overshoots': ['mean', 'std'],
        'path_efficiency': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    # Save summary
    summary.to_csv('TRIAL_LEVEL_DESCRIPTIVE_STATS.csv')
    print("\n💾 Saved: TRIAL_LEVEL_DESCRIPTIVE_STATS.csv")

def create_visualizations(trials_df):
    """Create publication-quality visualizations"""
    print(f"\n🎨 Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Completion Time Heatmap
    ax1 = plt.subplot(3, 4, 1)
    pivot = trials_df.pivot_table(values='completion_time', 
                                   index='visual_level', 
                                   columns='haptic_level', 
                                   aggfunc='mean')
    pivot = pivot.reindex(['Light', 'Medium', 'Heavy'], axis=0)
    pivot = pivot.reindex(['Light', 'Medium', 'Heavy'], axis=1)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax1, 
                cbar_kws={'label': 'Time (s)'})
    ax1.set_title('Completion Time (s)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Haptic Cue', fontsize=11)
    ax1.set_ylabel('Visual Cue', fontsize=11)
    
    # 2. Corrections Heatmap
    ax2 = plt.subplot(3, 4, 2)
    pivot2 = trials_df.pivot_table(values='corrections', 
                                    index='visual_level', 
                                    columns='haptic_level', 
                                    aggfunc='mean')
    pivot2 = pivot2.reindex(['Light', 'Medium', 'Heavy'], axis=0)
    pivot2 = pivot2.reindex(['Light', 'Medium', 'Heavy'], axis=1)
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax2, 
                cbar_kws={'label': 'Count'})
    ax2.set_title('Movement Corrections', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Haptic Cue', fontsize=11)
    ax2.set_ylabel('Visual Cue', fontsize=11)
    
    # 3. Congruent vs Incongruent
    ax3 = plt.subplot(3, 4, 3)
    data = [trials_df[trials_df['congruency'] == 'Congruent']['corrections'],
            trials_df[trials_df['congruency'] == 'Incongruent']['corrections']]
    bp = ax3.boxplot(data, labels=['Congruent', 'Incongruent'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Corrections', fontsize=11)
    ax3.set_title('Congruency Effect', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Mismatch Direction
    ax4 = plt.subplot(3, 4, 4)
    incongruent = trials_df[trials_df['congruency'] == 'Incongruent']
    data = [incongruent[incongruent['direction'] == 'Underestimation']['corrections'],
            incongruent[incongruent['direction'] == 'Overestimation']['corrections']]
    bp = ax4.boxplot(data, labels=['Under-\nestimation', 'Over-\nestimation'], 
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightpink')
    ax4.set_ylabel('Corrections', fontsize=11)
    ax4.set_title('Mismatch Direction Effect', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Path Efficiency Heatmap
    ax5 = plt.subplot(3, 4, 5)
    pivot3 = trials_df.pivot_table(values='path_efficiency', 
                                    index='visual_level', 
                                    columns='haptic_level', 
                                    aggfunc='mean')
    pivot3 = pivot3.reindex(['Light', 'Medium', 'Heavy'], axis=0)
    pivot3 = pivot3.reindex(['Light', 'Medium', 'Heavy'], axis=1)
    sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax5, 
                cbar_kws={'label': 'Efficiency'})
    ax5.set_title('Path Efficiency', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Haptic Cue', fontsize=11)
    ax5.set_ylabel('Visual Cue', fontsize=11)
    
    # 6. Overshoots Heatmap
    ax6 = plt.subplot(3, 4, 6)
    pivot4 = trials_df.pivot_table(values='overshoots', 
                                    index='visual_level', 
                                    columns='haptic_level', 
                                    aggfunc='mean')
    pivot4 = pivot4.reindex(['Light', 'Medium', 'Heavy'], axis=0)
    pivot4 = pivot4.reindex(['Light', 'Medium', 'Heavy'], axis=1)
    sns.heatmap(pivot4, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax6, 
                cbar_kws={'label': 'Count'})
    ax6.set_title('Overshoots', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Haptic Cue', fontsize=11)
    ax6.set_ylabel('Visual Cue', fontsize=11)
    
    # 7. Completion Time by Condition
    ax7 = plt.subplot(3, 4, 7)
    condition_means = trials_df.groupby('condition_code')['completion_time'].mean().sort_values()
    colors = ['lightgreen' if code in ['LL', 'MM', 'HH'] else 'lightcoral' 
              for code in condition_means.index]
    condition_means.plot(kind='bar', ax=ax7, color=colors, width=0.7)
    ax7.set_xlabel('Condition', fontsize=11)
    ax7.set_ylabel('Time (s)', fontsize=11)
    ax7.set_title('Completion Time by Condition', fontsize=13, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 8. Mismatch Magnitude
    ax8 = plt.subplot(3, 4, 8)
    data = [trials_df[trials_df['mismatch_magnitude'] == 0]['corrections'],
            incongruent[incongruent['mismatch_magnitude'] == 1]['corrections'],
            incongruent[incongruent['mismatch_magnitude'] == 2]['corrections']]
    bp = ax8.boxplot(data, labels=['0-step\n(Match)', '1-step', '2-step'], 
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightyellow')
    bp['boxes'][2].set_facecolor('lightcoral')
    ax8.set_ylabel('Corrections', fontsize=11)
    ax8.set_title('Mismatch Magnitude', fontsize=13, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('TRIAL_LEVEL_VISUALIZATIONS.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: TRIAL_LEVEL_VISUALIZATIONS.png")
    plt.show()

def main():
    trials_df = load_all_participants()
    
    if trials_df is None or len(trials_df) == 0:
        return
    
    descriptive_statistics(trials_df)
    create_visualizations(trials_df)
    
    # Save master dataset
    trials_df.to_csv('ALL_TRIALS_MASTER.csv', index=False)
    print("\n💾 Saved: ALL_TRIALS_MASTER.csv")
    
    print("\n" + "="*80)
    print("✅ DESCRIPTIVE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
