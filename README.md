# Effects of Visual-Haptic Weight Congruence on Perceived Realism and Performance in VR Object Manipulation

## 📋 Study Overview
This study investigates how visuo-haptic weight congruence affects VR object manipulation performance and perceived realism. Using a 3×3 within-subjects design (3 visual weight cues × 3 haptic weight cues), we examined how mismatches between visual appearance and haptic feedback influence movement behavior and subjective experience.

**Key Finding:** While congruent conditions received higher realism ratings, they showed no performance advantage. The critical finding was an **asymmetric mismatch effect**: when objects looked heavy but felt light (underestimation), participants made 2.8× more corrections and exhibited 26% lower path efficiency compared to the reverse mismatch (overestimation).

## 🔬 Hypotheses Tested

| Hypothesis | Description | Result |
|------------|-------------|---------|
| **H1** | Congruent visual–haptic conditions will yield better performance than incongruent conditions | **Not supported** |
| **H2** | Underestimation mismatch (looks heavy/feels light) will cause more performance degradation than overestimation mismatch (looks light/feels heavy) | **Strongly supported** |
| **H3** | Congruent conditions will be rated as more realistic than incongruent conditions | **Strongly supported** |

## 📊 Repository Structure
```text
├── Data/
│   ├── P*.csv                          # Individual participant data files
│   ├── ALL_TRIALS_MASTER.csv           # Combined trial-level dataset
│   └── TRIAL_LEVEL_DESCRIPTIVE_STATS.csv # Descriptive statistics
│
├── Scripts/
│   ├── 1_descriptive_analysis.py       # Trial-level exploratory analysis
│   ├── 2_inferential_analysis.py       # Participant-level hypothesis testing
│   └── realism.py                     # Realism ratings analysis
│
├── Results/
│   ├── TRIAL_LEVEL_VISUALIZATIONS.png  # All descriptive plots
│   └── PARTICIPANT_LEVEL_RESULTS.png  # Inferential results plots
│
└── README.md                           # This file
```



## 🔧 Analysis Pipeline

### **Step 1: Descriptive Analysis (`1_descriptive_analysis.py`)**
- **Purpose**: Exploratory trial-level analysis and visualizations
- **Input**: Individual `P*.csv` participant files
- **Output**: 
  - `ALL_TRIALS_MASTER.csv` - Combined dataset
  - `TRIAL_LEVEL_DESCRIPTIVE_STATS.csv` - Descriptive statistics
  - `TRIAL_LEVEL_VISUALIZATIONS.png` - Comprehensive visualizations
- **Note**: Statistics in this script are descriptive only – not for hypothesis testing

### **Step 2: Inferential Analysis (`2_inferential_analysis.py`)**
- **Purpose**: Proper repeated-measures statistical testing with participant-level aggregation
- **Input**: `ALL_TRIALS_MASTER.csv` (from Step 1)
- **Tests performed**:
  - Omnibus 3×3 repeated-measures ANOVA
  - H1: Congruent vs. Incongruent conditions (paired t-tests)
  - H2: Underestimation vs. Overestimation mismatch (paired t-tests)
  - Exploratory: Mismatch magnitude effects
- **Output**:
  - Console output with full statistical reporting
  - `PARTICIPANT_LEVEL_RESULTS.png` - Visualization of participant-level effects
- **Important**: This is the **correct** analysis for your paper's results section

### **Step 3: Realism Analysis (`realism.py`)**
- **Purpose**: Analysis of subjective realism ratings
- **Tests**: Congruence effects on perceived realism, detection accuracy


## Author
- **Younes Trichine**
