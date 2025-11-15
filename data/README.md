# Dataset Instructions

## 📥 Downloading the UNSW-NB15 Dataset

### Option 1: Kaggle (Recommended)
1. Visit the Kaggle dataset page: [UNSW-NB15 Dataset](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
2. Click "Download" (you may need to create a free Kaggle account)
3. Extract the downloaded files to this `data/` folder

### Option 2: Official Source
1. Visit the official UNSW website: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download the CSV files (typically `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`)
3. Place all CSV files in this directory

## 📂 Expected File Structure

After downloading, your `data/` folder should look like:
```
data/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv
├── UNSW-NB15_features.csv  (optional - feature descriptions)
└── README.md (this file)
```

## 📋 Dataset Overview

- **Total Records:** ~257,673 (combined from all 4 files)
- **Features:** 49 columns
  - `id`: Record identifier
  - `dur`: Duration of connection
  - `proto`: Protocol type (tcp, udp, etc.)
  - `service`: Service type (http, ftp, ssh, etc.)
  - `state`: Connection state
  - `spkts`, `dpkts`: Source/destination packets
  - `sbytes`, `dbytes`: Source/destination bytes
  - ... and 40 more features
  - `label`: Binary target (0 = Normal, 1 = Attack)
  - `attack_cat`: Attack category (for reference, but we use binary classification)

## ⚠️ Important Notes

1. **File Size:** The CSV files are ~100MB each. Git is configured to ignore them (see `.gitignore`)
2. **Combining Files:** You'll need to concatenate all 4 CSV files in the preprocessing notebook
3. **Missing Values:** Some columns may have missing values - handle them appropriately
4. **Label Column:** Use the `label` column for binary classification (not `attack_cat`)

## 🔗 Resources

- **Original Paper:** Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems.
- **Feature Descriptions:** https://research.unsw.edu.au/projects/unsw-nb15-dataset (see "Feature Descriptions" section)

## 🛠️ Quick Load Code

```python
import pandas as pd
import glob

# Load all CSV files
data_files = glob.glob('data/UNSW-NB15_*.csv')
dfs = [pd.read_csv(file) for file in sorted(data_files)]
df = pd.concat(dfs, ignore_index=True)

print(f"Total records: {len(df)}")
print(f"Features: {df.shape[1]}")
```

---

**Need Help?** Check the first notebook (`01_data_exploration_eda.ipynb`) for detailed data loading examples.
