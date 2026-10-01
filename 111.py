import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.cluster import KMeans

def add_temporal_features(df, channel_prefix, dates, new_prefix):
    values = []
    for date in dates:
        col = f"{channel_prefix}_{date}"
        if col in df.columns:
            values.append(df[col].values)
        else:
            return df
    values = np.array(values)
    df[f'{new_prefix}_mean'] = np.mean(values, axis=0).astype('float32')
    df[f'{new_prefix}_std'] = np.std(values, axis=0).astype('float32')
    df[f'{new_prefix}_delta'] = (values[-1] - values[0]).astype('float32')
    df[f'{new_prefix}_slope'] = ((values[-1] - values[0]) / (len(dates) - 1)).astype('float32')
    return df

cols = [
    'x', 'y', 'class', 'tile_id',
    'S2_B02', 'S2_B03', 'S2_B04', 'S2_B05', 'S2_B08', 'S2_B11', 'S2_B12',
    'S1_VV_20250601', 'S1_VH_20250601',
    'S1_VV_20250603', 'S1_VH_20250603',
    'S1_VV_20250604', 'S1_VH_20250604',
    'S1_VV_20250610', 'S1_VH_20250610'
]

dtypes = {col: 'float32' for col in cols if col not in ['class', 'tile_id']}
dtypes['class'] = 'object'
dtypes['tile_id'] = 'object'

df = pd.read_csv("dataset (1).csv", usecols=cols, dtype=dtypes, engine='c')

eps = 1e-10
df['NDVI'] = (df['S2_B08'] - df['S2_B04']) / (df['S2_B08'] + df['S2_B04'] + eps)
df['NDVI_RE'] = (df['S2_B08'] - df['S2_B05']) / (df['S2_B08'] + df['S2_B05'] + eps)
df['SAVI'] = ((df['S2_B08'] - df['S2_B04']) / (df['S2_B08'] + df['S2_B04'] + 0.5)) * 1.5
df['NDWI'] = (df['S2_B03'] - df['S2_B08']) / (df['S2_B03'] + df['S2_B08'] + eps)
df['NDBI'] = (df['S2_B11'] - df['S2_B08']) / (df['S2_B11'] + df['S2_B08'] + eps)
df['S1_Ratio_log'] = np.log1p(df['S1_VV_20250610'] / (df['S1_VH_20250610'] + eps))

df['target'] = (df['class'] == 'vegetation').astype(int)

dates = ['20250601', '20250603', '20250604', '20250610']
df = add_temporal_features(df, 'S1_VV', dates, 'S1T_VV')
df = add_temporal_features(df, 'S1_VH', dates, 'S1T_VH')

features = [
    'S2_B02', 'S2_B03', 'S2_B04', 'S2_B05', 'S2_B08', 'S2_B11', 'S2_B12',
    'NDVI', 'SAVI', 'NDWI', 'NDBI', 'S1_Ratio_log',
    'S1_VV_20250610', 'S1_VH_20250610',
    'S1T_VV_mean', 'S1T_VV_std', 'S1T_VV_delta', 'S1T_VV_slope',
    'S1T_VH_mean', 'S1T_VH_std', 'S1T_VH_delta', 'S1T_VH_slope',
]

features = [f for f in features if f in df.columns]

valid_mask = df[features].notna().all(axis=1)
df_valid = df[valid_mask].copy()

X = df_valid[features].values
y = df_valid['target'].values

if 'tile_id' in df_valid.columns and df_valid['tile_id'].nunique() > 1:
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df_valid['tile_id']))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['bare_soil', 'vegetation']))
print(f"F1_macro: {f1_score(y_test, y_pred, average='macro'):.3f}")

feat_imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print(feat_imp.head(10).to_string(index=False))