import sys
import site
# Add user site-packages so managed service can find ccxt/lightgbm etc.
sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, '/home/nebula/nyxml4')
from ml import data_fetcher, features as feat_eng, trainer, model_store

print('Fetching 9 months of MEXC data...')
data = data_fetcher.fetch_all(months=9)
print(f'5m: {len(data["df5"])}, 15m: {len(data["df15"])}, 1h: {len(data["df1h"])}, cvd: {len(data["cvd"])}')

print('Building features...')
df_feat = feat_eng.build_features(data['df5'], data['df15'], data['df1h'], data['cvd'])
print(f'Feature matrix: {df_feat.shape}')
assert all(col in df_feat.columns for col in feat_eng.FEATURE_COLS), f'Missing feature columns: {[c for c in feat_eng.FEATURE_COLS if c not in df_feat.columns]}'
assert list(df_feat[feat_eng.FEATURE_COLS].columns) == feat_eng.FEATURE_COLS, f'Feature order mismatch: {list(df_feat.columns)}'
print('Feature order OK')

print('Training model...')
result = trainer.train(df_feat, slot='current')
print(f'Val WR: {result["val_wr"]:.4f}, threshold: {result["threshold"]:.3f}')
print(f'Test WR: {result["test_metrics"]["wr"]:.4f}, trades: {result["test_metrics"]["trades"]}')
print(f'Trades/day: {result["test_metrics"]["trades_per_day"]:.1f}')
print(f'Sample count: {result["test_metrics"]}')
print(f'Model saved: {model_store.has_model("current")}')

model = result['model']
fi = dict(zip(feat_eng.FEATURE_COLS, model.feature_importance(importance_type='gain')))
top = sorted(fi.items(), key=lambda x: -x[1])[:10]
print('Top 10 features by gain:')
for name, score in top:
    print(f'  {name}: {score:.1f}')
print('TRAINING_COMPLETE')
