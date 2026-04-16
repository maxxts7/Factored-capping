import csv
from collections import Counter
from pathlib import Path
import numpy as np

N_BOOTSTRAP = 10000
SEED = 42

files = [
    ('assistant_cap_jailbreak', 'Final Results/assistant_cap_jailbreak_reclassified.csv'),
    ('assistant_cap_benign', 'Final Results/assistant_cap_benign_reclassified.csv'),
    ('cross_cap_jailbreak', 'Final Results/cross_cap_jailbreak_reclassified.csv'),
    ('cross_cap_benign', 'Final Results/cross_cap_benign_reclassified.csv'),
]


def bootstrap_ci(labels_list, target_label, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute 95% confidence interval for the rate of target_label via bootstrapping."""
    rng = np.random.default_rng(seed)
    binary = np.array([1 if l == target_label else 0 for l in labels_list])
    n = len(binary)
    if n == 0:
        return 0.0, 0.0
    rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(binary, size=n, replace=True)
        rates[i] = sample.mean()
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


for name, filepath in files:
    labels = Counter()
    labels_list = []
    corrections = Counter()
    layers = Counter()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)

    for row in rows:
        label = row.get('llm_label', '').strip()
        # Rows where capping didn't fire have no label -- treat as unchanged
        if not label:
            label = 'no_intervention'
        labels[label] += 1
        labels_list.append(label)

        correction = row.get('correction_applied', '').strip()
        if correction:
            corrections[correction] += 1

        layer = row.get('layers', '').strip()
        if layer:
            layers[layer] += 1

    print('\n' + '='*70)
    print(name.upper())
    print('='*70)
    print(f'Total Rows: {total}\n')

    print('LABELS (with 95% bootstrap CI):')
    for label, count in labels.most_common():
        pct = 100*count/total if total > 0 else 0
        lo, hi = bootstrap_ci(labels_list, label)
        print(f'  {label:25s}: {count:5d} ({pct:5.1f}%)  95% CI: [{lo*100:5.1f}%, {hi*100:5.1f}%]')

    if not labels:
        print('  [NO LABELS FOUND]')

    print('\nCORRECTION_APPLIED:')
    for val, count in corrections.most_common():
        pct = 100*count/total if total > 0 else 0
        print(f'  {val:25s}: {count:5d} ({pct:5.1f}%)')

    if not corrections:
        print('  [NO CORRECTION DATA]')

    print('\nTOP 10 LAYER PATTERNS:')
    for i, (pattern, count) in enumerate(layers.most_common(10), 1):
        pct = 100*count/total if total > 0 else 0
        print(f'  {i:2d}. {pattern:35s}: {count:5d} ({pct:5.1f}%)')

    if not layers:
        print('  [NO LAYER DATA]')
