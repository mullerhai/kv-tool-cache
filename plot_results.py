"""
Comprehensive plotting script for the KV-Tool Cache experiment results.
Reads `results.json` produced by experiments and writes PNG images into `images/`.

Generates figures for:
 - Exp1: similarity & redundancy
 - Exp2: average latency and low/high split
 - Exp3: hit rate decomposition
 - Exp4: average PPL
 - Exp5: latency vs context length
 - Exp6: ablation latency and hit-rate breakdown

Run:
    python plot_results.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

with open("results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# ------------------ Exp1: Similarity & Redundancy ------------------
exp1 = results.get('exp1', {})
fig, ax = plt.subplots(figsize=(6,4))
labels = ['Same-tool KV sim', 'Cross-tool KV sim']
vals = [exp1.get('avg_kv_sim', np.nan), exp1.get('avg_cross_kv_sim', np.nan)]
sns.barplot(x=labels, y=vals, palette='pastel', ax=ax)
ax.set_ylabel('Cosine similarity')
ax.set_ylim(0, 1)
ax.set_title('KV similarity: same-tool vs cross-tool')
for i,v in enumerate(vals):
    ax.text(i, v+0.02, f"{v:.3f}", ha='center')
meta = f"Repeat ratio: {exp1.get('repeat_ratio',0):.1%}\nKV redundancy: {exp1.get('kv_redundancy_pct',0):.1f}%\nAvg Jaccard: {exp1.get('avg_jaccard',0):.3f}"
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'exp1_similarity.png'), dpi=200)
plt.close(fig)

# ------------------ Exp2: Latency plots ------------------
exp2 = results.get('exp2', {})
methods = ["Vanilla", "KV Only", "Independent", "Co-design"]
latencies = [exp2.get('vanilla_avg', np.nan), exp2.get('kv_only_avg', np.nan), exp2.get('independent_avg', np.nan), exp2.get('codesign_avg', np.nan)]
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=methods, y=latencies, palette="muted", ax=ax)
ax.set_ylabel('Avg latency (s)')
ax.set_title('Per-method average latency (per-turn)')
for p, v in zip(ax.patches, latencies):
    ax.annotate(f"{v:.3f}s", (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'exp2_avg_latency.png'), dpi=200)
plt.close(fig)

# Low/high repeat grouped
labels = methods
low = [exp2.get('vanilla_low_avg', np.nan), exp2.get('kv_only_low_avg', np.nan), exp2.get('ind_low_avg', np.nan), exp2.get('co_low_avg', np.nan)]
high = [exp2.get('vanilla_high_avg', np.nan), exp2.get('kv_only_high_avg', np.nan), exp2.get('ind_high_avg', np.nan), exp2.get('co_high_avg', np.nan)]
fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(labels))
ax.bar(x-0.2, low, width=0.4, label='Low-repeat')
ax.bar(x+0.2, high, width=0.4, label='High-repeat')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Avg latency (s)')
ax.set_title('Latency by repeat type')
ax.legend()
for i,(l,h) in enumerate(zip(low,high)):
    ax.text(i-0.2, l+0.005, f"{l:.3f}", ha='center')
    ax.text(i+0.2, h+0.005, f"{h:.3f}", ha='center')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'exp2_latency_low_high.png'), dpi=200)
plt.close(fig)

# ------------------ Exp3: Hit rates decomposition ------------------
exp3 = results.get('exp3', {})
labels = ["Tool Hit Rate", "KV Reuse Rate (est)", "Joint Hit Rate"]
ind_tool = exp3.get('ind_tool_hit_rate', np.nan)
co_tool = exp3.get('co_tool_hit_rate', np.nan)
ind_kv = exp3.get('ind_kv_hit_rate', exp1.get('kv_redundancy_pct',0)/100 if exp1 else np.nan)
co_kv = exp3.get('co_kv_hit_rate', 1.0)
ind_joint = exp3.get('independent_joint_hit', np.nan)
co_joint = exp3.get('codesign_joint_hit', np.nan)
ind = [ind_tool, ind_kv, ind_joint]
co = [co_tool, co_kv, co_joint]
fig, ax = plt.subplots(figsize=(7,4))
width = 0.35
x = np.arange(len(labels))
ax.bar(x-width/2, [v*100 for v in ind], width=width, label='Independent')
ax.bar(x+width/2, [v*100 for v in co], width=width, label='Co-design')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Rate (%)')
ax.set_title('Cache Hit Rate Decomposition')
ax.legend()
for i,(a,b) in enumerate(zip(ind,co)):
    ax.text(i-width/2, a*100+1, f"{a*100:.1f}%", ha='center')
    ax.text(i+width/2, b*100+1, f"{b*100:.1f}%", ha='center')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'exp3_hit_rates.png'), dpi=200)
plt.close(fig)

# ------------------ Exp4: PPL comparison ------------------
exp4 = results.get('exp4', {})
ppl_v = exp4.get('avg_ppl_vanilla', np.nan)
ppl_c = exp4.get('avg_ppl_cached', np.nan)
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(['Vanilla','Co-design'], [ppl_v, ppl_c], color=['#4c72b0','#55a868'])
ax.set_ylabel('Avg PPL')
ax.set_title('Average Perplexity (Vanilla vs Co-design)')
for i,v in enumerate([ppl_v,ppl_c]):
    ax.text(i, v+5, f"{v:.1f}", ha='center')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR,'exp4_ppl.png'), dpi=200)
plt.close(fig)

# ------------------ Exp5: Scalability curves ------------------
exp5 = results.get('exp5', {})
ctx = exp5.get('context_lengths', [32,64,128,256,512])
ind_lats = exp5.get('independent_latencies', [])
co_lats = exp5.get('codesign_latencies', [])
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(ctx, ind_lats, marker='o', label='Independent')
ax.plot(ctx, co_lats, marker='o', label='Co-design')
ax.set_xscale('log')
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Avg latency (s)')
ax.set_title('Latency vs Context Length')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR,'exp5_scalability.png'), dpi=200)
plt.close(fig)

# ------------------ Exp6: Ablation (latency and hit rates) ------------------
exp6 = results.get('exp6', {})
configs = list(exp6.keys())
latencies = [exp6[c]['avg_latency'] for c in configs]
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=latencies, y=configs, orient='h', palette='mako', ax=ax)
ax.set_xlabel('Avg latency (s)')
ax.set_title('Ablation: Average Latency per Configuration')
for p,v in zip(ax.patches, latencies):
    ax.annotate(f"{v:.3f}s", (p.get_width()+0.001, p.get_y()+p.get_height()/2), va='center')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR,'exp6_ablation_latency.png'), dpi=200)
plt.close(fig)

# Ablation: rates
tool_rates = [exp6[c]['tool_hit_rate']*100 for c in configs]
kv_rates = [exp6[c]['kv_hit_rate']*100 for c in configs]
joint_rates = [exp6[c]['joint_hit_rate']*100 for c in configs]
fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(configs))
ax.bar(x-0.25, tool_rates, width=0.25, label='Tool HR')
ax.bar(x, kv_rates, width=0.25, label='KV HR')
ax.bar(x+0.25, joint_rates, width=0.25, label='Joint HR')
ax.set_xticks(x)
ax.set_xticklabels(configs)
plt.xticks(rotation=25)
ax.set_ylabel('Rate (%)')
ax.set_title('Ablation: Hit Rates by Configuration')
ax.legend()
for i,(t,k,j) in enumerate(zip(tool_rates, kv_rates, joint_rates)):
    ax.text(i-0.25, t+1, f"{t:.1f}%", ha='center')
    ax.text(i, k+1, f"{k:.1f}%", ha='center')
    ax.text(i+0.25, j+1, f"{j:.1f}%", ha='center')
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR,'exp6_ablation_rates.png'), dpi=200)
plt.close(fig)

print('[✓] Plots saved to images/ folder')

# ------------------ Exp7: Tool frequency & summary latency bars ------------------
try:
    # Parse DIALOGUE_DATASET from experiment_cache_codesign.py without importing the heavy module
    src = open('experiment_cache_codesign.py', 'r', encoding='utf-8').read()
    start = src.find('DIALOGUE_DATASET = [')
    if start != -1:
        arr_text = src[start: src.find(']', start) + 1]
        # crude extraction of tool entries
        lines = [ln.strip().lstrip('(').rstrip(',').rstrip(')') for ln in arr_text.splitlines() if ln.strip().startswith('(')]
        tools = []
        for ln in lines:
            # format: (0,  "search",     {"query": "KV cache in LLMs"},     "KV cache stores ...", 0),
            parts = ln.split(',')
            if len(parts) >= 3:
                tool_name = parts[1].strip().strip('"').strip("'")
                tools.append(tool_name)
        from collections import Counter
        freq = Counter(tools)
    else:
        freq = {}
except Exception:
    freq = {}

# plot tool freq bar
if freq:
    names = list(freq.keys())
    counts = [freq[n] for n in names]
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=counts, y=names, palette='cool', ax=ax)
    ax.set_xlabel('Call frequency')
    ax.set_title('Tool call frequency (dataset)')
    for p,c in zip(ax.patches, counts):
        ax.annotate(f"{c}x", (p.get_width()+0.05, p.get_y()+p.get_height()/2), va='center')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'exp7_tool_frequency.png'), dpi=200)
    plt.close(fig)

# Reuse exp2 latencies to draw per-turn latency bars for Exp7 summary
try:
    exp2 = results.get('exp2', {})
    methods = ["Vanilla", "KV Only", "Independent", "Co-design"]
    latencies = [exp2.get('vanilla_avg', np.nan), exp2.get('kv_only_avg', np.nan), exp2.get('independent_avg', np.nan), exp2.get('codesign_avg', np.nan)]
    fig, ax = plt.subplots(figsize=(7,3))
    sns.barplot(x=latencies, y=methods, palette='viridis', ax=ax)
    ax.set_xlabel('Avg latency (s)')
    ax.set_title('Average per-turn latency (summary)')
    for p, v in zip(ax.patches, latencies):
        ax.annotate(f"{v:.3f}s", (p.get_width()+0.001, p.get_y()+p.get_height()/2), va='center')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'exp7_latency_summary.png'), dpi=200)
    plt.close(fig)
except Exception:
    pass

print('[✓] Exp7 plots generated (if dataset parsed)')

