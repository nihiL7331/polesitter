import matplotlib.pyplot as plt
import pandas as pd

COL_TEXT = '#C9D1D9'
COL_BORDER = '#8B949E'
COL_NAIVE = '#D29922'
COL_FMM_ST = '#1F6FEB'
COL_FMM_MT = '#8957E5'
COL_BG = '#21262D'

plt.rcParams.update({
    'font.family': 'monospace',
    'font.monospace': ['BigBlueTerm437 Nerd Font Mono'],
    'text.color': COL_TEXT,
    'axes.titlecolor': COL_TEXT,
    'axes.labelcolor': COL_TEXT,
    'axes.edgecolor': COL_BORDER,
    'xtick.color': COL_BORDER,
    'ytick.color': COL_BORDER,
    'grid.color': COL_BORDER,
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'legend.facecolor': COL_BG,
    'legend.edgecolor': COL_BORDER,
})

data = pd.read_csv('results.csv')
N = data['N']
t_direct = data['Direct_Time']
t_fmm_st = data['FMM_Time_ST']
t_fmm_mt = data['FMM_Time_MT']

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(N, t_direct, marker='o', linewidth=2.5, color=COL_NAIVE, label='naive $\mathcal{O}(N^2)$')
ax.plot(N, t_fmm_st, marker='o', linewidth=2.5, color=COL_FMM_ST, label='FMM $\mathcal{O}(N)$ 1-core')
ax.plot(N, t_fmm_mt, marker='o', linewidth=2.5, color=COL_FMM_MT, label='FMM $\mathcal{O}(N)$ 8-core')

ax.set_title('perf scaling: direct evaluation vs FMM', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('# of particles (N)', fontsize=12)
ax.set_ylabel('exec time (secs)', fontsize=12)

ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=11, loc='upper left')

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('performance_graph.svg', dpi=300, transparent=True)

fig_log, ax_log = plt.subplots(figsize=(8, 5))

ax_log.plot(N, t_direct, marker='o', linewidth=2.5, color=COL_NAIVE,  label='naive $\mathcal{O}(N^2)$')
ax_log.plot(N, t_fmm_st, marker='o', linewidth=2.5, color=COL_FMM_ST, label='FMM $\mathcal{O}(N)$ 1-core')
ax_log.plot(N, t_fmm_mt, marker='o', linewidth=2.5, color=COL_FMM_MT, label='FMM $\mathcal{O}(N)$ 8-core')

ax_log.set_xscale('log')
ax_log.set_yscale('log')

ax_log.set_title('algorithmic scaling (log-log)', fontsize=14, pad=15, fontweight='bold')
ax_log.set_xlabel('# of particles (N)', fontsize=12)
ax_log.set_ylabel('exec time (secs)', fontsize=12)

ax_log.grid(True, which="both", linestyle='--', alpha=0.3)
ax_log.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('performance_graph_log.svg', dpi=300, transparent=True)
