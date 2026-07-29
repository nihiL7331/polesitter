import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('results.csv')
N = data['N']
t_direct = data['Direct_Time']
t_fmm = data['FMM_Time']

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(N, t_direct, marker='o', linewidth=2.5, color='#ff5555', label='Naive $\mathcal{O}(N^2)$')
ax.plot(N, t_fmm, marker='o', linewidth=2.5, color='#50fa7b', label='polesitter FMM $\mathcal{O}(N)$')

ax.set_title('Performance Scaling: Direct Evaluation vs FMM', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Number of Particles (N)', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)

ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=11, loc='upper left')

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('performance_graph.png', dpi=300, transparent=True)
