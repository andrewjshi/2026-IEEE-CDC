import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────────────────────────────────
T   = 1.0
X0  = 1.0
dt  = 0.001
C   = 2.0
Ka  = 3.0
Kx  = 2.0
w   = 0.0

steps  = int(T / dt)
t_grid = np.linspace(0, T, steps + 1)

corners = {
    r'$(\lambda_1,\lambda_2)=(0,0)$ MFG':         (0.0, 0.0),
    r'$(\lambda_1,\lambda_2)=(1,0)$ Coop-Output': (1.0, 0.0),
    r'$(\lambda_1,\lambda_2)=(0,1)$ Coop-Effort': (0.0, 1.0),
    r'$(\lambda_1,\lambda_2)=(1,1)$ MFC':         (1.0, 1.0),
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

# ── Solver ──────────────────────────────────────────────────────────────────
def solve(lam1, lam2):
    denom_Pi = C + Ka*(1 - lam2)**2

    P = np.zeros(steps + 1)
    for i in range(steps - 1, -1, -1):
        dP = -(P[i+1] + Kx/2 - 2*P[i+1]**2 / (C + Ka))
        P[i] = P[i+1] - dP * dt

    Pi = np.zeros(steps + 1)
    for i in range(steps - 1, -1, -1):
        dPi = -(P[i+1] + Kx*(1-lam1)**2/2 - 2*Pi[i+1]**2 / denom_Pi)
        Pi[i] = Pi[i+1] - dPi * dt

    z   = np.zeros(steps + 1)
    phi = np.zeros(steps + 1)
    a   = np.zeros(steps + 1)
    z[0] = X0

    for i in range(steps):
        a[i]  = -2.0 / C * (Pi[i]*z[i] + phi[i])
        dphi  = (-Ka*(1-lam2)**2 * Pi[i] / denom_Pi * a[i]
                 - Kx*(1-lam1)**2/2 * z[i]
                 - w/2)
        z[i+1]   = z[i]   + a[i]  * dt
        phi[i+1] = phi[i] + dphi  * dt

    a[-1] = -2.0 / C * (Pi[-1]*z[-1] + phi[-1])
    cost = np.trapz(C/2 * a**2, dx=dt)
    return z, a, cost

# ── Compute trajectories ─────────────────────────────────────────────────────
traj = {}
for label, (l1, l2) in corners.items():
    z, a, c = solve(l1, l2)
    traj[label] = {'z': z, 'a': a, 'cost': c}

mfc_cost = traj[r'$(\lambda_1,\lambda_2)=(1,1)$ MFC']['cost']

# ── Figure 1: Mean state ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for (label, data), col in zip(traj.items(), colors):
    ls = '-' if label.endswith('MFG') or label.endswith('MFC') else '--'
    ax.plot(t_grid, data['z'], label=label, color=col, linestyle=ls)
ax.set_xlabel('Time')
ax.set_ylabel(r'$z_t$')
ax.legend(fontsize=7)
fig.savefig('exp2_mean_state.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)

# ── Figure 2: Mean control ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for (label, data), col in zip(traj.items(), colors):
    ls = '-' if label.endswith('MFG') or label.endswith('MFC') else '--'
    ax.plot(t_grid, data['a'], label=label, color=col, linestyle=ls)
ax.set_xlabel('Time')
ax.set_ylabel(r'$a_t$')
ax.legend(fontsize=7)
fig.savefig('exp2_mean_control.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)

# ── Figure 3: 2D PoA heatmap ──────────────────────────────────────────────────
N_grid   = 30
lam_vals = np.linspace(0, 1, N_grid)
PoA      = np.zeros((N_grid, N_grid))

for i, l1 in enumerate(lam_vals):
    for j, l2 in enumerate(lam_vals):
        _, _, c = solve(l1, l2)
        PoA[i, j] = c / mfc_cost

print(f"PoA range: [{PoA.min():.3f}, {PoA.max():.3f}]")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(PoA, origin='lower', extent=[0,1,0,1],
               aspect='auto', cmap='RdYlGn_r', vmin=1.0, vmax=PoA.max())
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$J_{MFCG}/J_{MFC}$')
ax.set_xlabel(r'$\lambda_2$ (effort cooperation)')
ax.set_ylabel(r'$\lambda_1$ (output cooperation)')
for (l1, l2), mk in [((0,0),'o'),((1,0),'s'),((0,1),'^'),((1,1),'*')]:
    ax.plot(l2, l1, mk, color='black', markersize=9,
            markeredgecolor='white', markeredgewidth=0.5)
fig.savefig('exp2_poa_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Done.")
