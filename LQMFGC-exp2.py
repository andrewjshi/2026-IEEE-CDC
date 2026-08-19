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
    r'$(\lambda_1,\lambda_2)=(1,0)$ Coop-State': (1.0, 0.0),
    r'$(\lambda_1,\lambda_2)=(0,1)$ Coop-Adjustment': (0.0, 1.0),
    r'$(\lambda_1,\lambda_2)=(1,1)$ MFC':         (1.0, 1.0),
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

# ── Solver ──────────────────────────────────────────────────────────────────
def solve(lam1, lam2):
    denom_Pi = C + Ka*(1 - 2*lam2 + 2*lam2**2)

    P = np.zeros(steps + 1)
    for i in range(steps - 1, -1, -1):
        dP = -(P[i+1] + Kx/2 - 2*P[i+1]**2 / (C + Ka))
        P[i] = P[i+1] - dP * dt

    Pi = np.zeros(steps + 1)
    for i in range(steps - 1, -1, -1):
        state_term = Kx*(0.5 - lam1*(1-lam1))
        dPi = -(P[i+1] + state_term - 2*Pi[i+1]**2 / denom_Pi)
        Pi[i] = Pi[i+1] - dPi * dt

    Nsum = Ka*(1-lam2)*(2*lam2-1)
    Msum = Kx*(1-lam1)*(2*lam1-1)
    mean_denom = C + Ka*lam2

    # Corrected two-point boundary-value problem, solved by linear shooting.
    def sweep(z0, phi0):
        z = np.zeros(steps + 1)
        phi = np.zeros(steps + 1)
        z[0], phi[0] = z0, phi0
        for i in range(steps):
            a_i = -2.0 / mean_denom * (Pi[i]*z[i] + phi[i])
            dphi = (2*Pi[i]/denom_Pi * (phi[i] + 0.5*Nsum*a_i)
                    - 0.5*Msum*z[i])
            z[i+1] = z[i] + a_i*dt
            phi[i+1] = phi[i] + dphi*dt
        return z, phi

    z_p, phi_p = sweep(X0, 0.0)
    z_h, phi_h = sweep(0.0, 1.0)
    shot = -phi_p[-1] / phi_h[-1]
    z = z_p + shot*z_h
    phi = phi_p + shot*phi_h
    a = -2.0 / mean_denom * (Pi*z + phi)

    # Common on-equilibrium welfare for every (lam1, lam2).
    variance = np.zeros(steps + 1)
    centered_gain = 2*P/(C + Ka)
    for i in range(steps):
        dvariance = ((1 - 2*centered_gain[i])*variance[i] + z[i]**2)
        variance[i+1] = variance[i] + dvariance*dt
    control_second_moment = a**2 + centered_gain**2*variance
    state_second_moment = z**2 + variance
    running_cost = ((C+Ka)/2*control_second_moment
                    + Kx/2*state_second_moment)
    cost = np.trapz(running_cost, dx=dt)
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
ax.set_xlabel(r'$\lambda_2$ (adjustment cooperation)')
ax.set_ylabel(r'$\lambda_1$ (state cooperation)')
for (l1, l2), mk in [((0,0),'o'),((1,0),'s'),((0,1),'^'),((1,1),'*')]:
    ax.plot(l2, l1, mk, color='black', markersize=9,
            markeredgecolor='white', markeredgewidth=0.5)
fig.savefig('exp2_poa_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Done.")
