import numpy as np
import matplotlib.pyplot as plt

def solve_experiment_1(T=1.0, X0=1.0, lams=None, dt=0.001):
    if lams is None:
        lams = np.linspace(0, 1, 25)
    steps = int(T / dt)
    t_grid = np.linspace(0, T, steps + 1)
    results = {}
    costs = []

    for lam in lams:
        P = np.zeros(steps + 1)
        Pi = np.zeros(steps + 1)
        for i in range(steps - 1, -1, -1):
            dP = P[i+1]**2 - P[i+1] - 0.5
            P[i] = P[i+1] - dP * dt
            
            const_term = 0.5 * (1 - lam)**2
            dPi = Pi[i+1]**2 - P[i+1] - const_term
            Pi[i] = Pi[i+1] - dPi * dt

        z = np.zeros(steps + 1)
        phi = np.zeros(steps + 1)
        a = np.zeros(steps + 1)
        z[0] = X0
        
        for i in range(steps):
            dz = -Pi[i] * z[i] - phi[i]
            z[i+1] = z[i] + dz * dt
            
            dphi = -0.5 * (1 - lam)**2 * z[i]
            phi[i+1] = phi[i] + dphi * dt
            a[i] = -Pi[i] * z[i] - phi[i]
        
        a[-1] = -Pi[-1] * z[-1] - phi[-1]
        results[lam] = {'z': z, 'a': a}
        costs.append(0.5 * Pi[0] * X0**2)

    optimal_cost = costs[-1]
    poa = [c / optimal_cost for c in costs]
    
    return t_grid, results, lams, poa

# Execute
lams_trajectory = [0, 0.25, 0.5, 0.75, 1]
lams_poa = np.linspace(0, 1, 25)

t, res, _, _ = solve_experiment_1(lams=lams_trajectory)
_, _, _, poa_vals = solve_experiment_1(lams=lams_poa)

# Plot 1: Mean State
fig, ax = plt.subplots(figsize=(5, 4))
for l in lams_trajectory:
    ls = '-' if l in (0, 1) else '--'
    if l == 0:
        label = r'$\lambda=0$ (MFG)'
    elif l == 1:
        label = r'$\lambda=1$ (MFC)'
    else:
        label = f'$\\lambda={l}$'
    ax.plot(t, res[l]['z'], linestyle=ls, label=label)
ax.set_xlabel("Time")
ax.legend()
fig.savefig("exp1_mean_state.pdf", bbox_inches='tight', dpi=300)
plt.close(fig)

# Plot 2: Mean Control
fig, ax = plt.subplots(figsize=(5, 4))
for l in lams_trajectory:
    ls = '-' if l in (0, 1) else '--'
    if l == 0:
        label = r'$\lambda=0$ (MFG)'
    elif l == 1:
        label = r'$\lambda=1$ (MFC)'
    else:
        label = f'$\\lambda={l}$'
    ax.plot(t, res[l]['a'], linestyle=ls, label=label)
ax.set_xlabel("Time")
ax.legend()
fig.savefig("exp1_mean_control.pdf", bbox_inches='tight', dpi=300)
plt.close(fig)

# Plot 3: Price of Anarchy
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(lams_poa, poa_vals, '-', color='black')
#ax.set_title("Price of Anarchy ($J_{MFCG} / J_{MFC}$)")
ax.set_xlabel("$\\lambda$")
fig.savefig("exp1_price_of_anarchy.pdf", bbox_inches='tight', dpi=300)
plt.close(fig)

print("Done.")

