import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

def rosenbrock(v):
    return 100.0 * (v[1] - v[0]**2)**2 + (1.0 - v[0])**2

x         = np.array([-1.2, 0.5])
delta     = 1.0
DELTA_MAX = 5.0
TOL_GRAD  = 1e-6
EPS       = 1e-12
MAX_ITER  = 2000

hist_x     = [x.copy()]
hist_delta = [delta]
hist_sk    = []
hist_acc   = []

cabecalho = (f"{'k':>5} | {'f(x)':>14} | {'||g||':>11} | "
             f"{'rho':>9} | {'Delta':>8} | Decisão")
print(f"\n{cabecalho}")
print("─" * 68)

converged = False
k_final   = 0

for k in range(MAX_ITER):
    k_final = k
    x0, x1 = x[0], x[1]
    fk = rosenbrock(x)

    gk = np.array([
        -400.0 * x0 * (x1 - x0**2) - 2.0 * (1.0 - x0),
         200.0 * (x1 - x0**2)
    ])
    norm_g = np.linalg.norm(gk)

    if norm_g < TOL_GRAD:
        print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
              f"{'---':>9} | {delta:>8.5f} | CONVERGIU ✓")
        converged = True
        break

    Bk = np.array([
        [-400.0*(x1 - x0**2) + 800.0*x0**2 + 2.0, -400.0*x0],
        [-400.0*x0,                                  200.0    ]
    ])

    gBg = gk @ Bk @ gk

    if gBg <= 0.0:
        sk = -(delta / norm_g) * gk
    else:
        alpha_bar = norm_g**2 / gBg
        alpha_k   = min(alpha_bar, delta / norm_g)
        sk        = -alpha_k * gk

    pred = -(gk @ sk + 0.5 * sk @ Bk @ sk)
    hist_sk.append(sk.copy())

    if abs(pred) < EPS:
        x = x + sk
        hist_acc.append(True)
        hist_x.append(x.copy())
        hist_delta.append(delta)
        print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
              f"{'denom≈0':>9} | {delta:>8.5f} | CONVERGIU ✓")
        converged = True
        break

    ared = fk - rosenbrock(x + sk)
    rho  = ared / pred

    if rho < 0.25:
        delta   = 0.25 * delta
        hist_acc.append(False)
        decisao = "REJEITA ▼"
    elif rho < 0.75:
        x       = x + sk
        hist_acc.append(True)
        decisao = "aceita  ─"
    else:
        x       = x + sk
        delta   = min(2.0 * delta, DELTA_MAX)
        hist_acc.append(True)
        decisao = "aceita  ▲"

    hist_x.append(x.copy())
    hist_delta.append(delta)

    print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
          f"{rho:>9.5f} | {delta:>8.5f} | {decisao}")

if not converged:
    print(f"\n! Limite de {MAX_ITER} iterações atingido sem convergência.")

x0f, x1f = x[0], x[1]
gf = np.array([
    -400.0*x0f*(x1f - x0f**2) - 2.0*(1.0 - x0f),
     200.0*(x1f - x0f**2)
])
print(f"\n{'═'*56}")
print(f"  Solução:   x* = ({x[0]:.10f},  {x[1]:.10f})")
print(f"  f(x*)    = {rosenbrock(x):.6e}")
print(f"  ‖g(x*)‖  = {np.linalg.norm(gf):.6e}")
print(f"{'═'*56}")

# Formar figura (só para frescar)

hist_x     = np.array(hist_x)
n_iters    = len(hist_sk)
n_accepted = sum(hist_acc)
n_rejected = n_iters - n_accepted

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('#08080e')
ax.set_facecolor('#08080e')

xi = np.linspace(-2.1, 1.8, 700)
yi = np.linspace(-0.9, 2.2, 700)
XI, YI = np.meshgrid(xi, yi)
ZI = 100.0*(YI - XI**2)**2 + (1.0 - XI)**2
ZI = np.clip(ZI, 0.05, None) # evita log(0) na norma

lv_fill = np.logspace(np.log10(0.05), np.log10(ZI.max()), 45)
cf = ax.contourf(XI, YI, ZI, levels=lv_fill,
                 norm=LogNorm(vmin=0.05, vmax=ZI.max()),
                 cmap='inferno', alpha=0.90, zorder=1)
ax.contour(XI, YI, ZI, levels=lv_fill, colors='white',
           linewidths=0.3, alpha=0.18, zorder=2)

cb = plt.colorbar(cf, ax=ax, shrink=0.72, pad=0.025, aspect=24)
cb.set_label('f(x₁, x₂)  [escala log]', color='#c8c8d8', fontsize=11)
cb.ax.yaxis.set_tick_params(color='#c8c8d8')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#c8c8d8')

# Primeiras 18 + amostras do restante — fade conforme índice
n_early   = min(18, n_iters)
c_indices = list(range(n_early))
if n_iters > n_early:
    stride     = max(1, (n_iters - n_early) // 10)
    c_indices += list(range(n_early, n_iters, stride))
c_indices = sorted(set(c_indices))[:30]

for rank, i in enumerate(c_indices):
    alpha_c = max(0.05, 0.55 * (1.0 - rank / max(1, len(c_indices) - 1)))
    circ = plt.Circle(hist_x[i], hist_delta[i],
                      fill=False, color='#00e5ff',
                      linewidth=0.85, linestyle='--',
                      alpha=alpha_c, zorder=3)
    ax.add_patch(circ)

rej_idx = [i for i, a in enumerate(hist_acc)
           if not a and np.linalg.norm(hist_sk[i]) > 5e-4]
if len(rej_idx) > 55:
    stride_r = len(rej_idx) // 55
    rej_idx  = rej_idx[::stride_r]

for i in rej_idx:
    ax.annotate('', xy=hist_x[i] + hist_sk[i], xytext=hist_x[i],
                arrowprops=dict(arrowstyle='->', color='#ff3b3b',
                                lw=0.85, mutation_scale=9),
                zorder=5)

ax.plot(hist_x[:, 0], hist_x[:, 1], '-',
        color='#3dff4a', lw=1.6, alpha=0.78, zorder=6)

step_a = max(1, n_iters // 22)
for i in range(0, n_iters - 1, step_a):
    if hist_acc[i]:
        disp = np.linalg.norm(hist_x[i+1] - hist_x[i])
        if disp > 3e-4:                 # omite micro-deslocamentos
            ax.annotate('', xy=hist_x[i+1], xytext=hist_x[i],
                        arrowprops=dict(arrowstyle='->',
                                        color='#3dff4a',
                                        lw=1.15, mutation_scale=11),
                        zorder=7)
            
ax.scatter(*hist_x[0], s=180, c='white', marker='s', zorder=12,
           edgecolors='#3dff4a', linewidths=1.8)
ax.text(hist_x[0, 0] + 0.07, hist_x[0, 1] - 0.14,
        r'$x_0$', color='white', fontsize=12,
        fontweight='bold', zorder=13)

ax.scatter(1.0, 1.0, s=600, c='#ffd700', marker='*', zorder=12,
           edgecolors='#ff9900', linewidths=1.1)
ax.text(1.09, 1.06, r'$x^{\!*}=(1,\;1)$',
        color='#ffd700', fontsize=11, fontweight='bold', zorder=13)

leg_elems = [
    Line2D([0],[0], marker='s', color='none',
           markerfacecolor='white', markeredgecolor='#3dff4a',
           markersize=9, label=r'Início  $x_0=(-1.2,\;0.5)$'),
    Line2D([0],[0], marker='*', color='none',
           markerfacecolor='#ffd700', markersize=15,
           label=r'Mínimo global  $x^*=(1,\;1)$'),
    Line2D([0],[0], color='#3dff4a', lw=2.0,
           label=f'Passos aceitos  ({n_accepted})'),
    Line2D([0],[0], color='#ff3b3b', lw=1.5,
           label=f'Passos rejeitados  ({n_rejected})'),
    Line2D([0],[0], color='#00e5ff', lw=1.5, linestyle='--',
           label=r'Regiões de confiança  $\Delta_k$'),
]
ax.legend(handles=leg_elems, loc='lower right',
          facecolor='#0e0e1a', edgecolor='#3a3a55',
          labelcolor='white', fontsize=9.5,
          framealpha=0.92, borderpad=0.85)

xf0  = f"{hist_x[-1, 0]:.8f}"
xf1  = f"{hist_x[-1, 1]:.8f}"
fval = f"{rosenbrock(hist_x[-1]):.3e}"
gval = f"{np.linalg.norm(gf):.3e}"
info = (f"Iterações: {k_final}\n"
        f"$x^*\\approx({xf0},\\,{xf1})$\n"
        f"$f(x^*)={fval}$     $\\|g\\|={gval}$")
ax.text(0.015, 0.015, info, transform=ax.transAxes,
        color='#a8ccee', fontsize=8.5, va='bottom', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0c1625',
                  edgecolor='#2a4060', alpha=0.92))

ax.set_xlim(-2.1, 1.8)
ax.set_ylim(-0.9, 2.2)
ax.set_xlabel(r'$x_1$', color='#c8c8d8', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x_2$', color='#c8c8d8', fontsize=13, labelpad=6)
ax.set_title(
    'Região de Confiança — Ponto de Cauchy\n'
    r'Rosenbrock: $f(x)=100\,(x_2-x_1^2)^2+(1-x_1)^2$',
    color='white', fontsize=13, fontweight='bold', pad=14)

ax.tick_params(colors='#888899')
for sp in ax.spines.values():
    sp.set_edgecolor('#22223a')

plt.tight_layout()
out = 'out/3/rosenbrock_trust_region.png'
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nGráfico salvo → {out}")
plt.close()