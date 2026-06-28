import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "out/2"
os.makedirs(OUT_DIR, exist_ok=True)

X0 = np.array([-1.2, 1.0])
TOL = 1e-6
MAX_ITER = 10_000
C = 1e-4
BETA = 0.9


def f(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def grad(x, y):
    return np.array([
        -400 * x * (y - x**2) + 2 * (x - 1),
        200 * (y - x**2),
    ])


def hes(x, y):
    h12 = -400 * x
    return np.array([
        [1200 * x**2 - 400 * y + 2, h12],
        [h12, 200],
    ])


def armijo(x, g, d):
    alpha = 1.0
    fx = f(x[0], x[1])
    slope = C * np.dot(g, d)
    nf = 0
    while f(x[0] + alpha * d[0], x[1] + alpha * d[1]) > fx + alpha * slope:
        alpha *= BETA
        nf += 2
    return alpha, nf


def _make_result(x, traj, f_hist, grad_hist, k, t0, nf, ng, nh):
    return dict(
        traj=traj, f_hist=f_hist, grad_hist=grad_hist,
        iters=k + 1, time=time.time() - t0,
        f_final=f(x[0], x[1]),
        grad_final=np.linalg.norm(grad(x[0], x[1])),
        nf=nf, ng=ng, nh=nh,
    )


def _record(x, traj, f_hist, grad_hist, g):
    traj.append(x.copy())
    f_hist.append(f(x[0], x[1]))
    grad_hist.append(np.linalg.norm(g))


def run_gd_armijo():
    x = X0.copy()
    traj, f_hist, grad_hist = [x.copy()], [], []
    nf = ng = 0
    t0 = time.time()
    for k in range(MAX_ITER):
        g = grad(x[0], x[1])
        ng += 1
        if np.linalg.norm(g) <= TOL:
            break
        d = -g
        alpha, step_nf = armijo(x, g, d)
        nf += step_nf
        x = x + alpha * d
        _record(x, traj, f_hist, grad_hist, g)
    return _make_result(x, traj, f_hist, grad_hist, k, t0, nf, ng, 0)


def run_newton():
    x = X0.copy()
    traj, f_hist, grad_hist = [x.copy()], [], []
    nf = ng = nh = 0
    t0 = time.time()
    for k in range(MAX_ITER):
        g = grad(x[0], x[1])
        H = hes(x[0], x[1])
        ng += 1
        nh += 1
        if np.linalg.norm(g) <= TOL:
            break
        d = -np.linalg.inv(H) @ g
        x = x + d
        nf += 1
        _record(x, traj, f_hist, grad_hist, g)
    return _make_result(x, traj, f_hist, grad_hist, k, t0, nf, ng, nh)


def run_bfgs():
    x = X0.copy()
    H = np.eye(len(x))
    traj, f_hist, grad_hist = [x.copy()], [], []
    nf = ng = 0
    t0 = time.time()
    for k in range(MAX_ITER):
        g = grad(x[0], x[1])
        ng += 1
        if np.linalg.norm(g) <= TOL:
            break
        d = -H @ g
        alpha, step_nf = armijo(x, g, d)
        nf += step_nf
        x_new = x + alpha * d
        s = x_new - x
        gn = grad(x_new[0], x_new[1])
        ng += 1
        y = gn - g
        rho = 1.0 / (y @ s)
        I = np.eye(len(x))
        H = ((I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s))
             + rho * np.outer(s, s))
        x = x_new
        _record(x, traj, f_hist, grad_hist, gn)
    return _make_result(x, traj, f_hist, grad_hist, k, t0, nf, ng, 0)


def _rosenbrock_mesh():
    X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
    return X, Y, 100 * (Y - X**2)**2 + (1 - X)**2


def plot_convergence(results, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r, lbl in zip(results, labels):
        axes[0].plot(r['f_hist'], label=lbl)
        axes[1].semilogy(r['grad_hist'], label=lbl)
    axes[0].set(title='f(xk) versus iteração',
                xlabel='Iteração', ylabel='f(xk)')
    axes[0].legend()
    axes[0].grid()
    axes[1].set(title='||∇f(xk)|| versus iteração',
                xlabel='Iteração', ylabel='Norma do gradiente')
    axes[1].legend()
    axes[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'convergencia.png'), dpi=150)
    plt.close()


def plot_trajectory(traj, title, filename):
    traj = np.array(traj)
    hx, hy = traj[:, 0], traj[:, 1]
    X, Y, Z = _rosenbrock_mesh()
    levels = np.logspace(-1, 3, 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.75)
    ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label='f(x, y)')
    ax.plot(hx, hy, '-o', markersize=4, linewidth=2,
            color='white', label='Trajetória')
    ax.scatter(hx[0], hy[0], s=100, color='cyan',
               zorder=5, label='Ponto Inicial')
    ax.scatter(hx[-1], hy[-1], s=100, color='lime',
               zorder=5, label='Ponto Final')
    ax.scatter(1, 1, marker='*', s=250, color='red', zorder=5, label='Ótimo')
    ax.set(title=title, xlabel='x', ylabel='y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


if __name__ == '__main__':
    res_gd = run_gd_armijo()
    res_newton = run_newton()
    res_bfgs = run_bfgs()

    LABELS = ['Gradiente Armijo', 'Newton', 'BFGS']
    RESULTS = [res_gd, res_newton, res_bfgs]

    plot_convergence(RESULTS, LABELS)
    plot_trajectory(
        res_gd['traj'], 'Gradiente Descendente - Armijo', 'traj_gd_armijo.png')
    plot_trajectory(res_newton['traj'], 'Método de Newton', 'traj_newton.png')
    plot_trajectory(res_bfgs['traj'], 'Método BFGS', 'traj_bfgs.png')

    tabela = pd.DataFrame({
        'Método': LABELS,
        'Tempo (s)': [r['time'] for r in RESULTS],
        'Iterações': [r['iters'] for r in RESULTS],
        'f(x final)': [r['f_final'] for r in RESULTS],
        '||∇f|| final': [r['grad_final'] for r in RESULTS],
        'Avaliações f': [r['nf'] for r in RESULTS],
    })

    print(tabela.to_string(index=False))
