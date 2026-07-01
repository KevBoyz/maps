import numpy as np


def func(x): return x[0]**2 + 2*x[1]**2 - x[0]*x[1]
def gradient(x): return np.array([2*x[0] - x[1], 4*x[1] - x[0]])
def constraint(x): return np.array([1.0 - x[0]])


def residuals(x, z, mu): return (
    gradient(x) + np.array([[-1.0, 0.0]]).T @ z,
    constraint(x) * z + mu
)


def kkt_error(x, z): return np.linalg.norm(
    np.concatenate([residuals(x, z, 0.0)[0], residuals(x, z, 0.0)[1]])
)


def newton(x, z, mu):
    g_val = constraint(x)
    Jg = np.array([[-1.0, 0.0]])
    H = np.array([[2.0, -1.0], [-1.0, 4.0]])
    Z = np.diag(z)
    G = np.diag(g_val)
    M = np.block([[H, Jg.T], [Z @ Jg, G]])
    r_stat, r_comp = residuals(x, z, mu)
    rhs = np.concatenate([-r_stat, -r_comp])
    return M, rhs


def fraction_to_boundary(x, z, dx, dz, tau=0.995):
    alpha = 1.0
    for i in range(len(z)):
        if dz[i] < 0:
            alpha = min(alpha, -tau * z[i] / dz[i])
    g_val = constraint(x)
    Jg = np.array([[-1.0, 0.0]])
    dg = Jg @ dx

    for i in range(len(g_val)):
        if dg[i] > 0:
            alpha = min(alpha, -tau * g_val[i] / dg[i])

    return min(alpha, 1.0)


def ipm():
    x = np.array([2.0, 2.0])
    z = np.array([1.0])
    mu = 1.0

    mu_factor = 0.2
    tol = 1e-6
    max_iter_outer = 50
    max_iter_inner = 20

    header = (f"{'iter':>4}  {'x1':>10}  {'x2':>10}  {'z':>10}"
              f"  {'mu':>10}  {'residuo kkt':>12}")
    print(f"""
    otimizador primal-dual de pontos internos
{header}
""")

    k = 0
    ekkt = kkt_error(x, z)

    print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
          f"  {mu:>10.6f}  {ekkt:>12.2e}")

    while ekkt > tol and k < max_iter_outer:

        for _ in range(max_iter_inner):
            M, rhs = newton(x, z, mu)

            try:
                delta = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                print("sistema singular - parando iteracao interna")
                break

            dx = delta[:len(x)]
            dz = delta[len(x):]

            alpha = fraction_to_boundary(x, z, dx, dz)

            x = x + alpha * dx
            z = z + alpha * dz

            r_s, r_c = residuals(x, z, mu)
            inner_res = np.linalg.norm(np.concatenate([r_s, r_c]))
            if inner_res < 1e-8:
                break

        mu = mu_factor * mu
        k += 1
        ekkt = kkt_error(x, z)

        print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
              f"  {mu:>10.6f}  {ekkt:>12.2e}")

    print(f"""

{'solucao final'}
x final = {x[0]:.8f}, {x[1]:.8f}
z final = {z[0]:.8f}
valor f = {func(x):.8f}
valor g = {constraint(x)[0]:.8f}
residuo kkt final {ekkt:.2e}
estado: {('convergiu' if ekkt <= tol else 'nao convergiu')} (tol = {tol:.0e})

""")

    return x, z


def check_kkt(x, z):
    print("condicoes kkt")
    print("-" * 45)
    g_val = constraint(x)
    Jg = np.array([[-1.0, 0.0]])
    gf = gradient(x)
    stat = gf + Jg.T @ z
    comp = g_val * z
    feas = g_val
    print(f"""estacionaridade: grad f + z grad g = {stat}
complementaridade: g * z = {comp[0]:.2e}
viabilidade primal g(x*) = {feas[0]:.6f}
dual z* > 0? = {z[0] > 0}
""")


if __name__ == "__main__":
    x_opt, z_opt = ipm()
    check_kkt(x_opt, z_opt)