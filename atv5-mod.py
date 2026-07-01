import numpy as np

def func(x):
    return x[0]**2 + 2*x[1]**2 - x[0]*x[1]


def gradient(x):
    return np.array([
        2*x[0] - x[1],
        4*x[1] - x[0]
    ])


def constraint(x):
    return np.array([
        1.0 - x[0]
    ])


def residuals(x, z, mu):
    Jg = np.array([[-1.0, 0.0]])
    r_stat = gradient(x) + Jg.T @ z
    r_comp = constraint(x) * z + mu
    return r_stat, r_comp


def kkt_error(x, z):
    r_stat, r_comp = residuals(x, z, 0.0)
    return np.linalg.norm(np.concatenate([r_stat, r_comp]))


def newton(x, z, mu):
    g_val = constraint(x)
    Jg = np.array([
        [-1.0, 0.0]
    ])
    H = np.array([
        [2.0, -1.0],
        [-1.0, 4.0]
    ])

    Z = np.diag(z)
    G = np.diag(g_val)

    M = np.block([
        [H, Jg.T],
        [Z @ Jg, G]
    ])

    r_stat, r_comp = residuals(x, z, mu)
    rhs = np.concatenate([
        -r_stat,
        -r_comp
    ])
    return M, rhs


def fraction_to_boundary(x, z, dx, dz, tau=0.995):
    alpha = 1.0

    for i in range(len(z)):
        if dz[i] < 0:
            alpha = min(alpha, -tau * z[i] / dz[i])
    g_val = constraint(x)
    Jg = np.array([
        [-1.0, 0.0]
    ])
    dg = Jg @ dx
    for i in range(len(g_val)):
        if dg[i] > 0:
            alpha = min(alpha, -tau * g_val[i] / dg[i])
    return min(alpha, 1.0)


def central_path_test():
    x = np.array([1.25, 0.25])
    z = np.array([2.0])
    mu = 0.5

    grad_f = gradient(x)
    grad_g = np.array([
        -1.0,
        0.0
    ])

    r_stat, r_comp = residuals(x, z, mu)
    print(f"x = {x}")
    print(f"z = {z[0]:.6f}")
    print(f"mu = {mu:.6f}\n")
    print("grad f =")
    print(grad_f)
    print("\ngrad g =")
    print(grad_g)
    print("\nresíduo de estacionaridade =")
    print(r_stat)
    print("\n||r_stat|| = {:.6e}".format(np.linalg.norm(r_stat)))
    print("\nresíduo de complementaridade =")
    print(r_comp)
    print("||r_comp|| = {:.6e}".format(np.linalg.norm(r_comp)))
    print("\nPertence exatamente ao Caminho Central?")
    print("Estacionaridade:",
          np.linalg.norm(r_stat) < 1e-12)
    print("Complementaridade:",
          np.linalg.norm(r_comp) < 1e-12)

    print()


def ipm():
    x = np.array([2.0, 2.0])
    z = np.array([1.0])
    mu = 1.0
    mu_factor = 0.2
    tol = 1e-6
    max_iter_outer = 50
    max_iter_inner = 20
    header = (
        f"{'iter':>4}"
        f"{'x1':>12}"
        f"{'x2':>12}"
        f"{'z':>12}"
        f"{'alpha':>12}"
        f"{'mu':>12}"
        f"{'f(x)':>12}"
        f"{'erro KKT':>14}"
    )

    print(f"""
OTIMIZADOR PRIMAL-DUAL DE PONTOS INTERIORES

{header}
""")
    k = 0
    ekkt = kkt_error(x, z)

    print(
        f"{k:>4}"
        f"{x[0]:>12.6f}"
        f"{x[1]:>12.6f}"
        f"{z[0]:>12.6f}"
        f"{1.0:>12.6f}"
        f"{mu:>12.6f}"
        f"{func(x):>12.6f}"
        f"{ekkt:>14.2e}"
    )

    while ekkt > tol and k < max_iter_outer:
        alpha = 1.0

        for _ in range(max_iter_inner):
            M, rhs = newton(x, z, mu)

            try:
                delta = np.linalg.solve(M, rhs)

            except np.linalg.LinAlgError:
                print("Sistema singular.")
                break

            dx = delta[:len(x)]
            dz = delta[len(x):]
            alpha = fraction_to_boundary(
                x,
                z,
                dx,
                dz
            )

            x = x + alpha * dx
            z = z + alpha * dz
            r_s, r_c = residuals(x, z, mu)
            if np.linalg.norm(np.concatenate([r_s, r_c])) < 1e-8:
                break

        mu *= mu_factor
        k += 1
        ekkt = kkt_error(x, z)

        print(
            f"{k:>4}"
            f"{x[0]:>12.6f}"
            f"{x[1]:>12.6f}"
            f"{z[0]:>12.6f}"
            f"{alpha:>12.6f}"
            f"{mu:>12.6f}"
            f"{func(x):>12.6f}"
            f"{ekkt:>14.2e}"
        )

    print(f"""
SOLUÇÃO FINAL
x* = [{x[0]:.8f}, {x[1]:.8f}]
z* = {z[0]:.8f}
f(x*) = {func(x):.8f}
g(x*) = {constraint(x)[0]:.8e}
Erro KKT = {ekkt:.2e}
Estado = {"CONVERGIU" if ekkt <= tol else "NÃO CONVERGIU"}
""")

    return x, z


def check_kkt(x, z):

    print("""
VERIFICAÇÃO DAS CONDIÇÕES KKT
""")

    g_val = constraint(x)
    Jg = np.array([
        [-1.0, 0.0]
    ])

    grad = gradient(x)
    stat = grad + Jg.T @ z
    comp = g_val * z

    print("Estacionaridade")
    print(stat)
    print("||r_stat|| = {:.6e}\n".format(np.linalg.norm(stat)))
    print("Complementaridade")
    print(comp)
    print("||r_comp|| = {:.6e}\n".format(np.linalg.norm(comp)))
    print("Viabilidade primal")
    print(g_val)
    print("\nDual positivo")
    print(z > 0)
    print("\nErro KKT")
    print("{:.6e}".format(kkt_error(x, z)))



if __name__ == "__main__":

    central_path_test()
    x_opt, z_opt = ipm()
    check_kkt(x_opt, z_opt)
