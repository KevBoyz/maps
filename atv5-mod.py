import numpy as np


def funcao(x): return x[0]**2 + 2*x[1]**2 - x[0]*x[1]
def gradiente(x): return np.array([2*x[0] - x[1], 4*x[1] - x[0]])
def restricao(x): return np.array([1.0 - x[0]])


def residuos(x, z, mu): return (
    gradiente(x) + np.array([[-1.0, 0.0]]).T @ z,
    restricao(x) * z + mu
)


def kkt_erro(x, z): return np.linalg.norm(
    np.concatenate([residuos(x, z, 0.0)[0], residuos(x, z, 0.0)[1]])
)


def newton(x, z, mu):
    gv = restricao(x)
    Jg = np.array([[-1.0, 0.0]])
    H = np.array([[2.0, -1.0], [-1.0, 4.0]])
    Z = np.diag(z)
    G = np.diag(gv)
    M = np.block([[H, Jg.T], [Z @ Jg, G]])
    r_estac, r_compl = residuos(x, z, mu)
    rhs = np.concatenate([-r_estac, -r_compl])
    return M, rhs


def fracao_para_fronteira(x, z, dx, dz, tau=0.995):
    alpha = 1.0
    for i in range(len(z)):
        if dz[i] < 0:
            alpha = min(alpha, -tau * z[i] / dz[i])
    gv = restricao(x)
    Jg = np.array([[-1.0, 0.0]])
    dg = Jg @ dx

    for i in range(len(gv)):
        if dg[i] > 0:
            alpha = min(alpha, -tau * gv[i] / dg[i])

    return min(alpha, 1.0)


def pdi():
    x = np.array([2.0, 2.0])
    z = np.array([1.0])
    mu = 1.0

    mu_fator = 0.2
    tol = 1e-6
    max_iter_ext = 50
    max_iter_int = 20

    cabecalho = (f"{'iter':>4}  {'x1':>10}  {'x2':>10}  {'z':>10}"
                 f"  {'mu':>10}  {'residuo kkt':>12}")
    separador = "=" * len(cabecalho)
    print(f"""
    otimizador primal-dual de pontos internos
{cabecalho}
{separador}
""")

    k = 0
    ekkt = kkt_erro(x, z)

    print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
          f"  {mu:>10.6f}  {ekkt:>12.2e}")

    while ekkt > tol and k < max_iter_ext:

        for _ in range(max_iter_int):
            M, rhs = newton(x, z, mu)

            try:
                delta = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                print("sistema singular - parando iteracao interna")
                break

            dx = delta[:len(x)]
            dz = delta[len(x):]

            alpha = fracao_para_fronteira(x, z, dx, dz)

            x = x + alpha * dx
            z = z + alpha * dz

            r_e, r_c = residuos(x, z, mu)
            res_interno = np.linalg.norm(np.concatenate([r_e, r_c]))
            if res_interno < 1e-8:
                break

        mu = mu_fator * mu
        k += 1
        ekkt = kkt_erro(x, z)

        print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
              f"  {mu:>10.6f}  {ekkt:>12.2e}")

    print(f"""
{separador}

{'solucao final':^{len(separador)}}
{separador}
x final = {x[0]:.8f}, {x[1]:.8f}
z final = {z[0]:.8f}
valor f = {funcao(x):.8f}
valor g = {restricao(x)[0]:.8f}
residuo kkt final {ekkt:.2e}
estado: {('convergiu' if ekkt <= tol else 'nao convergiu')} (tol = {tol:.0e})
{"="*len(separador)}

""")

    return x, z


def checar_kkt(x, z):
    print("condicoes kkt")
    print("-" * 45)
    gv = restricao(x)
    Jg = np.array([[-1.0, 0.0]])
    gf = gradiente(x)
    estac = gf + Jg.T @ z
    compl = gv * z
    viab = gv
    print(f"""estacionaridade: grad f + z grad g = {estac}
complementaridade: g * z = {compl[0]:.2e}
viabilidade primal g(x*) = {viab[0]:.6f}
dual z* > 0? = {z[0] > 0}
""")


if __name__ == "__main__":
    x_opt, z_opt = pdi()
    checar_kkt(x_opt, z_opt)
