import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2 - x[0]*x[1]

def grad_f(x):
    return np.array([2*x[0] - x[1],
                     4*x[1] - x[0]])

def hess_f():
    return np.array([[2.0, -1.0],
                     [-1.0, 4.0]])

def g(x):
    return np.array([1.0 - x[0]])          # g(x) = 1 - x1

def grad_g(x):
    return np.array([[-1.0, 0.0]])          # Jg (1×2): linha = ∇g^T

def hess_L(x, lam, z):
    return hess_f()

def residuos(x, z, mu):
    gv  = g(x)          
    Jg  = grad_g(x)     
    gf  = grad_f(x)

    r_estac  = gf + Jg.T @ z         
    r_compl  = gv * z + mu             
    return r_estac, r_compl

def erro_kkt(x, z):
    """Erro KKT puro (µ = 0): norma dos resíduos sem perturbação."""
    r_e, r_c = residuos(x, z, mu=0.0)
    return np.linalg.norm(np.concatenate([r_e, r_c]))

# Montagem da matriz KKT assimétrica completa (eq. 8)
# Sem restrições de igualdade: sistema é (n+p) × (n+p)
#
#  [ ∇²L    Jg^T ] [ Δx ]   [ -r_estac ]
#  [ Z Jg   G    ] [ Δz ] = [ -r_compl ]

def sistema_newton(x, z, mu):
    gv = g(x)           # (p,)
    Jg = grad_g(x)      # (p, n)
    H  = hess_L(x, None, z)   # (n, n)

    Z  = np.diag(z)     # (p, p)
    G  = np.diag(gv)    # (p, p)

    n, p = x.shape[0], z.shape[0]

    M = np.block([
        [H,          Jg.T ],
        [Z @ Jg,     G    ]
    ])

    r_estac, r_compl = residuos(x, z, mu)
    rhs = np.concatenate([-r_estac, -r_compl])

    return M, rhs

def fraction_to_boundary(x, z, dx, dz, tau=0.995):
    """
    Calcula o maior α ∈ (0,1] tal que:
      g(x + α dx) < 0  →  controla via x diretamente (g linear)
      z + α dz > 0
    Usa a regra: α ≤ τ * min(-v/dv) para dv < 0.
    """
    alpha = 1.0
    for i in range(len(z)):
        if dz[i] < 0:
            alpha = min(alpha, -tau * z[i] / dz[i])
    gv  = g(x)
    Jg  = grad_g(x)
    dg  = Jg @ dx      

    for i in range(len(gv)):
        if dg[i] > 0: 
            alpha = min(alpha, -tau * gv[i] / dg[i])

    return min(alpha, 1.0)

def solver_mpi():
    x  = np.array([2.0, 2.0])
    z  = np.array([1.0])
    mu = 1.0

    mu_fator = 0.2          # µ_{k+1} = 0.2 · µ_k
    tol      = 1e-6         # tolerância KKT pura
    max_iter_ext = 50       # iterações externas (reduções de µ)
    max_iter_int = 20       # iterações Newton internas por µ

    cabecalho = (f"{'k':>4}  {'x1':>10}  {'x2':>10}  {'z':>10}"
                 f"  {'mu':>10}  {'Erro KKT':>12}")
    separador = "-" * len(cabecalho)

    print("\n" + "="*len(cabecalho))
    print("  SOLVER — Método de Pontos Interiores Primal-Dual")
    print("="*len(cabecalho))
    print(cabecalho)
    print(separador)

    k = 0
    ekkt = erro_kkt(x, z)

    print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
          f"  {mu:>10.6f}  {ekkt:>12.2e}")

    while ekkt > tol and k < max_iter_ext:

        for _ in range(max_iter_int):
            M, rhs = sistema_newton(x, z, mu)

            try:
                delta = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                print("  [AVISO] Sistema singular — interrompendo iteração interna.")
                break

            dx = delta[:len(x)]
            dz = delta[len(x):]

            alpha = fraction_to_boundary(x, z, dx, dz)

            x = x + alpha * dx
            z = z + alpha * dz

            r_e, r_c = residuos(x, z, mu)
            res_interno = np.linalg.norm(np.concatenate([r_e, r_c]))
            if res_interno < 1e-8:
                break

        mu = mu_fator * mu
        k += 1
        ekkt = erro_kkt(x, z)

        print(f"{k:>4}  {x[0]:>10.6f}  {x[1]:>10.6f}  {z[0]:>10.6f}"
              f"  {mu:>10.6f}  {ekkt:>12.2e}")

    print(separador)

    print(f"\n{'RESULTADO FINAL':^{len(separador)}}")
    print(separador)
    print(f"  x*  = [{x[0]:.8f},  {x[1]:.8f}]")
    print(f"  z*  = [{z[0]:.8f}]")
    print(f"  f(x*) = {f(x):.8f}")
    print(f"  g(x*) = {g(x)[0]:.8f}  (deve ser ≤ 0)")
    print(f"  Erro KKT final = {ekkt:.2e}")
    status = "✓ CONVERGIU" if ekkt <= tol else "✗ NÃO CONVERGIU"
    print(f"  Status: {status}  (tol = {tol:.0e})")
    print("="*len(separador) + "\n")

    return x, z

# Verificação analítica da solução KKT

def verificar_kkt(x, z):
    print("VERIFICAÇÃO DAS CONDIÇÕES KKT NA SOLUÇÃO")
    print("-" * 45)

    gv = g(x)
    Jg = grad_g(x)
    gf = grad_f(x)

    estac  = gf + Jg.T @ z
    compl  = gv * z
    viab   = gv

    print(f"  Estacionaridade  ∇f + z∇g  = {estac}")
    print(f"  Complementaridade  g·z      = {compl[0]:.2e}")
    print(f"  Viabilidade primal g(x*)    = {viab[0]:.6f}")
    print(f"  Dual z* > 0?                = {z[0] > 0}")
    print()

if __name__ == "__main__":
    x_opt, z_opt = solver_mpi()
    verificar_kkt(x_opt, z_opt)