import numpy as np

# ================================================================
#  Método da Região de Confiança — Ponto de Cauchy
#  Função: Rosenbrock  f(x, y) = 100·(y − x²)² + (1 − x)²
#  Ponto inicial: x₀ = (−1.2,  0.5)
#  Solução esperada: x* = (1.0,  1.0),  f(x*) = 0
#
#  Gradiente e Hessiana calculados analiticamente (estático).
# ================================================================


def rosenbrock(v):
    """Avalia f(x, y) no vetor v = [x, y]."""
    return 100.0 * (v[1] - v[0]**2)**2 + (1.0 - v[0])**2


# ── Derivadas analíticas da Rosenbrock ─────────────────────────
#
#  ∂f/∂x  = -400·x·(y - x²) - 2·(1 - x)
#  ∂f/∂y  =  200·(y - x²)
#
#  ∂²f/∂x²    = -400·(y - x²) + 800·x² + 2
#  ∂²f/∂x∂y   = -400·x
#  ∂²f/∂y²    =  200
# ────────────────────────────────────────────────────────────────


# ── Parâmetros do algoritmo ─────────────────────────────────────
x         = np.array([-1.2, 0.5])   # ponto inicial
delta     = 1.0                      # raio inicial da região de confiança
DELTA_MAX = 5.0                      # raio máximo permitido
TOL_GRAD  = 1e-6                     # critério de parada  ‖gk‖ < tol
EPS       = 1e-12                    # salvaguarda: denominador de ρk
MAX_ITER  = 2000

# ── Cabeçalho da tabela de iterações ───────────────────────────
cabecalho = (f"{'k':>5} | {'f(x)':>14} | {'||g||':>11} | "
             f"{'rho':>9} | {'Delta':>8} | Decisão")
print(f"\n{cabecalho}")
print("─" * 68)

converged = False

for k in range(MAX_ITER):

    x0, x1 = x[0], x[1]        # coordenadas do ponto atual
    fk = rosenbrock(x)

    # ── Gradiente analítico (estático) ─────────────────────────
    gk = np.array([
        -400.0 * x0 * (x1 - x0**2) - 2.0 * (1.0 - x0),
         200.0 * (x1 - x0**2)
    ])
    norm_g = np.linalg.norm(gk)

    # ── Critério de parada ─────────────────────────────────────
    if norm_g < TOL_GRAD:
        print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
              f"{'---':>9} | {delta:>8.5f} | CONVERGIU ✓")
        converged = True
        break

    # ── Hessiana analítica (estática) ──────────────────────────
    Bk = np.array([
        [-400.0*(x1 - x0**2) + 800.0*x0**2 + 2.0,  -400.0*x0],
        [-400.0*x0,                                   200.0    ]
    ])

    # ── Ponto de Cauchy ────────────────────────────────────────
    #   gᵀBg decide qual ramo utilizar (Seção 4.2 / Passo 2)
    gBg = gk @ Bk @ gk

    if gBg <= 0.0:
        # Caso 1 — curvatura não-positiva:
        #   modelo desce indefinidamente na direção −g;
        #   o melhor passo toca exatamente a fronteira do raio.
        sk = -(delta / norm_g) * gk

    else:
        # Caso 2 — curvatura positiva:
        #   existe mínimo do modelo na direção −g.
        #   ᾱ = ‖gk‖² / (gᵀBg)  →  passo ideal irrestrito
        alpha_bar = norm_g**2 / gBg
        #   trunca no raio se o mínimo estiver fora da cerca
        alpha_k = min(alpha_bar, delta / norm_g)
        sk = -alpha_k * gk

    # ── Razão de fidelidade ρk  (Seção 7.1 / Passo 3) ─────────
    #   Redução Prevista:  Δm = −(gᵀs + ½ sᵀBs)
    pred = -(gk @ sk + 0.5 * sk @ Bk @ sk)

    # Salvaguarda numérica: denominador ≈ 0 → ótimo atingido
    if abs(pred) < EPS:
        x = x + sk
        print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
              f"{'denom≈0':>9} | {delta:>8.5f} | CONVERGIU ✓")
        converged = True
        break

    #   Redução Real:  Δf = f(xk) − f(xk + sk)
    ared = fk - rosenbrock(x + sk)
    rho  = ared / pred

    # ── Atualização de posição e raio (Tabela — Seção 7.1) ─────
    if rho < 0.25:
        # Modelo errou: rejeita passo e encolhe a região
        delta   = 0.25 * delta
        decisao = "REJEITA ▼"

    elif rho < 0.75:
        # Modelo razoável: aceita passo, mantém raio
        x       = x + sk
        decisao = "aceita  ─"

    else:
        # Modelo excelente: aceita passo e expande a região
        x       = x + sk
        delta   = min(2.0 * delta, DELTA_MAX)
        decisao = "aceita  ▲"

    print(f"{k:>5} | {fk:>14.8f} | {norm_g:>11.4e} | "
          f"{rho:>9.5f} | {delta:>8.5f} | {decisao}")

if not converged:
    print(f"\n! Limite de {MAX_ITER} iterações atingido sem convergência.")

# ── Resultado final ─────────────────────────────────────────────
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