import numpy as np


# ==============================================================
# ATIVIDADE 1 — ITEM a)
# ==============================================================

def resolver_item_1a(g1_val, mu1_val):
    g1 = float(g1_val)
    mu1 = float(mu1_val)
    epsilon = 1e-8

    if g1 < -epsilon:
        mu1_corrigido = 0.0
        status = "Inativa (Forçado mu1 = 0.0)"
    else:
        mu1_corrigido = mu1
        status = "Ativa ou na fronteira de tolerância"

    return mu1_corrigido, status


# ==============================================================
# ATIVIDADE 1 — ITEM b)
# ==============================================================

def verificar_complementaridade_robusta(g_vetor, mu_vetor, epsilon=1e-8):
    g = np.atleast_1d(np.array(g_vetor, dtype=float))
    mu = np.atleast_1d(np.array(mu_vetor, dtype=float))
    residuos_individuais = np.abs(mu * g)
    residuo_max_kkt = np.max(residuos_individuais)
    condicao_satisfeita = residuo_max_kkt <= epsilon
    return residuo_max_kkt, condicao_satisfeita


# ==============================================================
# ATIVIDADE 2 — ITEM a)
# ==============================================================

def calcular_jacobiana_item_2a(x_ponto):
    x1 = float(x_ponto[0])
    x2 = float(x_ponto[1])
    J = np.array([
        [1.0, 1.0],
        [2.0 * x1, 2.0 * x2]
    ], dtype=float)
    return J


# ==============================================================
# ATIVIDADE 2 — ITEM b)
# ==============================================================

def avaliar_falha_licq_item_2b(J):
    posto = np.linalg.matrix_rank(J)
    JJ_T = np.dot(J, J.T)
    det_JJ_T = np.linalg.det(JJ_T)

    print(f"Posto da Matriz J: {posto} (Esperado para LICQ: 2)")
    print(f"Determinante de (J J^T): {det_JJ_T:.4f}")

    try:
        J_inversa = np.linalg.inv(JJ_T)
        status = "Sucesso na inversão (Inesperado)"
    except np.linalg.LinAlgError as e:
        J_inversa = None
        status = f"FALHA CATASTRÓFICA RECONHECIDA: {e}"

    return JJ_T, J_inversa, status


# ==============================================================
# ATIVIDADE 3 — Formulação do Critério de Parada (KKT Error)
# ==============================================================

def calcular_erro_kkt_consolidado(grad_f, J_h=None, h_val=None, J_g=None, g_val=None,
                                   lambda_val=None, mu_val=None):
    grad_f = np.array(grad_f, dtype=float).flatten()
    h = np.array(h_val, dtype=float).flatten() if h_val is not None else np.array([])
    g = np.array(g_val, dtype=float).flatten() if g_val is not None else np.array([])
    lam = np.array(lambda_val, dtype=float).flatten() if lambda_val is not None else np.array([])
    mu = np.array(mu_val, dtype=float).flatten() if mu_val is not None else np.array([])

    R_est = np.copy(grad_f)
    if h.size > 0 and J_h is not None:
        R_est += np.dot(np.array(J_h, dtype=float).T, lam)
    if g.size > 0 and J_g is not None:
        R_est += np.dot(np.array(J_g, dtype=float).T, mu)

    R_eq = np.copy(h)
    R_ineq = np.maximum(0.0, g) if g.size > 0 else np.array([])
    R_comp = mu * g if g.size > 0 else np.array([])

    norm_est = np.max(np.abs(R_est)) if R_est.size > 0 else 0.0
    norm_eq = np.max(np.abs(R_eq)) if R_eq.size > 0 else 0.0
    norm_ineq = np.max(np.abs(R_ineq)) if R_ineq.size > 0 else 0.0
    norm_comp = np.max(np.abs(R_comp)) if R_comp.size > 0 else 0.0

    erro_kkt_max = max(norm_est, norm_eq, norm_ineq, norm_comp)

    return {
        "Erro_KKT_Consolidado": erro_kkt_max,
        "Residuo_Estacionaridade": norm_est,
        "Residuo_Igualdade": norm_eq,
        "Residuo_Desigualdade_Violada": norm_ineq,
        "Residuo_Complementaridade": norm_comp
    }


# ==============================================================
# ATIVIDADE 4 — Arquitetura de um Resolvedor (Validador KKT)
# ==============================================================

def arquitetura_validador_kkt(grad_f, J_h=None, h_val=None, J_g=None, g_val=None,
                               lambda_val=None, mu_val=None, epsilon=1e-8):
    residuos = calcular_erro_kkt_consolidado(
        grad_f, J_h, h_val, J_g, g_val, lambda_val, mu_val
    )
    mu = np.array(mu_val, dtype=float).flatten() if mu_val is not None else np.array([])

    erro_primal = max(residuos["Residuo_Igualdade"], residuos["Residuo_Desigualdade_Violada"])
    viabilidade_primal_satisfeita = erro_primal <= epsilon

    viabilidade_dual_satisfeita = True
    if mu.size > 0 and np.any(mu < -epsilon):
        viabilidade_dual_satisfeita = False

    erro_global = residuos["Erro_KKT_Consolidado"]

    if erro_global <= epsilon and viabilidade_dual_satisfeita:
        classificacao = "PONTO CRÍTICO KKT VÁLIDO (Convergência Atingida com Sucesso)"
        codigo_status = 0
    elif viabilidade_primal_satisfeita:
        classificacao = ("PONTO VIÁVEL MAS NÃO-ÓTIMO "
                         "(Falta convergência nas condições de Estacionaridade/Dualidade)")
        codigo_status = 1
    else:
        classificacao = "PONTO INVIÁVEL (Viola as restrições físicas do problema)"
        codigo_status = -1

    return {
        "status_code": codigo_status,
        "classificacao": classificacao,
        "erro_kkt_max": erro_global,
        "detalhes_residuos": residuos,
        "viabilidade_primal_ok": viabilidade_primal_satisfeita,
        "viabilidade_dual_ok": viabilidade_dual_satisfeita
    }


# ==============================================================
# ATIVIDADE 5 — ITEM a)
# ==============================================================

def atividade_5_item_a():
    x_k = np.array([2.5, 2.5], dtype=float)
    mu_k = np.array([5.0], dtype=float)
    return x_k, mu_k


# ==============================================================
# ATIVIDADE 5 — ITEM b)
# ==============================================================

def atividade_5_item_b(x_k, mu_k):
    x1, x2 = x_k[0], x_k[1]

    grad_f = np.array([2.0 * x1, 2.0 * x2], dtype=float)
    g_val = np.array([5.0 - x1 - x2], dtype=float)
    J_g = np.array([[-1.0, -1.0]], dtype=float)

    R_est = grad_f + np.dot(J_g.T, mu_k)
    R_prim = np.maximum(0.0, g_val)
    R_dual = np.maximum(0.0, -mu_k)
    R_comp = mu_k * g_val

    return R_est, R_prim, R_dual, R_comp


# ==============================================================
# ATIVIDADE 5 — ITEM c)
# ==============================================================

def atividade_5_item_c(r_est, r_prim, r_dual, r_comp):
    norm_est = np.max(np.abs(r_est))
    norm_prim = np.max(np.abs(r_prim))
    norm_dual = np.max(np.abs(r_dual))
    norm_comp = np.max(np.abs(r_comp))
    erro_kkt_max = max(norm_est, norm_prim, norm_dual, norm_comp)

    print("======= DIAGNÓSTICO DETALHADO DO SOLVER KKT =======")
    print(f"ERRO MÁXIMO CONSOLIDADO (E_KKT): {erro_kkt_max:.6e}")
    print("-" * 51)
    print(f"-> Norma de Estacionaridade:     {norm_est:.6e}")
    print(f"-> Norma de Viabilidade Primal:  {norm_prim:.6e}")
    print(f"-> Norma de Viabilidade Dual:    {norm_dual:.6e}")
    print(f"-> Norma de Complementaridade:   {norm_comp:.6e}")
    print("===================================================")

    return erro_kkt_max


# ==============================================================
# ATIVIDADE 5 — ITEM d)
# ==============================================================

def atividade_5_item_d(erro_kkt, g_val, mu_val, epsilon=1e-8):
    print("\n=== INTERPRETAÇÃO GEOMÉTRICA E FÍSICA DO VELEIRO ===")

    g_escalar = float(np.array(g_val).flatten()[0])
    mu_escalar = float(np.array(mu_val).flatten()[0])
    erro_escalar = float(erro_kkt)

    if erro_escalar <= epsilon:
        if np.abs(g_escalar) <= epsilon and mu_escalar > epsilon:
            print("DIAGNÓSTICO CONCLUSIVO: O barco CONTINUA ENCALHADO!")
            print(f"Justificativa: A restrição está ativa (g = {g_escalar}) e o multiplicador")
            print(f"dual positivo (mu = {mu_escalar}) prova que a fronteira de areia anula")
            print("o gradiente descendente. Ligar o motor violaria a viabilidade primal.")
        else:
            print("DIAGNÓSTICO CONCLUSIVO: O barco está livre! O motor pode ser ligado.")
    else:
        print("DIAGNÓSTICO CONCLUSIVO: O sistema não convergiu para um ponto estacionário.")

    print("====================================================\n")


# ==============================================================
# ATIVIDADE 6 — ITEM a)
# ==============================================================

def deduzir_blocos_rhs(grad_f, J_h, h_val, lambda_k):
    g_f = np.array(grad_f, dtype=float).flatten()
    A = np.array(J_h, dtype=float)
    h = np.array(h_val, dtype=float).flatten()
    lam = np.array(lambda_k, dtype=float).flatten()

    grad_L = g_f + np.dot(A.T, lam)
    vetor_rhs = np.concatenate([-grad_L, -h])
    return vetor_rhs


# ==============================================================
# ATIVIDADE 6 — ITEM b)
# ==============================================================

def run_loop_newton_kkt(funcao_problema, x_inicial, lambda_inicial, epsilon=1e-8, max_iter=100):
    x_k = np.array(x_inicial, dtype=float).flatten()
    lam_k = np.array(lambda_inicial, dtype=float).flatten()
    hist_erro = []

    print("=== INICIALIZANDO LOOP ITERATIVO (ITEM B) ===")

    k = 0
    while k < max_iter:
        grad_f, Hess_L, J_h, h_val = funcao_problema(x_k, lam_k)
        grad_L = grad_f + np.dot(J_h.T, lam_k)
        erro_kkt = max(np.max(np.abs(grad_L)), np.max(np.abs(h_val)))
        hist_erro.append(erro_kkt)

        print(f"Iteracao {k:02d} | Erro KKT Consolidado: {erro_kkt:.6e}")

        if erro_kkt <= epsilon:
            print(f"-> CONVERGÊNCIA ATINGIDA NA ITERAÇÃO {k}!\n")
            break

        bloco_zeros = np.zeros((J_h.shape[0], J_h.shape[0]))
        matriz_kkt = np.block([[Hess_L, J_h.T], [J_h, bloco_zeros]])
        vetor_rhs = np.concatenate([-grad_L, -h_val])
        passo = np.linalg.solve(matriz_kkt, vetor_rhs)

        dx = passo[:len(x_k)]
        d_lam = passo[len(x_k):]
        x_k += dx
        lam_k += d_lam
        k += 1

    return x_k, lam_k, hist_erro


# ==============================================================
# ATIVIDADE 6 — ITEM c)
# ==============================================================

def simular_problema_quadratico(x_vetor, lam_vetor):
    x1, x2 = x_vetor[0], x_vetor[1]
    grad_f = np.array([2.0 * x1, 2.0 * x2])
    Hess_L = np.array([[2.0, 0.0], [0.0, 2.0]])
    J_h = np.array([[1.0, 1.0]])
    h_val = np.array([x1 + x2 - 2.0])
    return grad_f, Hess_L, J_h, h_val


# ==============================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # ATIVIDADE 1 — ITEM a)
    # ----------------------------------------------------------
    g1_x = -5.4
    mu1_corrente = 2.75
    mu1_final, situacao = resolver_item_1a(g1_x, mu1_corrente)
    print(f"Valor de g1(x): {g1_x}")
    print(f"Multiplicador original na memória: {mu1_corrente}")
    print(f"Estado da restrição: {situacao}")
    print(f"Valor exato de mu1 atribuído na memória: {mu1_final}")

    # ----------------------------------------------------------
    # ATIVIDADE 1 — ITEM b)
    # ----------------------------------------------------------
    g_teste = np.array([-1.25e-16, -2.40])
    mu_teste = np.array([3.50, 0.0])
    epsilon_alvo = 1e-8
    residuo, status = verificar_complementaridade_robusta(g_teste, mu_teste, epsilon=epsilon_alvo)
    print(f"Vetor g(x) com ruído: {g_teste}")
    print(f"Vetor mu: {mu_teste}")
    print(f"Resíduo de complementaridade calculado: {residuo:.4e}")
    print(f"O resíduo limpa a memória sob a tolerância {epsilon_alvo}? {status}")

    # ----------------------------------------------------------
    # ATIVIDADE 2 — ITEM a)
    # ----------------------------------------------------------
    ponto_teste = np.array([1.0, 1.0])
    jacobiana_avaliada = calcular_jacobiana_item_2a(ponto_teste)
    print("=== RESOLUÇÃO DA ATIVIDADE 2 - ITEM A ===")
    print(f"Ponto de teste x_k: {ponto_teste}")
    print("Matriz Jacobiana J(1,1) computada:")
    print(jacobiana_avaliada)

    # ----------------------------------------------------------
    # ATIVIDADE 2 — ITEM b)
    # ----------------------------------------------------------
    J_avaliada = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    print("=== RESOLUÇÃO DA ATIVIDADE 2 - ITEM B ===")
    matriz_bloco, inversa, resultado_status = avaliar_falha_licq_item_2b(J_avaliada)
    print(f"Resultado da tentativa de inversão: {resultado_status}")

    # ----------------------------------------------------------
    # ATIVIDADE 3 — Formulação do Critério de Parada (KKT Error)
    # ----------------------------------------------------------
    gradiente_f = np.array([0.01, -0.02])
    jacobiana_g = np.array([[2.0, 1.0]])
    valor_g = np.array([-1.5e-9])
    multiplicador_mu = np.array([0.012])
    resultado = calcular_erro_kkt_consolidado(
        grad_f=gradiente_f,
        J_g=jacobiana_g,
        g_val=valor_g,
        mu_val=multiplicador_mu
    )
    print("======= VERIFICAÇÃO DO ERRO KKT =======")
    print(f"-> ERRO MAXIMO CONSOLIDADO: {resultado['Erro_KKT_Consolidado']:.6e}")
    print(f"   Norma Estacionaridade:  {resultado['Residuo_Estacionaridade']:.6e}")
    print(f"   Norma Igualdade:         {resultado['Residuo_Igualdade']:.6e}")
    print(f"   Norma Desigualdade:      {resultado['Residuo_Desigualdade_Violada']:.6e}")
    print(f"   Norma Complementaridade: {resultado['Residuo_Complementaridade']:.6e}")
    print("===========================================================")

    # ----------------------------------------------------------
    # ATIVIDADE 4 — Arquitetura de um Resolvedor
    # ----------------------------------------------------------
    print("=== CASO TESTE 1: Simulando um Ponto Ótimo Legítimo ===")
    gf_ok = [0.0, 0.0]
    relat_1 = arquitetura_validador_kkt(grad_f=gf_ok)
    print(f"Resultado: {relat_1['classificacao']}")
    print(f"Erro Máximo: {relat_1['erro_kkt_max']:.2e}\n")

    print("=== CASO TESTE 2: Simulando um Ponto Inviável (Restrição Estourada) ===")
    gf_inv = [0.0, 0.0]
    g_violado = [0.55]
    relat_2 = arquitetura_validador_kkt(grad_f=gf_inv, g_val=g_violado)
    print(f"Resultado: {relat_2['classificacao']}")
    print(f"Resíduo de Desigualdade Violada: {relat_2['detalhes_residuos']['Residuo_Desigualdade_Violada']:.4f}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM a)
    # ----------------------------------------------------------
    x, mu = atividade_5_item_a()
    print("=== ATIVIDADE 5 — ITEM A ===")
    print(f"Ponto Primal x_k alocado: {x}")
    print(f"Multiplicador Dual mu_k alocado: {mu}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM b)
    # ----------------------------------------------------------
    x_5b = np.array([2.5, 2.5])
    mu_5b = np.array([5.0])
    r_est, r_prim, r_dual, r_comp = atividade_5_item_b(x_5b, mu_5b)
    print("=== ATIVIDADE 5 — ITEM B ===")
    print(f"Resíduo de Estacionaridade (Vetor): {r_est}")
    print(f"Resíduo de Viabilidade Primal:      {r_prim}")
    print(f"Resíduo de Viabilidade Dual:        {r_dual}")
    print(f"Resíduo de Complementaridade:       {r_comp}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM c)
    # ----------------------------------------------------------
    r_e = np.array([0.0, 0.0])
    r_p = np.array([0.0])
    r_d = np.array([0.0])
    r_c = np.array([0.0])
    atividade_5_item_c(r_e, r_p, r_d, r_c)

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM d)
    # ----------------------------------------------------------
    erro_do_item_c = 0.0
    g_do_item_b = 0.0
    mu_do_item_a = 5.0
    atividade_5_item_d(erro_kkt=erro_do_item_c, g_val=g_do_item_b, mu_val=mu_do_item_a)

    # ----------------------------------------------------------
    # ATIVIDADE 6 — ITEM a)
    # ----------------------------------------------------------
    print("=== ATIVIDADE 6 - ITEM A ===")
    rhs = deduzir_blocos_rhs(
        grad_f=[1.0, 2.0], J_h=[[1.0, 1.0]], h_val=[0.1], lambda_k=[0.5]
    )
    print(f"Vetor Lado Direito (RHS) linearizado: {rhs}")

    # ----------------------------------------------------------
    # ATIVIDADE 6 — ITEM c)
    # ----------------------------------------------------------
    x_init = np.array([10.0, 5.0])
    lam_init = np.array([0.0])
    x_opt, lam_opt, historico = run_loop_newton_kkt(
        simular_problema_quadratico, x_init, lam_init
    )
    print("======= RESULTADOS DO DIAGNÓSTICO FINAL =======")
    print(f"Ponto Ótimo Primal Encontrado x*: {x_opt} (Esperado analiticamente: [1. 1.])")
    print(f"Multiplicador Dual Encontrado lambda*: {lam_opt}")
    print(f"Total de Iteracoes executadas: {len(historico) - 1}")
    print("===============================================")