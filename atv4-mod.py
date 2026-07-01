import numpy as np


# ==============================================================
# FUNÇÃO AUXILIAR — reduz duplicação dos cálculos de norma infinita
# ==============================================================

def max_abs_norm(vector):
    vector = np.asarray(vector, dtype=float)
    return float(np.max(np.abs(vector))) if vector.size > 0 else 0.0


# ==============================================================
# ATIVIDADE 1 — ITEM a)
# ==============================================================

def solve_item_1a(g1_value, mu1_value):
    g1_value = float(g1_value)
    mu1_value = float(mu1_value)
    epsilon = 1e-8

    if g1_value < -epsilon:
        mu1_adjusted = 0.0
        status = "Inativa (Forçado mu1 = 0.0)"
    else:
        mu1_adjusted = mu1_value
        status = "Ativa ou na fronteira de tolerância"

    return mu1_adjusted, status


# ==============================================================
# ATIVIDADE 1 — ITEM b)
# ==============================================================

def check_robust_complementarity(g_vector, mu_vector, epsilon=1e-8):
    g_vector = np.atleast_1d(np.array(g_vector, dtype=float))
    mu_vector = np.atleast_1d(np.array(mu_vector, dtype=float))
    kkt_max_residual = max_abs_norm(mu_vector * g_vector)
    condition_satisfied = kkt_max_residual <= epsilon
    return kkt_max_residual, condition_satisfied


# ==============================================================
# ATIVIDADE 2 — ITEM a)
# ==============================================================

def compute_jacobian_item_2a(x_point):
    x1 = float(x_point[0])
    x2 = float(x_point[1])
    jacobian_matrix = np.array([
        [1.0, 1.0],
        [2.0 * x1, 2.0 * x2]
    ], dtype=float)
    return jacobian_matrix


# ==============================================================
# ATIVIDADE 2 — ITEM b)
# ==============================================================

def evaluate_licq_failure_item_2b(jacobian_matrix):
    rank = np.linalg.matrix_rank(jacobian_matrix)
    jjt_matrix = np.dot(jacobian_matrix, jacobian_matrix.T)
    det_jjt = np.linalg.det(jjt_matrix)

    print(f"(Atividade 2b) Posto da Matriz J: {rank} (Esperado para LICQ: 2)")
    print(f"(Atividade 2b) Determinante de (J J^T): {det_jjt:.4f}")

    try:
        jacobian_inverse = np.linalg.inv(jjt_matrix)
        status = "Sucesso na inversão (Inesperado)"
    except np.linalg.LinAlgError as e:
        jacobian_inverse = None
        status = f"FALHA CATASTRÓFICA RECONHECIDA: {e}"

    return jjt_matrix, jacobian_inverse, status


# ==============================================================
# ATIVIDADE 3 — Formulação do Critério de Parada (KKT Error)
# ==============================================================

def compute_consolidated_kkt_error(grad_f, jacobian_h=None, h_value=None, jacobian_g=None, g_value=None,
                                    lambda_value=None, mu_value=None):
    grad_f = np.array(grad_f, dtype=float).flatten()
    h_vec = np.array(h_value, dtype=float).flatten() if h_value is not None else np.array([])
    g_vec = np.array(g_value, dtype=float).flatten() if g_value is not None else np.array([])
    lambda_vec = np.array(lambda_value, dtype=float).flatten() if lambda_value is not None else np.array([])
    mu_vec = np.array(mu_value, dtype=float).flatten() if mu_value is not None else np.array([])

    r_stationarity = np.copy(grad_f)
    if h_vec.size > 0 and jacobian_h is not None:
        r_stationarity += np.dot(np.array(jacobian_h, dtype=float).T, lambda_vec)
    if g_vec.size > 0 and jacobian_g is not None:
        r_stationarity += np.dot(np.array(jacobian_g, dtype=float).T, mu_vec)

    r_equality = np.copy(h_vec)
    r_inequality = np.maximum(0.0, g_vec) if g_vec.size > 0 else np.array([])
    r_complementarity = mu_vec * g_vec if g_vec.size > 0 else np.array([])

    norm_stationarity = max_abs_norm(r_stationarity)
    norm_equality = max_abs_norm(r_equality)
    norm_inequality = max_abs_norm(r_inequality)
    norm_complementarity = max_abs_norm(r_complementarity)

    kkt_error_max = max(norm_stationarity, norm_equality, norm_inequality, norm_complementarity)

    return {
        "KKT_Error_Consolidated": kkt_error_max,
        "Stationarity_Residual": norm_stationarity,
        "Equality_Residual": norm_equality,
        "Inequality_Violated_Residual": norm_inequality,
        "Complementarity_Residual": norm_complementarity
    }


# ==============================================================
# ATIVIDADE 4 — Arquitetura de um Resolvedor (Validador KKT)
# ==============================================================

def kkt_validator_architecture(grad_f, jacobian_h=None, h_value=None, jacobian_g=None, g_value=None,
                                lambda_value=None, mu_value=None, epsilon=1e-8):
    residuals = compute_consolidated_kkt_error(
        grad_f, jacobian_h, h_value, jacobian_g, g_value, lambda_value, mu_value
    )
    mu_vec = np.array(mu_value, dtype=float).flatten() if mu_value is not None else np.array([])

    primal_error = max(residuals["Equality_Residual"], residuals["Inequality_Violated_Residual"])
    primal_feasibility_ok = primal_error <= epsilon

    dual_feasibility_ok = not (mu_vec.size > 0 and np.any(mu_vec < -epsilon))

    global_error = residuals["KKT_Error_Consolidated"]

    if global_error <= epsilon and dual_feasibility_ok:
        classification = "PONTO CRÍTICO KKT VÁLIDO (Convergência Atingida com Sucesso)"
        status_code = 0
    elif primal_feasibility_ok:
        classification = ("PONTO VIÁVEL MAS NÃO-ÓTIMO "
                           "(Falta convergência nas condições de Estacionaridade/Dualidade)")
        status_code = 1
    else:
        classification = "PONTO INVIÁVEL (Viola as restrições físicas do problema)"
        status_code = -1

    return {
        "status_code": status_code,
        "classification": classification,
        "kkt_error_max": global_error,
        "residual_details": residuals,
        "primal_feasibility_ok": primal_feasibility_ok,
        "dual_feasibility_ok": dual_feasibility_ok
    }


# ==============================================================
# ATIVIDADE 5 — ITEM a)
# ==============================================================

def activity_5_item_a():
    x_k = np.array([2.5, 2.5], dtype=float)
    mu_k = np.array([5.0], dtype=float)
    return x_k, mu_k


# ==============================================================
# ATIVIDADE 5 — ITEM b)
# ==============================================================

def activity_5_item_b(x_k, mu_k):
    x1, x2 = x_k[0], x_k[1]

    grad_f = np.array([2.0 * x1, 2.0 * x2], dtype=float)
    g_value = np.array([5.0 - x1 - x2], dtype=float)
    jacobian_g = np.array([[-1.0, -1.0]], dtype=float)

    r_stationarity = grad_f + np.dot(jacobian_g.T, mu_k)
    r_primal = np.maximum(0.0, g_value)
    r_dual = np.maximum(0.0, -mu_k)
    r_complementarity = mu_k * g_value

    return r_stationarity, r_primal, r_dual, r_complementarity


# ==============================================================
# ATIVIDADE 5 — ITEM c)
# ==============================================================

def activity_5_item_c(r_stationarity, r_primal, r_dual, r_complementarity):
    norm_stationarity = max_abs_norm(r_stationarity)
    norm_primal = max_abs_norm(r_primal)
    norm_dual = max_abs_norm(r_dual)
    norm_complementarity = max_abs_norm(r_complementarity)
    kkt_error_max = max(norm_stationarity, norm_primal, norm_dual, norm_complementarity)

    print("======= DIAGNÓSTICO DETALHADO DO SOLVER KKT (Atividade 5 - Item c) =======")
    print(f"ERRO MÁXIMO GLOBAL (E_KKT): {kkt_error_max:.6e}")
    print("-" * 51)
    print(f"-> Índice de Estacionaridade:      {norm_stationarity:.6e}")
    print(f"-> Índice de Viabilidade Primal:   {norm_primal:.6e}")
    print(f"-> Índice de Viabilidade Dual:     {norm_dual:.6e}")
    print(f"-> Índice de Complementaridade:    {norm_complementarity:.6e}")
    print("===========================================================================")

    return kkt_error_max


# ==============================================================
# ATIVIDADE 5 — ITEM d)
# ==============================================================

def activity_5_item_d(kkt_error, g_value, mu_value, epsilon=1e-8):
    print("\n=== LEITURA GEOMÉTRICA E FÍSICA DO VELEIRO (Atividade 5 - Item d) ===")

    g_scalar = float(np.array(g_value).flatten()[0])
    mu_scalar = float(np.array(mu_value).flatten()[0])
    error_scalar = float(kkt_error)

    if error_scalar <= epsilon:
        if np.abs(g_scalar) <= epsilon and mu_scalar > epsilon:
            print("CONCLUSÃO: O barco PERMANECE ENCALHADO!")
            print(f"Motivo: A restrição segue ativa (g = {g_scalar}) e o multiplicador")
            print(f"dual positivo (mu = {mu_scalar}) comprova que a borda de areia anula")
            print("o gradiente descendente. Acionar o motor quebraria a viabilidade primal.")
        else:
            print("CONCLUSÃO: O barco está solto! O motor pode ser acionado.")
    else:
        print("CONCLUSÃO: O sistema ainda não convergiu para um ponto estacionário.")

    print("========================================================================\n")


# ==============================================================
# ATIVIDADE 6 — ITEM a)
# ==============================================================

def derive_rhs_blocks(grad_f, jacobian_h, h_value, lambda_k):
    grad_f_vec = np.array(grad_f, dtype=float).flatten()
    jacobian_matrix = np.array(jacobian_h, dtype=float)
    h_vec = np.array(h_value, dtype=float).flatten()
    lambda_vec = np.array(lambda_k, dtype=float).flatten()

    lagrangian_gradient = grad_f_vec + np.dot(jacobian_matrix.T, lambda_vec)
    rhs_vector = np.concatenate([-lagrangian_gradient, -h_vec])
    return rhs_vector


# ==============================================================
# ATIVIDADE 6 — ITEM b)
# ==============================================================

def run_newton_kkt_loop(problem_function, x_initial, lambda_initial, epsilon=1e-8, max_iter=100):
    x_k = np.array(x_initial, dtype=float).flatten()
    lambda_k = np.array(lambda_initial, dtype=float).flatten()
    error_history = []

    print("=== INICIANDO LAÇO ITERATIVO (Atividade 6 - Item b) ===")

    k = 0
    while k < max_iter:
        grad_f, hessian_l, jacobian_h, h_value = problem_function(x_k, lambda_k)
        lagrangian_gradient = grad_f + np.dot(jacobian_h.T, lambda_k)
        kkt_error = max(max_abs_norm(lagrangian_gradient), max_abs_norm(h_value))
        error_history.append(kkt_error)

        print(f"(Atividade 6b) Iteração {k:02d} | Erro KKT Global: {kkt_error:.6e}")

        if kkt_error <= epsilon:
            print(f"-> CONVERGÊNCIA OBTIDA NA ITERAÇÃO {k}!\n")
            break

        zero_block = np.zeros((jacobian_h.shape[0], jacobian_h.shape[0]))
        kkt_matrix = np.block([[hessian_l, jacobian_h.T], [jacobian_h, zero_block]])
        rhs_vector = np.concatenate([-lagrangian_gradient, -h_value])
        step = np.linalg.solve(kkt_matrix, rhs_vector)

        dx = step[:len(x_k)]
        d_lambda = step[len(x_k):]
        x_k += dx
        lambda_k += d_lambda
        k += 1

    return x_k, lambda_k, error_history


# ==============================================================
# ATIVIDADE 6 — ITEM c)
# ==============================================================

def simulate_quadratic_problem(x_vector, lambda_vector):
    x1, x2 = x_vector[0], x_vector[1]
    grad_f = np.array([2.0 * x1, 2.0 * x2])
    hessian_l = np.array([[2.0, 0.0], [0.0, 2.0]])
    jacobian_h = np.array([[1.0, 1.0]])
    h_value = np.array([x1 + x2 - 2.0])
    return grad_f, hessian_l, jacobian_h, h_value


# ==============================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # ATIVIDADE 1 — ITEM a)
    # ----------------------------------------------------------
    g1_x = -5.4
    mu1_current = 2.75
    mu1_final, situation = solve_item_1a(g1_x, mu1_current)
    print("(Atividade 1a)")
    print(f"Grandeza de g1(x): {g1_x}")
    print(f"Multiplicador prévio em memória: {mu1_current}")
    print(f"Condição da restrição: {situation}")
    print(f"Valor definitivo de mu1 gravado em memória: {mu1_final}")

    # ----------------------------------------------------------
    # ATIVIDADE 1 — ITEM b)
    # ----------------------------------------------------------
    g_test = np.array([-1.25e-16, -2.40])
    mu_test = np.array([3.50, 0.0])
    epsilon_target = 1e-8
    residual, complement_status = check_robust_complementarity(g_test, mu_test, epsilon=epsilon_target)
    print("(Atividade 1b)")
    print(f"Vetor g(x) com ruído: {g_test}")
    print(f"Vetor mu: {mu_test}")
    print(f"Índice de complementaridade apurado: {residual:.4e}")
    print(f"O índice fica abaixo da tolerância {epsilon_target}? {complement_status}")

    # ----------------------------------------------------------
    # ATIVIDADE 2 — ITEM a)
    # ----------------------------------------------------------
    test_point = np.array([1.0, 1.0])
    evaluated_jacobian = compute_jacobian_item_2a(test_point)
    print("=== SOLUÇÃO DA ATIVIDADE 2 - ITEM A ===")
    print(f"Ponto de teste x_k: {test_point}")
    print("Matriz Jacobiana J(1,1) obtida:")
    print(evaluated_jacobian)

    # ----------------------------------------------------------
    # ATIVIDADE 2 — ITEM b)
    # ----------------------------------------------------------
    evaluated_jacobian_matrix = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    print("=== SOLUÇÃO DA ATIVIDADE 2 - ITEM B ===")
    block_matrix, inverse_matrix, result_status = evaluate_licq_failure_item_2b(evaluated_jacobian_matrix)
    print(f"Resultado da tentativa de inversão: {result_status}")

    # ----------------------------------------------------------
    # ATIVIDADE 3 — Formulação do Critério de Parada (KKT Error)
    # ----------------------------------------------------------
    grad_f_vec = np.array([0.01, -0.02])
    jacobian_g_vec = np.array([[2.0, 1.0]])
    g_value_vec = np.array([-1.5e-9])
    mu_value_vec = np.array([0.012])
    result = compute_consolidated_kkt_error(
        grad_f=grad_f_vec,
        jacobian_g=jacobian_g_vec,
        g_value=g_value_vec,
        mu_value=mu_value_vec
    )
    print("======= CONFERÊNCIA DO ERRO KKT (Atividade 3) =======")
    print(f"-> ERRO MÁXIMO GLOBAL: {result['KKT_Error_Consolidated']:.6e}")
    print(f"   Índice Estacionaridade:  {result['Stationarity_Residual']:.6e}")
    print(f"   Índice Igualdade:        {result['Equality_Residual']:.6e}")
    print(f"   Índice Desigualdade:     {result['Inequality_Violated_Residual']:.6e}")
    print(f"   Índice Complementaridade:{result['Complementarity_Residual']:.6e}")
    print("======================================================")

    # ----------------------------------------------------------
    # ATIVIDADE 4 — Arquitetura de um Resolvedor
    # ----------------------------------------------------------
    print("=== CASO TESTE 1 (Atividade 4): Simulando um Ponto Ótimo Legítimo ===")
    grad_f_ok = [0.0, 0.0]
    result_1 = kkt_validator_architecture(grad_f=grad_f_ok)
    print(f"Resultado: {result_1['classification']}")
    print(f"Erro Máximo: {result_1['kkt_error_max']:.2e}\n")

    print("=== CASO TESTE 2 (Atividade 4): Simulando um Ponto Inviável (Restrição Estourada) ===")
    grad_f_invalid = [0.0, 0.0]
    g_violated = [0.55]
    result_2 = kkt_validator_architecture(grad_f=grad_f_invalid, g_value=g_violated)
    print(f"Resultado: {result_2['classification']}")
    print(f"Índice de Desigualdade Violada: {result_2['residual_details']['Inequality_Violated_Residual']:.4f}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM a)
    # ----------------------------------------------------------
    x, mu = activity_5_item_a()
    print("=== ATIVIDADE 5 — ITEM A ===")
    print(f"Ponto Primal x_k reservado: {x}")
    print(f"Multiplicador Dual mu_k reservado: {mu}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM b)
    # ----------------------------------------------------------
    x_5b = np.array([2.5, 2.5])
    mu_5b = np.array([5.0])
    r_stationarity, r_primal, r_dual, r_complementarity = activity_5_item_b(x_5b, mu_5b)
    print("=== ATIVIDADE 5 — ITEM B ===")
    print(f"Índice de Estacionaridade (Vetor): {r_stationarity}")
    print(f"Índice de Viabilidade Primal:      {r_primal}")
    print(f"Índice de Viabilidade Dual:        {r_dual}")
    print(f"Índice de Complementaridade:       {r_complementarity}")

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM c)
    # ----------------------------------------------------------
    r_stationarity_test = np.array([0.0, 0.0])
    r_primal_test = np.array([0.0])
    r_dual_test = np.array([0.0])
    r_complementarity_test = np.array([0.0])
    activity_5_item_c(r_stationarity_test, r_primal_test, r_dual_test, r_complementarity_test)

    # ----------------------------------------------------------
    # ATIVIDADE 5 — ITEM d)
    # ----------------------------------------------------------
    error_from_item_c = 0.0
    g_from_item_b = 0.0
    mu_from_item_a = 5.0
    activity_5_item_d(kkt_error=error_from_item_c, g_value=g_from_item_b, mu_value=mu_from_item_a)

    # ----------------------------------------------------------
    # ATIVIDADE 6 — ITEM a)
    # ----------------------------------------------------------
    print("=== ATIVIDADE 6 - ITEM A ===")
    rhs = derive_rhs_blocks(
        grad_f=[1.0, 2.0], jacobian_h=[[1.0, 1.0]], h_value=[0.1], lambda_k=[0.5]
    )
    print(f"Vetor Lado Direito (RHS) linearizado: {rhs}")

    # ----------------------------------------------------------
    # ATIVIDADE 6 — ITEM c)
    # ----------------------------------------------------------
    x_initial = np.array([10.0, 5.0])
    lambda_initial = np.array([0.0])
    x_optimal, lambda_optimal, error_history = run_newton_kkt_loop(
        simulate_quadratic_problem, x_initial, lambda_initial
    )
    print("======= RESULTADOS DO DIAGNÓSTICO FINAL (Atividade 6 - Item c) =======")
    print(f"Ponto Ótimo Primal Encontrado x*: {x_optimal} (Esperado analiticamente: [1. 1.])")
    print(f"Multiplicador Dual Encontrado lambda*: {lambda_optimal}")
    print(f"Total de Iterações executadas: {len(error_history) - 1}")
    print("========================================================================")