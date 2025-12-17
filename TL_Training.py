import cvxpy as cp


def training_qp_w(P, K, q_0, q_1, S, h_w, Z, lambda_w, alpha):
	N, p = Z.shape

	Q = (1 / N) * P + lambda_w * (1 - alpha) * K
	f = (1 / N) * q_0 + lambda_w * alpha * q_1

	t = cp.Variable(2 * p + 2 * N)

	constraints = [S @ t <= h_w]

	objective = cp.Minimize(0.5 * cp.quad_form(t, Q) + f.T @ t)
	problem = cp.Problem(objective, constraints)

	try:
		problem.solve(solver = cp.OSQP, eps_abs = 1e-10, eps_rel = 1e-10, max_iter = 100000)

		w_plus = t.value[ : p]
		w_minus = t.value[p : 2 * p]
		u = t.value[2 * p : 2 * p + N]
		v = t.value[2 * p + N : 2 * p + 2 * N]

		info = {
			'w_plus' : w_plus,
			'w_minus' : w_minus,
			'u' : u,
			'v' : v,
			't_value' : t.value,
			'Q' : Q,
			'f' : f,
			'Problem' : problem
		}

		return info

	except cp.error.SolverError as e:
		return None


def training_qp_delta(P, K, q_0, q_1, S, h_d, Z0, lambda_d, alpha):
	nT, p = Z0.shape

	Q = (1 / nT) * P + lambda_d * (1 - alpha) * K
	f = (1 / nT) * q_0 + lambda_d * alpha * q_1

	t = cp.Variable(2 * p + 2 * nT)

	constraints = [S @ t <= h_d]

	objective = cp.Minimize(0.5 * cp.quad_form(t, Q) + f.T @ t)
	problem = cp.Problem(objective, constraints)

	try:
		problem.solve(solver = cp.OSQP, eps_abs = 1e-10, eps_rel = 1e-10, max_iter = 100000)

		d_plus = t.value[ : p]
		d_minus = t.value[p : 2 * p]
		u = t.value[2 * p : 2 * p + nT]
		v = t.value[2 * p + nT : 2 * p + 2 * nT]

		info = {
			'd_plus' : d_plus,
			'd_minus' : d_minus,
			'u' : u,
			'v' : v,
			't_value' : t.value,
			'Q' : Q,
			'f' : f,
			'Problem' : problem
		}
		return info

	except cp.error.SolverError as e:
		return None


def check_KKT(Q, f, S, h, t, v):
	KKT = True

	Stationary = Q @ t + f + S.T @ v
	for s in Stationary:
		if abs(s) >= 1e-5:
			KKT = False
			return KKT

	Primal_Feas = S @ t - h
	for p in Primal_Feas:
		if p >= 1e-5:
			KKT = False
			return KKT

	Dual_Feas = v
	for d in Dual_Feas:
		if d < -1e-5:
			KKT = False
			return KKT

	Comp_Slack = Dual_Feas * Primal_Feas
	for c in Comp_Slack:
		if abs(c) >= 1e-5:
			KKT = False
			return KKT

	return KKT