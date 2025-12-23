import numpy as np
from scipy.linalg import block_diag


def generate_coef(p, K, noise = 0.01):
	beta_target = np.zeros(p + 1)

	Beta_S = np.zeros((p + 1, K))

	for m in range(K):
		Beta_S[1:, m] += np.random.normal(loc = 0, scale = noise, size = p)

	return Beta_S, beta_target


def generate_data(p, nS, nT, K):

	Beta_S, beta_target = generate_coef(p, K)
	Beta_all = np.column_stack([Beta_S[:, k] for k in range(K)] + [beta_target])

	XS_list, YS_list = [], []
	N_vec = [nS] * K + [nT]

	for k in range(K + 1):
		Xk_raw = np.random.normal(0, 1, size = (N_vec[k], p))
		intercept_col = np.ones((N_vec[k], 1))
		Xk = np.hstack([intercept_col, Xk_raw])

		true_Yk = Xk @ Beta_all[:, k]

		noise = np.random.normal(0, 1, size = N_vec[k])

		Yk = true_Yk + noise

		XS_list.append(Xk)
		YS_list.append(Yk)

	X0 = XS_list[-1]
	Y0 = YS_list[-1]
	XS_list = XS_list[:-1]
	YS_list = YS_list[:-1]

	SigmaS_list = [np.identity(nS) for _ in range(K)]
	Sigma0 = np.identity(nT)

	X = np.vstack(XS_list + [X0])
	Y = np.concatenate(YS_list + [Y0])
	Sigma = block_diag(*SigmaS_list, Sigma0)

	return X, X0, Y, Y0, Sigma


def Construct_matrix(X, Y, gamma):
	N, p = X.shape

	P = np.zeros((2 * p + 2 * N, 2 * p + 2 * N))
	P[2 * p : 2 * p + N, 2 * p : 2 * p + N] = np.identity(N)

	q_0 = np.zeros(2 * p + 2 * N)
	q_0[2 * p + N : 2 * p + 2 * N] = (gamma * np.ones(N))

	K = np.zeros((2 * p + 2 * N, 2 * p + 2 * N))
	K[: p, : p] = np.identity(p)
	K[: p, p : 2 * p] = -np.identity(p)
	K[p : 2 * p, : p] = -np.identity(p)
	K[p : 2 * p, p : 2 * p] = np.identity(p)

	q_1 = np.zeros(2 * p + 2 * N)
	q_1[: p] = np.ones(p)
	q_1[p : 2 * p] = np.ones(p)

	S = np.zeros((5 * N + 2 * p, 2 * p + 2 * N))
	S[: N, : p] = X
	S[: N, p : 2 * p] = -X
	S[: N, 2 * p : 2 * p + N] = - np.identity(N)
	S[: N, 2 * p + N : 2 * p + 2 * N] = - np.identity(N)

	S[N : 2 * N, : p] = -X
	S[N : 2 * N, p : 2 * p] = X
	S[N : 2 * N, 2 * p : 2 * p + N] = - np.identity(N)
	S[N : 2 * N, 2 * p + N : 2 * p + 2 * N] = - np.identity(N)

	S[2 * N : 3 * N, 2 * p : 2 * p + N] = -np.identity(N)
	S[3 * N : 4 * N, 2 * p : 2 * p + N] = np.identity(N)
	S[4 * N : 5 * N, 2 * p + N : 2 * p + 2 * N] = -np.identity(N)
	S[5 * N : 5 * N + p, : p] = -np.identity(p)
	S[5 * N + p : 5 * N + 2 * p, p : 2 * p] = -np.identity(p)

	h_w = np.zeros(5 * N + 2 * p)
	h_w[: N] = Y
	h_w[N : 2 * N] = -Y
	h_w[3 * N : 4 * N] = gamma * np.ones(N)

	u_0 = np.zeros(5 * N + 2 * p)
	u_0[3 * N : 4 * N] = gamma * np.ones(N)

	u_1 = np.zeros(5 * N + 2 * p)

	return P, K, q_0, q_1, S, h_w, u_0, u_1