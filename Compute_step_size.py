import numpy as np


def compute_quotient(numerator, denominator):
	if denominator == 0:
		return np.inf

	quotient = numerator / denominator

	if quotient <= 0:
		return np.inf

	return quotient


def compute_interval(Q_z, f_z, S_Az, S_Azc, h_z, u0_z, u1_z, Z, t_z, v_Az, Az, Azc):
	N, p = Z.shape

	u0_A = u0_z[Az]
	u1_A = u1_z[Az]
	h_Ac = h_z[Azc]

	dim_Q = Q_z.shape[0]
	dim_mat = dim_Q + len(v_Az)

	Matrix = np.zeros((dim_mat, dim_mat))
	Matrix[: dim_Q, : dim_Q] = Q_z
	Matrix[: dim_Q, dim_Q : dim_mat] = S_Az.T
	Matrix[dim_Q : dim_mat, : dim_Q] = S_Az

	Mat_inv = np.linalg.inv(Matrix)

	product_matrix = np.zeros(dim_mat)
	product_matrix[: dim_Q] = -f_z
	product_matrix[dim_Q : dim_mat] = u0_A

	product_matrix_z = np.zeros(dim_mat)
	product_matrix_z[dim_Q : dim_mat] = u1_A

	A_t_nu = Mat_inv @ product_matrix_z
	b_t_nu = Mat_inv @ product_matrix

	A_p = A_t_nu[: p]
	A_m = A_t_nu[p : 2 * p]
	A_mat = A_p - A_m

	b_p = b_t_nu[: p]
	b_m = b_t_nu[p : 2 * p]
	b_mat = b_p - b_m

	psi = A_t_nu[: dim_Q]
	gamma = A_t_nu[dim_Q : dim_mat]

	t_1 = np.inf
	t_2 = np.inf

	num = S_Azc @ t_z - h_Ac
	dem = S_Azc @ psi

	for j in range(len(Azc)):
		numerator = - num[j]
		denominator = dem[j]

		quotient = compute_quotient(numerator, denominator)

		if quotient < t_1:
			t_1 = quotient

	for j in range(len(Az)):
		numerator = - v_Az[j]
		denominator = gamma[j]

		quotient = compute_quotient(numerator, denominator)

		if quotient < t_2:
			t_2 = quotient

	t_z = min(t_1, t_2)

	return t_z, A_mat, b_mat