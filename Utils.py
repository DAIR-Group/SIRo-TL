import numpy as np

from mpmath import mp
mp.dps = 1000


def construct_Y0_mask(nT, N):
	Y0_mask = np.zeros((nT, N))
	Y0_mask[:, N - nT: ] = np.eye(nT)

	return Y0_mask


def construct_Active_lagrane(u_qp, S):
	A = []
	Ac = []

	for i in range(len(u_qp)):
		u = abs(u_qp[i])
		if u >= 1e-12:
			A.append(i)
		else:
			Ac.append(i)

	S_A = S[A]
	S_Ac = S[Ac]

	u_A = u_qp[A]
	u_Ac = u_qp[Ac]

	return A, Ac, u_A, u_Ac, S_A, S_Ac


def construct_test_statistic(i_Outlier, X, X0, Y, outliers_obs):
	N, p = X.shape
	nT = X0.shape[0]

	ei = np.zeros((nT, 1))
	ei[i_Outlier][0] = 1
	xi = X0[i_Outlier].reshape((p, 1))

	I_minusOobs = np.zeros((nT, nT))

	for i in range(nT):
		if i not in outliers_obs:
			I_minusOobs[i][i] = 1

	X_minusOobs = np.dot(I_minusOobs, X0)

	pinv = np.linalg.pinv(X_minusOobs)
	eta = (np.identity(nT) - np.dot(np.dot(xi.T, pinv), I_minusOobs))

	etai = np.zeros((N, 1))
	etai[N - nT : , :] = eta.T @ ei

	etaT_yobs = np.dot((etai.T), Y)[0]

	return etai, etaT_yobs


def calculate_a_b(etai, Y, Sigma):
	N = Y.shape[0]

	e1 = etai.T @ Sigma @ etai
	a = (Sigma @ etai) / e1

	a = a.reshape(-1, 1)

	e2 = np.identity(N) - a @ etai.T
	b = e2 @ Y

	return a, b.reshape(-1, 1), e1[0][0]


def compute_pivot(final_intervals, etai, etaT_yobs, Cov_matrix, tn_mu = 0):
	tn_sigma = np.sqrt(np.dot(np.dot(etai.T, Cov_matrix), etai))[0][0]

	list_intervals = []

	for inter in final_intervals:
		list_intervals.append([inter[0], inter[1]])

	new_list_intervals = []

	for each_interval in list_intervals:
		if len(new_list_intervals) == 0:
			new_list_intervals.append(each_interval)
		else:
			sub = each_interval[0] - new_list_intervals[-1][1]
			if abs(sub) < 0.01:
				new_list_intervals[-1][1] = each_interval[1]
			else:
				new_list_intervals.append(each_interval)

	list_intervals = new_list_intervals

	numerator = 0
	denominator = 0

	for each_interval in list_intervals:
		al = each_interval[0]
		ar = each_interval[1]

		denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

		if etaT_yobs >= ar:
			numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
		elif (etaT_yobs >= al) and (etaT_yobs < ar):
			numerator = numerator + mp.ncdf((etaT_yobs - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

	if denominator != 0:
		return float(numerator/denominator)

	else:
		return None