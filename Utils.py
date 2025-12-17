import numpy as np

from mpmath import mp
mp.dps = 500


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


def construct_test_statistic(j_Outlier, Z, Z0, Y, outliers_obs):
	N, p = Z.shape
	nT = Z0.shape[0]

	ej = np.zeros((nT, 1))
	ej[j_Outlier][0] = 1
	zj = Z0[j_Outlier].reshape((p, 1))

	I_minusOobs = np.zeros((nT, nT))

	for i in range(nT):
		if i not in outliers_obs:
			I_minusOobs[i][i] = 1

	X_minusOobs = np.dot(I_minusOobs, Z0)

	pinv = np.linalg.pinv(X_minusOobs)
	eta = (np.identity(nT) - np.dot(np.dot(zj.T, pinv), I_minusOobs))

	etaj = np.zeros((N, 1))
	etaj[N - nT : , :] = eta.T @ ej

	etaT_yobs = np.dot((etaj.T), Y)[0]

	return etaj, etaT_yobs


def calculate_a_b(etaj, Y, Sigma):
	N = Y.shape[0]

	e1 = etaj.T @ Sigma @ etaj
	a = (Sigma @ etaj) / e1

	a = a.reshape(-1, 1)

	e2 = np.identity(N) - a @ etaj.T
	b = e2 @ Y

	return a, b.reshape(-1, 1), e1[0][0]


def compute_pivot(intervals_lemma_w, etaj, etaT_yobs, Cov_matrix, tn_mu = 0):
	tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, Cov_matrix), etaj))[0][0]

	intervals_lemma = []

	for inter in intervals_lemma_w:
		intervals_lemma.append([inter[0], inter[1]])

	new_interval_lemma = []

	for each_interval in intervals_lemma:
		if len(new_interval_lemma) == 0:
			new_interval_lemma.append(each_interval)
		else:
			sub = each_interval[0] - new_interval_lemma[-1][1]
			if abs(sub) < 0.01:
				new_interval_lemma[-1][1] = each_interval[1]
			else:
				new_interval_lemma.append(each_interval)

	intervals_lemma = new_interval_lemma

	numerator = 0
	denominator = 0

	for each_interval in intervals_lemma:
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