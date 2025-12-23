import Gen_data, Utils, TL_Training, Compute_step_size
import numpy as np


def compute_solution_path(X, X0, Y0_mask, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, zk_threshold):
	N, p = X.shape
	nT = X0.shape[0]
	zk = -zk_threshold

	list_zk = [zk]
	list_g_beta = []
	list_f_beta = []

	while zk < zk_threshold:
		w_interval_z = [zk]
		d_interval_z = [zk]
		intervals_wdz = [zk]

		Yz = a_y * zk + b_y
		Yz = Yz.ravel()
		Y0z = Y0_mask @ Yz
		Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, u0w_z, u1w_z = Gen_data.Construct_matrix(X, Yz, gamma)

		u0w_z[: N] = b_y.reshape(N)
		u0w_z[N : 2 * N] = -b_y.reshape(N)
		u1w_z[: N] = a_y.reshape(N)
		u1w_z[N : 2 * N] = -a_y.reshape(N)

		info_w = TL_Training.training_qp_w(Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, X, lambda_w, alpha)
		if info_w == None:
			return None, None, None

		w_z = info_w['w_plus'] - info_w['w_minus']
		Q_wz = info_w['Q']
		f_wz = info_w['f']
		t_wz = info_w['t_value']
		prob_w = info_w['Problem']
		v_zw = prob_w.constraints[0].dual_value

		KKT_w = TL_Training.check_KKT(Q_wz, f_wz, Sw_z, hw_z, t_wz, v_zw)
		if not KKT_w:
			print(' | W Unsatisfy KKT Condition')
			return None, None, None

		Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, u0d_z, u1d_z = Gen_data.Construct_matrix(X0, Y0z, gamma)

		hd_z[: nT] = Y0z - X0 @ w_z
		hd_z[nT : 2 * nT] = - (Y0z - X0 @ w_z)
		hd_z[3 * nT : 4 * nT] = gamma * np.ones(nT)

		info_d = TL_Training.training_qp_delta(Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, X0, lambda_d, alpha)
		if info_d == None:
			return None, None, None

		d_z = info_d['d_plus'] - info_d['d_minus']
		Q_dz = info_d['Q']
		f_dz = info_d['f']
		t_dz = info_d['t_value']

		prob_d = info_d['Problem']
		v_dz = prob_d.constraints[0].dual_value

		KKT_d = TL_Training.check_KKT(Q_dz, f_dz, Sd_z, hd_z, t_dz, v_dz)
		if not KKT_d:
			print(' | DELTA Unsatisfy KKT Condition')
			return None, None, None

		Beta_z = w_z + d_z
		res_target_z = Y0z - X0 @ Beta_z

		outlier_mask_z = np.abs(res_target_z) > xi_threshold
		outlier_z = []
		for i, o in enumerate(outlier_mask_z):
			if o:
				outlier_z.append(i)

		Az, Azc, vw_Az, vw_Azc, S_Az, S_Azc = Utils.construct_Active_lagrane(v_zw, Sw_z)
		tr_w, A_w, b_w = Compute_step_size.compute_step_size(Q_wz, f_wz, S_Az, S_Azc, hw_z, u0w_z, u1w_z, X, t_wz, vw_Az, Az, Azc)

		if zk + tr_w < zk_threshold:
			w_interval_z.append(zk + tr_w)
		else:
			w_interval_z.append(zk_threshold)

		b_yT = b_y[N - nT : ,].reshape(nT)
		a_yT = a_y[N - nT : ,].reshape(nT)

		u0d_z[: nT] = b_yT - X0 @ b_w
		u0d_z[nT : 2 * nT] = - (b_yT - X0 @ b_w)
		u1d_z[: nT] = a_yT - X0 @ A_w
		u1d_z[nT : 2 * nT] = - (a_yT - X0 @ A_w)

		Oz, Ozc, vd_Oz, vd_Ozc, S_Oz, S_Ozc = Utils.construct_Active_lagrane(v_dz, Sd_z)
		tr_d, A_d, b_d = Compute_step_size.compute_step_size(Q_dz, f_dz, S_Oz, S_Ozc, hd_z, u0d_z, u1d_z, X0, t_dz, vd_Oz, Oz, Ozc)

		if zk + tr_d < zk_threshold:
			d_interval_z.append(zk + tr_d)
		else:
			d_interval_z.append(zk_threshold)

		intervals_wdz.append(min(w_interval_z[1], d_interval_z[1]))

		g_beta = a_y.reshape(N) - X @ (A_w + A_d)
		f_beta = b_y.reshape(N) - X @ (b_w + b_d)
		g_beta = g_beta[N - nT :]
		f_beta = f_beta[N - nT :]

		list_zk.append(intervals_wdz[1])
		list_g_beta.append(g_beta)
		list_f_beta.append(f_beta)

		zk = intervals_wdz[1] + 1e-4

	return list_zk, list_g_beta, list_f_beta