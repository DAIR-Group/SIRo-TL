def merge_intervals(intervals):
	intervals_sorted = sorted(intervals, key = lambda x : x[0])
	merged = []

	for l, r in intervals_sorted:
		if r <= l:
			continue

		if not merged or l > merged[-1][1]:
			merged.append([l, r])
		else:
			merged[-1][1] = max(merged[-1][1], r)

	result = []
	for a, b in merged:
		result.append((a, b))

	return result


def intersect_two_lists(A, B):
	if not A or not B:
		return []

	result = []
	i = 0
	j = 0

	while i < len(A) and j < len(B):
		lo = max(A[i][0], B[j][0])
		hi = min(A[i][1], B[j][1])

		if hi > lo:
			result.append((lo, hi))

		if A[i][1] < B[j][1]:
			i += 1
		else:
			j += 1

	return merge_intervals(result)


def intersect_many(lists, base):
	current = [base]

	for lst in lists:
		current = intersect_two_lists(current, lst)
		if not current:
			break

	return current


def subtract_intervals(base, subs):
	bl, br = base

	if bl > br:
		return []

	subs_merged = merge_intervals(subs)

	result = []
	left = bl

	for l, r in subs_merged:
		if r <= bl or l >= br:
			continue

		if l > left:
			result.append((left, l))

		left = r

	if left < br:
		result.append((left, br))

	return result


def compute_V_t_i(t_idx, i_idx, list_zk, xi, f_matrix, g_matrix):
	z_prev = float(list_zk[t_idx])
	z_curr = float(list_zk[t_idx + 1])

	f_ti = float(f_matrix[i_idx])
	g_ti = float(g_matrix[i_idx])

	base = (z_prev, z_curr)

	if abs(g_ti) <= 0:
		if abs(f_ti) >= xi:
			return [base]
		else:
			return []

	if g_ti > 0:
		a = (xi - f_ti) / g_ti
		b = (-xi - f_ti) / g_ti
	else:
		a = (-xi - f_ti) / g_ti
		b = (xi - f_ti) / g_ti

	part_left  = (z_prev, min(z_curr, b))
	part_right = (max(z_prev, a), z_curr)

	parts = []

	for l, r in (part_left, part_right):
		L = max(l, z_prev)
		R = min(r, z_curr)

		if R > L:
			parts.append((L, R))

	V_t_i = merge_intervals(parts)

	return V_t_i


def Outlier_detection_intervals(list_zk, list_f_mat, list_g_mat, xi, Outlier_obs):
	T = len(list_zk) - 1

	n = len(list_f_mat[0])
	all_ct = []

	t = 0
	while t < T:
		base = (list_zk[t], list_zk[t + 1])

		V_list = []
		i_var = 0
		while i_var < n:
			V_t_i = compute_V_t_i(t, i_var, list_zk, xi, list_f_mat[t], list_g_mat[t])
			V_list.append(V_t_i)
			i_var += 1

		list_for_A = []
		idx_o = 0
		while idx_o < len(Outlier_obs):
			list_for_A.append(V_list[Outlier_obs[idx_o]])
			idx_o += 1

		A_t = intersect_many(list_for_A, base)
		if not A_t:
			t += 1
			continue

		non_out = []
		idx = 0
		while idx < n:
			if idx not in Outlier_obs:
				non_out.append(idx)
			idx += 1

		complements = []
		for i_non in non_out:
			comp = subtract_intervals(base, V_list[i_non])
			complements.append(comp)

		has_empty = False

		for comp in complements:
			if not comp:
				has_empty = True
				break

		if has_empty:
			t += 1
			continue

		B_t = intersect_many(complements, base)

		if not B_t:
			t += 1
			continue

		C_t = intersect_two_lists(A_t, B_t)
		for interval in C_t:
			all_ct.append(interval)

		t += 1
		
	return all_ct