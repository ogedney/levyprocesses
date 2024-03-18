# Old helper code

# Plot weights against Kalman means
# print(len(a_ii1s), self.omegas.shape)
# plt.scatter(self.a_s[0, :], self.omegas)
# plt.plot(y_values[j] * np.ones(100), np.linspace(self.omegas.min(), self.omegas.max(), 100), color='orange')
# plt.xlim(min(min(self.a_s[0, :]), y_values[j]) - 0.1, max(max(self.a_s[0, :]), y_values[j]) + 0.1)
# plt.xlabel("Kalman mean y value")
# plt.ylabel("Log weights")
# plt.title(f'Time index = {j}, observation y_t = {round(y_values[j], 4)}')  # mean F_i = {round(F_sum / self.N, 4)}')
# plt.show()

# a_ii, C_ii, a_ii1, C_ii1, log_like, exp_bit, log_bit, F_i, y_hat = self.kalman_update(a, C_tilde, y_values[j], C, self.Kv, Ai, C_e)

# def kalman_update(self, a, P, y, Z, C_v, T, C_e):
#     """Kalman filtering recursion. Copied across from Simon's Matlab code."""
#     # [a_filt,P_filt,a_pred,P_pred,log_like,y_samp, exp_bit, log_bit]=kalman_update(a,P,y,Z,C_v,T,H,C_e)
#
#     # Prediction step
#     a_pred = T @ a
#     # print(f'a = {[round(a[0, 0], 4), round(a[1, 0], 4)]}, T = {[[round(T[0, 0], 4), round(T[0, 1], 4)], [round(T[1, 0], 4), round(T[1, 1], 4)]]},'
#     #       f'a_pred = {[round(a_pred[0, 0], 4), round(a_pred[1, 0], 4)]}')
#     # print(f'a_pred shape {a_pred.shape}')
#     P_pred = T @ P @ T.T + np.array([[C_e, 0], [0, 0]])
#     # print(f'T = {[[round(T[0, 0], 4), round(T[0, 1], 4)], [round(T[1, 0], 4), round(T[1, 1], 4)]]}, '
#     #       f'P = {[[round(P[0, 0], 4), round(P[0, 1], 4)], [round(P[1, 0], 4), round(P[1, 1], 4)]]}')
#     # print(f'C_e = {round(C_e, 4)}, '
#     #       f'P_pred = {[[round(P_pred[0, 0], 4), round(P_pred[0, 1], 4)], [round(P_pred[1, 0], 4), round(P_pred[1, 1], 4)]]}\n')
#     # print(f'P_pred shape {P_pred.shape}')
#     if not np.all(np.abs(P_pred-P_pred.T) < 10**-6):
#         raise RuntimeError
#
#     # Correction step
#     # Kalman gain
#     K = P_pred @ Z.T @ np.linalg.inv(Z @ P_pred @ Z.T + C_v)
#     # print(f'P_pred = {[[round(P_pred[0, 0], 4), round(P_pred[0, 1], 4)], [round(P_pred[1, 0], 4), round(P_pred[1, 1], 4)]]},'
#     #       f'w = {round((y - Z @ a_pred)[0], 4)}, K = {[round(K[0, 0], 4), round(K[1, 0], 4)]}')
#     # print(f'K shape = {K.shape}')
#     F = Z @ P_pred @ Z.T + C_v
#     # print()
#     # print(f'F shape = {F.shape}')
#     mu_y = Z @ a_pred
#     # print(f'mu_y shape = {mu_y.shape}')
#
#     w = y - mu_y
#     # print(
#     #     f"{f'P_pred = {[[round(P_pred[0, 0], 4), round(P_pred[0, 1], 4)], [round(P_pred[1, 0], 4), round(P_pred[1, 1], 4)]]}' : <50} "
#     #     f"{f'w = {round(w[0][0], 4)}' : <15} {f'K = {[round(K[0, 0], 4), round(K[1, 0], 4)]}' : <30}")
#     # print(f'w shape = {w.shape}')
#     a_filt = a_pred + K * w
#     # print(f'a_filt shape = {a_filt.shape}')
#     P_filt = (np.eye(len(a_pred)) - K @ Z) @ P_pred
#     if not np.all(np.abs(P_filt-P_filt.T) < 10**-6):
#         raise RuntimeError
#     # print(f'P_filt shape = {P_filt.shape}')
#
#     # Calculate likelihood
#
#     # Exponent:
#     exp_bit = - w ** 2 / (2 * F)
#
#     # The rest:
#     log_bit = - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(F)
#
#     log_like = exp_bit + log_bit
#     return a_filt, P_filt, a_pred, P_pred, log_like, exp_bit, log_bit, F, mu_y

#
# E_i_minus_1 = self.E_s[i]
# self.E_s[i] += (y_values[j] - y_hat) ** 2 / F_i
# self.sum_F_s[i] += np.log(F_i)
# # print(y_values[j])
# # print(y_hat)
# # print(F_i)
# # print((y_values[j] - y_hat) ** 2 / F_i)
# # print('\n')
#
# # print(C @ C_ii1 @ C.T)
#
# # Incremental log likelihood (p(y_t1:i) / p(y_t1:i-1))
# # with inverse gamma prior for sigma_w (IG(alpha_w, betaW))
# inc_likelihood = - 0.5 * np.log(F_i) \
#                  - (self.alpha_w + (j + 1) / 2) * np.log(self.beta_w + self.E_s[i] / 2) \
#                  + (self.alpha_w + j / 2) * np.log(self.beta_w + E_i_minus_1 / 2) \
#                  + scipy.special.loggamma((j + 1) / 2 + self.alpha_w) \
#                  - scipy.special.loggamma(j / 2 + self.alpha_w)
#
# il1 = - 0.5 * np.log(F_i)
# il2 = - (self.alpha_w + (j + 1) / 2) * np.log(self.beta_w + self.E_s[i] / 2) \
#                  + (self.alpha_w + j / 2) * np.log(self.beta_w + E_i_minus_1 / 2)
# il3 = + scipy.special.loggamma((j + 1) / 2 + self.alpha_w) \
#                  - scipy.special.loggamma(j / 2 + self.alpha_w)

# print(f'y_obs = {round(y_values[j], 4)}, y_hat = {round(y_hat, 4)}, F_i = {round(F_i, 4)}, '
#       f'dE = {round(((y_values[j] - y_hat) ** 2 / F_i), 4)} (~0.8), omega = {round(inc_likelihood, 4)},'
#       f' il1 = {round(il1, 4)}, il2 = {round(il2, 4)}, il3 = {round(il3, 4)}')

# print(f"{f'y_obs = {round(y_values[j], 4)}' : <20} {f'y_prev = {round(y_values[j-1], 4)}': <20} "
#       f"{f'mu_pred = {round(a_ii1[1][0], 4)}' : <20} {f'mu_filt = {round(a_ii[1][0], 4)}' : <20}"
#       f"{f'omega = {round(inc_likelihood[0][0], 4)}' : <20}")

# def generate_all_jumps(self, times):
#     self.dxs = np.zeros((len(times), self.N))
#     self.dx_squares = np.zeros((len(times), self.N))
#     for i in tqdm(range(self.N), colour='WHITE', desc='Particle jumps generated', ncols=150):
#         jumps, t_jumps = self.subordinator.generate_jumps()
#         for j in range(1, len(times)):
#             # print(j)
#             self.dxs[j, i] = np.sum(jumps[t_jumps < times[j]])
#             self.dx_squares[j, i] = np.sum(jumps[t_jumps < times[j]] ** 2)
#             jumps = jumps[t_jumps > times[j]]
#             t_jumps = t_jumps[t_jumps > times[j]]
