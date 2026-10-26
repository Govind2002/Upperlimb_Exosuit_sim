#!/usr/bin/env python3
"""
Level-A reference simulation (pure Python, no OpenSim): GC vs AGC on the paper's 1-DOF elbow model.

What it implements (paper equation numbers in brackets)
  total joint torque the human+exosuit must supply   tau_t = Ie*th'' + be*th' + Y*sigma   (1),(8)
  Y = g*sin(th),  sigma = m_e*l_c + M*(l_e + l_w)
  GC  : tau_a = sigma_nominal * Y                                                    (10)
  AGC : tau_a = sigma_hat * Y,  Theta_hat' = GAMMA * Z^T * xi                         (15),(19)-(23)
        Z = [th'', th', Y],  Theta = [Ie, be, sigma],  xi = tau_t - Z*Theta_hat
  human: tau_h = tau_t - tau_a   (human assumed to track the reference, Figs. 3-4: the summing junction)
  cable : tau_m = tau_a / G,  tau_a_real = G * tau_m, with G from (7); cable can only pull, so tau_m >= 0.

What it does NOT implement: muscles, metabolic cost, forward dynamics, OpenSim.
Use it to (i) understand the algorithm, (ii) tune gains, (iii) have numbers to compare your OpenSim pipeline against.

Run:  python agc_gc_reference_sim.py
"""
import numpy as np
from scipy.integrate import solve_ivp

# ---------------- parameters ----------------
g = 9.8066
m_e, l_c, l_e, l_w = 1.5343, 0.1815, 0.28, 0.04          # Table II
b_e = 0.5                                                  # Table II
a, b = 0.05, 0.10                                          # strap geometry, Table II
Rm, mu, phi, N = 0.03, 0.07, np.pi, 25.0                   # Table II
I_FOREARM_ABOUT_ELBOW = 0.0705                             # from arm26 mass properties (parallel-axis), NOT Table II's 0.0201
T_END = 6.0

# adaptation: Theta_hat' = GAMMA * Z^T xi.  Table III says "Lambda = diag(0.025)" with Theta_hat' = inv(Lambda) Z^T xi,
# which read literally means GAMMA = 40.  GAMMA = 0.025 reproduces the slow convergence in Fig. 6f.  Test both.
GAMMA_TABLE_III_LITERAL = 1.0 / 0.025
GAMMA_FIG6F_LIKE = 0.025
THETA0 = np.array([0.1, 0.1, 0.1])                         # Table III


def reference(t, amp=np.pi / 2, f=0.5):
    """theta_r = (amp/2)*(1-cos(2 pi f t)): 0..90 deg, 0.5 Hz (matches the RANGE/PERIOD seen in Fig. 6a; my choice)."""
    w = 2 * np.pi * f
    th = 0.5 * amp * (1 - np.cos(w * t))
    thd = 0.5 * amp * w * np.sin(w * t)
    thdd = 0.5 * amp * w**2 * np.cos(w * t)
    return th, thd, thdd


def J_f(theta):
    """moment arm magnitude |dh/dtheta| from Eq. (4); the paper's Eq.(5) as printed is negative."""
    return np.sqrt(a**2 + b**2) * np.sin(np.arctan(a / b) + theta / 2)


def G(theta, thd, mu_=mu):
    """Eq. (7). Direction of motor rotation is assumed to follow the sign of the elbow velocity (flexion = reeling in)."""
    base = N * J_f(theta) / Rm
    return base * np.exp(-mu_ * phi) if thd >= 0 else base * np.exp(+mu_ * phi)


def true_params(M):
    sigma = m_e * l_c + M * (l_e + l_w)
    Ie = I_FOREARM_ABOUT_ELBOW + M * (l_e + l_w) ** 2      # payload as a point mass at l_e + l_w
    return Ie, sigma


def simulate(M, controller, M_nominal=0.0, gamma=GAMMA_FIG6F_LIKE, tau_m_max=np.inf, mu_plant=mu, M_step=None, t_step=3.0):
    """controller in {'none','GC','AGC'}.  M_step: payload after t_step (payload change mid-trial)."""
    t = np.linspace(0, T_END, 60001)

    def M_at(tt):
        return M if (M_step is None or tt < t_step) else M_step

    def tau_t(tt):
        th, thd, thdd = reference(tt)
        Ie, sig = true_params(M_at(tt))
        return Ie * thdd + b_e * thd + g * np.sin(th) * sig

    def Z(tt):
        th, thd, thdd = reference(tt)
        return np.array([thdd, thd, g * np.sin(th)])

    def rhs(tt, th_hat):
        z = Z(tt)
        xi = tau_t(tt) - z @ th_hat
        return gamma * z * xi

    if controller == "AGC":
        sol = solve_ivp(rhs, (0, T_END), THETA0, t_eval=t, max_step=1e-3, rtol=1e-8, atol=1e-10, method="LSODA")
        th_hat = sol.y.T
    else:
        th_hat = np.tile(THETA0, (len(t), 1))

    ta, tm, th_h = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    for k, tt in enumerate(t):
        th, thd, thdd = reference(tt)
        Y = g * np.sin(th)
        if controller == "none":
            ta_des = 0.0
        elif controller == "GC":
            ta_des = (m_e * l_c + M_nominal * (l_e + l_w)) * Y
        else:
            ta_des = th_hat[k, 2] * Y
        ta_des = max(ta_des, 0.0)                          # cable cannot push
        Gk = G(th, thd, mu_=mu)                            # controller's model of G
        tm[k] = min(ta_des / Gk, tau_m_max) if ta_des > 0 else 0.0
        ta[k] = G(th, thd, mu_=mu_plant) * tm[k]           # what the plant actually delivers
        th_h[k] = tau_t(tt) - ta[k]
    return t, th_h, ta, tm, th_hat


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


if __name__ == "__main__":
    print("Reduction in RMS human elbow torque vs unassisted (%)   [paper Table IV: GC / AGC]")
    paper = {0: (35, 28), 3: (65, 50), 5: (67, 66), 10: (67, 66)}
    print(f"{'payload':>8} {'GC (true)':>10} {'AGC g=0.025':>12} {'AGC g=40':>10}   paper GC/AGC")
    for M in (0, 3, 5, 10):
        _, h0, *_ = simulate(M, "none")
        _, hg, *_ = simulate(M, "GC", M_nominal=M)
        _, ha, *_ = simulate(M, "AGC", gamma=GAMMA_FIG6F_LIKE)
        _, hb, *_ = simulate(M, "AGC", gamma=GAMMA_TABLE_III_LITERAL)
        r = lambda h: 100 * (1 - rms(h) / rms(h0))
        print(f"{M:6.1f}kg {r(hg):10.1f} {r(ha):12.1f} {r(hb):10.1f}   {paper[M][0]}/{paper[M][1]}")

    print("\n2.5 kg (YOUR extra condition, not in the paper)")
    t, h0, *_ = simulate(2.5, "none")
    for name, kw in (("GC true", dict(controller="GC", M_nominal=2.5)),
                     ("GC nominal=0kg", dict(controller="GC", M_nominal=0.0)),
                     ("GC nominal=5kg", dict(controller="GC", M_nominal=5.0)),
                     ("AGC", dict(controller="AGC"))):
        _, h, *_ = simulate(2.5, **kw)
        print(f"  {name:<16} RMS tau_h={rms(h):6.3f} N m   reduction={100*(1-rms(h)/rms(h0)):5.1f}%   peak|tau_h|={np.max(np.abs(h)):6.2f}")
    print(f"  {'unassisted':<16} RMS tau_h={rms(h0):6.3f} N m")

    # convergence time of sigma_hat for 2.5 kg
    _, _, _, _, th_hat = simulate(2.5, "AGC")
    _, sig = true_params(2.5)
    err = np.abs(sig - th_hat[:, 2])
    tt = np.linspace(0, T_END, len(err))
    for thr in (0.10, 0.05):
        idx = np.where(err > thr * sig)[0]
        print(f"  sigma_hat within {int(thr*100)}% of sigma ({sig:.3f}) after t = {tt[idx[-1]] if len(idx) else 0:.2f} s")

    print("\nPayload step 2.5 -> 5 kg at t=3 s (GC nominal = 2.5 kg)")
    _, h_gc, *_ = simulate(2.5, "GC", M_nominal=2.5, M_step=5.0)
    _, h_agc, *_ = simulate(2.5, "AGC", M_step=5.0)
    m2 = tt > 4.0
    print(f"  RMS tau_h after t>4 s : GC(nominal)={rms(h_gc[m2]):.3f}   AGC={rms(h_agc[m2]):.3f}")
