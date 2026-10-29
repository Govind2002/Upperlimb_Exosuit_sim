#!/usr/bin/env python3
"""
Offline GC-vs-AGC pipeline (human model H1: the human tracks the reference and supplies tau_h = tau_t - tau_a).

Torque source for tau_t
  --source analytic   : Eq. (1) with the model-consistent inertia   (runs anywhere, tested)
  --source opensim    : OpenSim InverseDynamicsSolver on the .osim   (needs OpenSim; API calls marked [verify])

Run
  python offline_pipeline.py --payload 2.5
  python offline_pipeline.py --payload 2.5 --nominal 5 --noise-acc 0.5 --noise-tau 0.2 --delay 0.02
  python offline_pipeline.py --payload 2.5 --source opensim --model models/arm26_elbow_payload_2p5kg.osim
"""
import argparse
import numpy as np

g = 9.8066
m_e, l_c, l_e, l_w, b_e = 1.5343, 0.1815, 0.28, 0.04, 0.5
I_FOREARM_ELBOW = 0.0705           # arm26 forearm+hand about the elbow axis (parallel-axis theorem)


# ----------------------------------------------------------------------------- reference motion
def reference(t, amp=np.pi / 2, f=0.5):
    w = 2 * np.pi * f
    return (0.5 * amp * (1 - np.cos(w * t)), 0.5 * amp * w * np.sin(w * t), 0.5 * amp * w**2 * np.cos(w * t))


# ----------------------------------------------------------------------------- torque sources
class AnalyticTorque:
    def __init__(self, M):
        self.sigma = m_e * l_c + M * (l_e + l_w)
        self.I = I_FOREARM_ELBOW + M * (l_e + l_w) ** 2

    def __call__(self, th, thd, thdd):
        return self.I * thdd + b_e * thd + self.sigma * g * np.sin(th)


class OpenSimTorque:
    """Net elbow flexion torque from OpenSim inverse dynamics. [verify] every API call on your version."""
    def __init__(self, path):
        import opensim as osim
        self.osim = osim
        self.model = osim.Model(path)
        self.state = self.model.initSystem()
        for i in range(self.model.getForceSet().getSize()):
            self.model.updForceSet().get(i).setAppliesForce(self.state, False)
        self.solver = osim.InverseDynamicsSolver(self.model)
        self.elbow = self.model.getCoordinateSet().get("r_elbow_flex")
        self.iq = self.elbow.getMobilizerQIndex()

    def __call__(self, th, thd, thdd):
        self.elbow.setValue(self.state, th)
        self.elbow.setSpeedValue(self.state, thd)
        self.model.realizeVelocity(self.state)
        udot = self.osim.Vector(self.state.getNU(), 0.0)
        udot.set(self.iq, thdd)
        return self.solver.solve(self.state, udot).get(self.iq)


# ----------------------------------------------------------------------------- exosuit actuator (Eqs. 2-7)
class CableActuator:
    def __init__(self, a=0.05, b=0.10, Rm=0.03, mu=0.07, phi=np.pi, N=25.0, tau_m_max=np.inf):
        self.a, self.b, self.Rm, self.mu, self.phi, self.N, self.tau_m_max = a, b, Rm, mu, phi, N, tau_m_max

    def J(self, th):                                   # |dh/dtheta|, see guide Section 2.6
        return np.hypot(self.a, self.b) * np.sin(np.arctan(self.a / self.b) + th / 2)

    def G(self, th, thd, mu=None):
        mu = self.mu if mu is None else mu
        base = self.N * self.J(th) / self.Rm
        return base * np.exp(-mu * self.phi) if thd >= 0 else base * np.exp(mu * self.phi)

    def command(self, tau_a_des, th, thd):             # controller's view (nominal mu)
        return float(np.clip(tau_a_des / self.G(th, thd), 0.0, self.tau_m_max))

    def deliver(self, tau_m, th, thd, mu_true=None):   # plant's view (true mu)
        return self.G(th, thd, mu_true) * tau_m


# ----------------------------------------------------------------------------- controllers
class GC:
    def __init__(self, sigma):
        self.s = sigma

    def sigma_hat(self):
        return self.s

    def update(self, Z, tau_t, dt):
        pass


class AGC:
    """Adaptive Gravity Compensation using the exact paper adaptation law 
    with Euler integration and projection modification."""
    def __init__(self, gamma=0.025, theta0=(0.1, 0.1, 0.1), th_min=None, th_max=None):
        self.th = np.array(theta0, float)
        self.gamma = gamma  # Corresponds to learning rate / adaptation gain scaling
        
        # Projection modification bounds to prevent parameter drift/divergence
        self.th_min = th_min if th_min is not None else np.array([0.0, 0.0, 0.0])
        self.th_max = th_max if th_max is not None else np.array([ 5.0,  5.0,  10.0])

    def sigma_hat(self):
        return self.th[2]

    def update(self, Z, tau_t, dt):
        """
        Z  : Regressor vector [thdd, thd, g*sin(th)]
        tau_t: Total torque (tau_h + G*tau_m)
        dt : Simulation time step
        """
        # 1. Compute prediction error xi based on Eq. (20):
        #    xi = (tau_h + G*tau_m) - Z @ self.th  --> which equals tau_t - Z @ self.th
        xi = tau_t - (Z @ self.th)
        
        # 2. Apply paper's adaptation law Eq. (23): \dot{\hat{\Theta}} = \Lambda^{-1} Z^T \xi[cite: 5]
        #    Discretized using forward Euler: \hat{\Theta} += dt * \Lambda^{-1} * Z * \xi
        d_th = self.gamma * Z * xi
        self.th += d_th * dt
        
        # 3. Projection modification: constrain parameter estimates within valid bounds
        self.th = np.clip(self.th, self.th_min, self.th_max)


class Unassisted:
    def sigma_hat(self):
        return 0.0

    def update(self, *a):
        pass


# ----------------------------------------------------------------------------- one run
def run(controller, torque_fn, act, t_end=6.0, dt=1e-3, noise_acc=0.0, noise_tau=0.0, noise_th=0.0, delay=0.0,
        tau_h_gain_err=1.0, mu_true=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, t_end, dt)
    n = len(t)
    log = {k: np.zeros(n) for k in ("th", "tau_t", "tau_a", "tau_h", "tau_m", "sigma_hat")}
    nd = int(round(delay / dt))
    buf = []                                            # delay line for the measured regressor/torque
    for k, tk in enumerate(t):
        th, thd, thdd = reference(tk)
        tau_t = torque_fn(th, thd, thdd)
        Y = g * np.sin(th)
        tau_a_des = max(controller.sigma_hat() * Y, 0.0)
        tau_m = act.command(tau_a_des, th, thd)
        tau_a = act.deliver(tau_m, th, thd, mu_true)
        tau_h = tau_t - tau_a
        # what the controller can measure (corrupted)
        th_m = th + noise_th * rng.standard_normal()
        thd_m = thd
        thdd_m = thdd + noise_acc * rng.standard_normal()
        tau_t_m = tau_a + tau_h_gain_err * tau_h + noise_tau * rng.standard_normal()   # tau_h_hat + tau_a
        Z = np.array([thdd_m, thd_m, g * np.sin(th_m)])
        buf.append((Z, tau_t_m))
        Zd, taud = buf[max(0, len(buf) - 1 - nd)]
        controller.update(Zd, taud, dt)
        for key, val in (("th", th), ("tau_t", tau_t), ("tau_a", tau_a), ("tau_h", tau_h), ("tau_m", tau_m),
                         ("sigma_hat", controller.sigma_hat())):
            log[key][k] = val
    log["t"] = t
    return log


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=float, default=2.5)
    ap.add_argument("--nominal", type=float, default=0.0, help="payload the nominal GC was tuned for (kg)")
    ap.add_argument("--source", choices=["analytic", "opensim"], default="analytic")
    ap.add_argument("--model", default="arm26_elbow_payload_2p5kg.osim")
    ap.add_argument("--gamma", type=float, default=0.025)
    ap.add_argument("--noise-acc", type=float, default=0.0)
    ap.add_argument("--noise-tau", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()

    torque_fn = AnalyticTorque(a.payload) if a.source == "analytic" else OpenSimTorque(a.model)
    act = CableActuator()
    sig_true = m_e * l_c + a.payload * (l_e + l_w)
    sig_nom = m_e * l_c + a.nominal * (l_e + l_w)
    cases = {
        "unassisted": Unassisted(),
        "GC true": GC(sig_true),
        f"GC nominal({a.nominal}kg)": GC(sig_nom),
        "AGC": AGC(gamma=a.gamma),
    }
    base = None
    print(f"payload {a.payload} kg   sigma_true={sig_true:.4f}   source={a.source}")
    print(f"{'controller':<22}{'RMS tau_h':>10}{'reduction':>11}{'peak|tau_h|':>13}{'final sigma_hat':>17}")
    for name, c in cases.items():
        L = run(c, torque_fn, act, noise_acc=a.noise_acc, noise_tau=a.noise_tau, delay=a.delay)
        r = rms(L["tau_h"])
        base = r if base is None else base
        print(f"{name:<22}{r:10.3f}{100*(1-r/base):10.1f}%{np.max(np.abs(L['tau_h'])):13.2f}{L['sigma_hat'][-1]:17.3f}")
