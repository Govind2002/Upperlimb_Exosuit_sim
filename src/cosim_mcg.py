#!/usr/bin/env python3
"""
MUSCLE-DRIVEN forward-dynamics co-simulation with a Python Muscle Command Generator (MCG),
following the architecture of Sambhav et al. (Fig. 1): the virtual human is the arm26 musculoskeletal model driven by muscle excitations.

Per control period (default 5 ms), exactly the paper's chain:
  1. desired acceleration    a_des = th_r'' + kv (th_r' - th') + kp (th_r - th)                    (their Eq. 2)
  2. inverse dynamics        tau_req = ID(th, th', a_des) - tau_a_known       OpenSim, forces off   (their Eq. 1; A = exosuit torque)
  3. static optimisation     min sum a_i^2   s.t.  sum d_i F_i(a_i) = tau_req,   0 <= a_i <= 1      (their Eq. 3-6)
                             F_i = Fmax cos(phi) [ f_l f_v a_i + f_p ]  with f_l, f_v, f_p, d_i read from the OpenSim state
                             solved EXACTLY by a 1-D bisection (separable convex problem), no fmincon needed
  4. inverse activation dyn. u_i = a_i + kappa(a_i, a*_i) (a*_i - a_i)/T_lookahead                  (their Eq. 7-8, CMC look-ahead)
  5. forward dynamics        excitations u_i -> PrescribedController -> OpenSim Manager.integrate one period
The exosuit (GC / AGC) enters as an ideal elbow torque tau_a on CoordinateActuator "exo_elbow_torque".

Plants
  --plant opensim : arm26_elbow_payload_2p5kg.osim, Thelen2003 muscles                            (UNTESTED here; calls marked [verify])
  --plant mock    : a TEST DOUBLE (1-DOF, six ideal Hill-like muscles with INVENTED constant moment arms) so the MCG logic and
                    the loop can be exercised without OpenSim. NEVER use mock results as findings.

Run
  python cosim_mcg.py --plant mock --payload 2.5
  python cosim_mcg.py --plant opensim --payload 2.5 --selftest
  python cosim_mcg.py --plant opensim --payload 2.5 --traj minjerk --T 1 --metabolics --out results/mcg --plot
"""
import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from offline_pipeline import GC, AGC, Unassisted, CableActuator, m_e, l_c, l_e, l_w, I_FOREARM_ELBOW, g
from cosim_forward import OpenSimID, AnalyticID, make_reference, rms

MUSCLES = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]
FLEXORS, EXTENSORS = [3, 4, 5], [0, 1, 2]
T_ACT, T_DEACT = 0.01, 0.04            # arm26 Thelen2003 activation / deactivation time constants (read from the file)
U_MIN, U_MAX = 0.01, 1.0


# =====================================================================================================  MCG building blocks
def solve_min_sumsq(c, b):
    """min sum a^2  s.t.  c.a = b,  0 <= a <= 1.   KKT => a = clip(lam*c, 0, 1); g(lam) = sum c*clip(lam*c) is monotone => bisection.
    Returns (a, shortfall) where shortfall = b - c.a (non-zero only if the torque is infeasible)."""
    c = np.asarray(c, float)
    sign = 1.0 if b >= 0 else -1.0
    idx = (c * sign) > 1e-9                          # muscles able to produce torque of the required sign
    if not idx.any():
        return np.zeros_like(c), b
    lam_end = sign / np.min(np.abs(c[idx]))          # beyond this every useful muscle is saturated
    f = lambda lam: float(np.sum(c * np.clip(lam * c, 0.0, 1.0)))
    if abs(f(lam_end)) <= abs(b):
        a = np.clip(lam_end * c, 0.0, 1.0)
        return a, b - float(c @ a)
    lo, hi = 0.0, lam_end
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if abs(f(mid)) < abs(b):
            lo = mid
        else:
            hi = mid
    a = np.clip(0.5 * (lo + hi) * c, 0.0, 1.0)
    return a, b - float(c @ a)


def inverse_activation(a, a_star, lookahead, t_act=T_ACT, t_deact=T_DEACT):
    """Excitation u that moves activation a toward a_star within the look-ahead window, inverting Thelen's
    da/dt = (u - a)/kappa,  kappa = t_act (0.5+1.5a) if u > a else t_deact/(0.5+1.5a)."""
    a, a_star = np.asarray(a, float), np.asarray(a_star, float)
    rate = (a_star - a) / lookahead
    kappa = np.where(a_star > a, t_act * (0.5 + 1.5 * a), t_deact / (0.5 + 1.5 * a))
    return np.clip(a + kappa * rate, U_MIN, U_MAX)


class MCG:
    def __init__(self, idm, kp=100.0, kv=20.0, lookahead=0.01, knows_exo="current"):
        self.idm, self.kp, self.kv, self.la, self.mode = idm, kp, kv, lookahead, knows_exo

    def command(self, ref, th, thd, tau_a_now, tau_a_prev, md):
        th_r, thd_r, thdd_r = ref
        a_des_acc = thdd_r + self.kv * (thd_r - thd) + self.kp * (th_r - th)
        known = {"current": tau_a_now, "previous": tau_a_prev, "none": 0.0}[self.mode]
        tau_req = self.idm(th, thd, a_des_acc) - known                 # torque the muscles (active + passive) must supply
        c = md["d"] * md["cF"]                                         # N m per unit activation
        b = tau_req - float(np.sum(md["d"] * md["Fp"]))                # active torque needed after the passive contribution
        a_star, shortfall = solve_min_sumsq(c, b)
        u = inverse_activation(md["a"], a_star, self.la)
        return u, dict(tau_req=tau_req, shortfall=shortfall, a_star=a_star)


# =====================================================================================================  plants
class MockMusclePlant:
    """TEST DOUBLE. 1-DOF elbow, six muscles with constant moment arms (INVENTED), Thelen activation dynamics, F = Fmax * a."""
    D = np.array([-0.020, -0.020, -0.020, 0.030, 0.030, 0.025])        # INVENTED constant moment arms (m): test double only
    FMAX = np.array([798.52, 624.3, 624.3, 624.3, 435.56, 987.26])     # from arm26.osim

    def __init__(self, I, sigma, th0=0.0, a0=U_MIN):
        self.I, self.sigma = I, sigma
        self.x = np.concatenate([[th0, 0.0], np.full(6, a0)])
        self.u = np.full(6, U_MIN)
        self.tau_a = 0.0
        self.t = 0.0

    def _deriv(self, x, u, tau_a):
        a = x[2:]
        F = self.FMAX * a
        acc = (float(self.D @ F) + tau_a - self.sigma * g * math.sin(x[0])) / self.I
        kappa = np.where(u > a, T_ACT * (0.5 + 1.5 * a), T_DEACT / (0.5 + 1.5 * a))
        return np.concatenate([[x[1], acc], (u - a) / kappa])

    def measure(self):
        return self.t, self.x[0], self.x[1], self._deriv(self.x, self.u, self.tau_a)[1]

    def muscle_data(self):
        a = self.x[2:]
        return dict(d=self.D, cF=self.FMAX, Fp=np.zeros(6), a=a, F=self.FMAX * a)

    def apply(self, u, tau_a):
        self.u, self.tau_a = np.asarray(u, float), tau_a

    def advance(self, dt):
        n = max(1, int(math.ceil(dt / 5e-4)))
        h = dt / n
        for _ in range(n):
            k1 = self._deriv(self.x, self.u, self.tau_a)
            k2 = self._deriv(self.x + 0.5 * h * k1, self.u, self.tau_a)
            k3 = self._deriv(self.x + 0.5 * h * k2, self.u, self.tau_a)
            k4 = self._deriv(self.x + h * k3, self.u, self.tau_a)
            self.x = self.x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt

    def metabolic_rate(self):
        return float("nan")


class OpenSimMusclePlant:
    """arm26 with muscles. Every OpenSim call marked [verify] must be checked against your OpenSim version."""
    def __init__(self, model_path, payload_kg=None, accuracy=1e-6, th0=0.0, a0=U_MIN, metabolics=False):
        import opensim as osim
        self.osim = osim
        model = osim.Model(model_path)
        if payload_kg is not None:
            model.getBodySet().get("payload").setMass(float(payload_kg))                          # [verify]
        brain = osim.PrescribedController()
        brain.setName("python_inputs")
        for name in MUSCLES:                                                                       # control index 0..5
            brain.addActuator(osim.Muscle.safeDownCast(model.getForceSet().get(name)))            # [verify]
        brain.addActuator(osim.CoordinateActuator.safeDownCast(model.getForceSet().get("exo_elbow_torque")))   # index 6

        #Newly added

        actuators = model.getActuators()

        for j in range(7):
            actuator = actuators.get(j)
            actuator_name = actuator.getName()
            control_value = U_MIN if j < 6 else 0.0

            brain.prescribeControlForActuator(actuator_name, osim.Constant(control_value))
          # [verify] int-index overload
        model.addController(brain)
        self.probe = None
        if metabolics:
            try:                                                                                   # [verify] whole block
                probe = osim.Umberger2010MuscleMetabolicsProbe()
                probe.setName("metabolics")
                probe.set_activation_maintenance_rate_on(True)
                probe.set_shortening_rate_on(True)
                probe.set_basal_rate_on(False)
                probe.set_mechanical_work_rate_on(True)
                for name in MUSCLES:
                    probe.addMuscle(name, 0.5)                                                     # 0.5 slow-twitch fraction: ASSUMED
                model.addProbe(probe)
                self.probe = probe
            except Exception as e:
                print(f"  [WARN] metabolic probe disabled: {e}")
        self.model = model
        self.state = model.initSystem()
        self.muscles = [osim.Muscle.safeDownCast(model.getForceSet().get(n)) for n in MUSCLES]
        funcs = osim.PrescribedController.safeDownCast(model.getControllerSet().get(0)).get_ControlFunctions()   # [verify]
        self.f = [osim.Constant.safeDownCast(funcs.get(j)) for j in range(7)]
        self.elbow = model.getCoordinateSet().get("r_elbow_flex")
        self.elbow.setValue(self.state, float(th0))
        for m in self.muscles:
            m.setActivation(self.state, float(a0))                                                 # [verify]
        model.equilibrateMuscles(self.state)
        self.manager = osim.Manager(model)
        self.manager.setIntegratorAccuracy(accuracy)                                               # [verify]
        self.state.setTime(0.0)
        self.manager.initialize(self.state)
        self.t = 0.0

    def measure(self):
        self.model.realizeAcceleration(self.state)
        return (self.t, self.elbow.getValue(self.state), self.elbow.getSpeedValue(self.state),
                self.elbow.getAccelerationValue(self.state))

    def muscle_data(self):
        s, e = self.state, self.elbow
        md = dict(d=[], cF=[], Fp=[], a=[], F=[])
        for m in self.muscles:
            fmax, cphi = m.getMaxIsometricForce(), m.getCosPennationAngle(s)                      # [verify] getters below
            md["d"].append(m.computeMomentArm(s, e))
            md["cF"].append(fmax * cphi * m.getActiveForceLengthMultiplier(s) * m.getForceVelocityMultiplier(s))
            md["Fp"].append(fmax * cphi * m.getPassiveForceMultiplier(s))
            md["a"].append(m.getActivation(s))
            md["F"].append(m.getTendonForce(s))
        return {k: np.array(v, float) for k, v in md.items()}

    def apply(self, u, tau_a):
        for j in range(6):
            self.f[j].setValue(float(u[j]))
        self.f[6].setValue(float(tau_a))

    def advance(self, dt):
        self.state = self.manager.integrate(self.t + dt)                                           # [verify]
        self.t += dt

    def metabolic_rate(self):
        if self.probe is None:
            return float("nan")
        try:
            return float(self.probe.getProbeOutputs(self.state).get(0))                            # [verify]
        except Exception:
            return float("nan")


# =====================================================================================================  one run
def simulate(plant, mcg, controller, act, ref_fn, t_end, dt, ideal_exo=False, noise_acc=0.0, noise_tau=0.0,
             noise_th=0.0, delay=0.0, tau_h_gain_err=1.0, mu_true=None, seed=0):
    rng = np.random.default_rng(seed)
    n, nd = int(round(t_end / dt)), int(round(delay / dt))
    cols = ["t", "theta", "theta_dot", "theta_ddot", "theta_ref", "err", "tau_req", "tau_h_muscle", "tau_a", "tau_m", "sigma_hat",
            "act_flexor_mean", "act_extensor_mean", "sum_a2", "shortfall", "met_rate"]
    L = {c: np.zeros(n) for c in cols}
    tau_a_prev = 0.0
    buf = []
    for k in range(n):
        t, th, thd, thdd = plant.measure()
        md = plant.muscle_data()
        tau_h_meas = float(np.sum(md["d"] * md["F"]))                      # physiological torque: what the muscles actually produce
        ref = ref_fn(t)

        if k > 0:                                                          # AGC adaptation: consistent measured pair
            Z = np.array([thdd + noise_acc * rng.standard_normal(), thd, g * math.sin(th + noise_th * rng.standard_normal())])
            tau_t_m = tau_a_prev + tau_h_gain_err * tau_h_meas + noise_tau * rng.standard_normal()
            buf.append((Z, tau_t_m))
            Zd, taud = buf[max(0, len(buf) - 1 - nd)]
            controller.update(Zd, taud, dt)

        Y = g * math.sin(th)
        tau_a_des = max(controller.sigma_hat() * Y, 0.0)
        if ideal_exo:
            tau_m, tau_a = float("nan"), tau_a_des
        else:
            tau_m = act.command(tau_a_des, th, thd)
            tau_a = act.deliver(tau_m, th, thd, mu_true)

        u, info = mcg.command(ref, th, thd, tau_a, tau_a_prev, md)
        plant.apply(u, tau_a)
        row = (t, th, thd, thdd, ref[0], th - ref[0], info["tau_req"], tau_h_meas, tau_a, tau_m, controller.sigma_hat(),
               float(np.mean(md["a"][FLEXORS])), float(np.mean(md["a"][EXTENSORS])), float(np.sum(md["a"] ** 2)),
               info["shortfall"], plant.metabolic_rate())
        for c, v in zip(cols, row):
            L[c][k] = v
        plant.advance(dt)
        tau_a_prev = tau_a
    return L


# =====================================================================================================  OpenSim self-test
def selftest(a, idm, I_eff):
    print("\n== self-test of the OpenSim muscle plant ==")
    ok = True
    p = OpenSimMusclePlant(a.model, a.payload, a.accuracy)
    md = p.muscle_data()
    signs = [("extensor" if i in EXTENSORS else "flexor", md["d"][i]) for i in range(6)]
    good = all((s == "flexor") == (v > 0) for s, v in signs)
    ok &= good
    print("  moment arms at 0 deg:", ", ".join(f"{MUSCLES[i]} {md['d'][i]:+.3f}" for i in range(6)), "  ", "OK" if good else "SIGN PROBLEM")

    def accel(u, tau):
        q = OpenSimMusclePlant(a.model, a.payload, a.accuracy)
        q.apply(u, tau)
        return q.measure()[3]
    base = accel([U_MIN] * 6, 0.0)
    exo = accel([U_MIN] * 6, 2.0) - base
    flag = abs(exo - 2.0 / I_eff) < 0.03 * (2.0 / I_eff)
    ok &= flag
    print(f"  exo channel: 2 N m adds {exo:+.4f} rad/s^2, expected {2.0/I_eff:+.4f}   {'OK' if flag else 'MISMATCH'}")
    bic = accel([U_MIN, U_MIN, U_MIN, 0.5, U_MIN, U_MIN], 0.0) - base
    tri = accel([0.5, U_MIN, U_MIN, U_MIN, U_MIN, U_MIN], 0.0) - base
    flag = bic > 0 and tri < 0
    ok &= flag
    print(f"  muscle channels: BIClong 0.5 -> {bic:+.3f}, TRIlong 0.5 -> {tri:+.3f} rad/s^2 (expect + and -)   {'OK' if flag else 'CHECK ORDER/SIGN'}")
    p2 = OpenSimMusclePlant(a.model, a.payload, a.accuracy, th0=math.pi / 3)
    md2 = p2.muscle_data()
    model_tau = float(np.sum(md2["d"] * (md2["cF"] * md2["a"] + md2["Fp"])))
    true_tau = float(np.sum(md2["d"] * md2["F"]))
    flag = abs(model_tau - true_tau) < 0.15 * max(abs(true_tau), 0.5)
    ok &= flag
    print(f"  static-optimisation force model at 60 deg: {model_tau:+.3f} N m vs OpenSim tendon forces {true_tau:+.3f} N m   {'OK' if flag else 'LARGE DIFFERENCE (check getters / tendon compliance)'}")
    print("  SELF-TEST", "PASSED" if ok else "FAILED: fix before running comparisons")
    return ok


# =====================================================================================================  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", choices=["opensim", "mock"], default="opensim")
    ap.add_argument("--model", default=r"models\arm26_elbow_payload_2p5kg.osim")
    ap.add_argument("--payload", type=float, default=2.5)
    ap.add_argument("--nominal", type=float, default=0.0)
    ap.add_argument("--traj", choices=["cosine", "minjerk"], default="cosine")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--accuracy", type=float, default=1e-6)
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--kv", type=float, default=20.0)
    ap.add_argument("--lookahead", type=float, default=0.01, help="CMC-style look-ahead window (s)")
    ap.add_argument("--human-knows-exo", choices=["current", "previous", "none"], default="current")
    ap.add_argument("--gamma", type=float, default=0.025)
    ap.add_argument("--ideal-exo", action="store_true")
    ap.add_argument("--noise-acc", type=float, default=0.0)
    ap.add_argument("--noise-tau", type=float, default=0.0)
    ap.add_argument("--noise-th", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--tau-h-gain-err", type=float, default=1.0)
    ap.add_argument("--mu-true", type=float, default=None)
    ap.add_argument("--metabolics", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    if a.plant == "opensim":
        idm = OpenSimID(a.model, a.payload)
        I_eff, sigma_eff = idm.effective_params()
        sigma_nom = OpenSimID(a.model, a.nominal).effective_params()[1]
        make_plant = lambda: OpenSimMusclePlant(a.model, a.payload, a.accuracy, metabolics=a.metabolics)
    else:
        d = l_e + l_w
        I_eff, sigma_eff = I_FOREARM_ELBOW + a.payload * d**2, m_e * l_c + a.payload * d
        sigma_nom = m_e * l_c + a.nominal * d
        idm = AnalyticID(I_eff, sigma_eff)
        make_plant = lambda: MockMusclePlant(I_eff, sigma_eff)
        print("*** MOCK PLANT: invented moment arms, for testing the loop only. Not a result. ***")
    print(f"plant={a.plant} payload={a.payload} kg  I_eff={I_eff:.4f}  sigma_eff={sigma_eff:.4f}  (nominal GC sigma={sigma_nom:.4f})")
    if a.selftest:
        if a.plant != "opensim":
            raise SystemExit("--selftest needs --plant opensim")
        if not selftest(a, idm, I_eff):
            raise SystemExit(1)

    ref_fn = make_reference(a.traj, T=a.T)
    t_end = a.t_end if a.t_end is not None else (6.0 if a.traj == "cosine" else a.T + 1.0)
    act = CableActuator()
    cases = {"unassisted": Unassisted(), "GC oracle": GC(sigma_eff), f"GC nominal ({a.nominal} kg)": GC(sigma_nom),
             "AGC": AGC(gamma=a.gamma)}
    runs = {}
    for name, ctrl in cases.items():
        mcg = MCG(idm, a.kp, a.kv, a.lookahead, a.human_knows_exo)
        runs[name] = simulate(make_plant(), mcg, ctrl, act, ref_fn, t_end, a.dt, a.ideal_exo, a.noise_acc, a.noise_tau,
                              a.noise_th, a.delay, a.tau_h_gain_err, a.mu_true)

    b = runs["unassisted"]
    print(f"\n{'controller':<22}{'RMS tau_h':>10}{'red.':>7}{'RMS flexor a':>13}{'red.':>7}{'RMS extensor a':>15}{'RMS err(rad)':>13}{'infeasible':>11}{'RMS met(W)':>11}")
    for name, L in runs.items():
        r = rms(L["tau_h_muscle"]); rf = rms(L["act_flexor_mean"]); re = rms(L["act_extensor_mean"])
        red = lambda x, x0: 100 * (1 - x / x0)
        met = rms(L["met_rate"]) if np.isfinite(L["met_rate"]).any() else float("nan")
        print(f"{name:<22}{r:10.3f}{red(r, rms(b['tau_h_muscle'])):6.1f}%{rf:13.3f}{red(rf, rms(b['act_flexor_mean'])):6.1f}%"
              f"{re:15.3f}{rms(L['err']):13.5f}{100*np.mean(np.abs(L['shortfall'])>1e-3):10.1f}%{met:11.2f}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for name, L in runs.items():
            with open(os.path.join(a.out, name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(list(L.keys())); w.writerows(zip(*L.values()))
        open(os.path.join(a.out, "config.txt"), "w").write(repr(vars(a)))
        print("saved CSVs to", a.out)
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
        for name, L in runs.items():
            ax[1].plot(L["t"], L["tau_h_muscle"], label=name); ax[2].plot(L["t"], L["act_flexor_mean"], label=name)
            ax[3].plot(L["t"], L["act_extensor_mean"], label=name); ax[4].plot(L["t"], L["sigma_hat"], label=name)
        ax[0].plot(b["t"], b["theta_ref"], "k--", label="reference"); ax[0].plot(b["t"], b["theta"], label="unassisted")
        ax[0].plot(runs["AGC"]["t"], runs["AGC"]["theta"], label="AGC")
        for x, yl in zip(ax, ("theta (rad)", "muscle torque (N m)", "flexor activation", "extensor activation", "sigma_hat")):
            x.set_ylabel(yl); x.legend(fontsize=7); x.grid(alpha=0.3)
        ax[-1].set_xlabel("t (s)"); plt.tight_layout()
        plt.savefig(os.path.join(a.out or ".", "cosim_mcg.png"), dpi=130); print("saved plot")


if __name__ == "__main__":
    main()
