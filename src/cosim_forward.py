#!/usr/bin/env python3
"""
Architecture 1: step-wise FORWARD-DYNAMICS co-simulation, Python <-> OpenSim.

  plant   : OpenSim torque-driven arm (arm26_torque_driven_2p5kg.osim), integrated by osim.Manager, one control period at a time
            (or --plant analytic: the same 1-DOF plant in numpy, to test the loop without OpenSim)
  human   : H2a  computed-torque feedback human who knows the full dynamics
              a_des = th_r'' + kv*(th_r' - th') + kp*(th_r - th)
              tau_h = ID(th, th', a_des) - tau_a_known           (same structure as the muscle command generator of Sambhav et al.)
  exosuit : ideal elbow torque tau_a (CoordinateActuator "exo_elbow_torque"), from GC or AGC, optionally through the cable model G(th, th')
  AGC     : gradient adaptation of Theta = [I, b, sigma], exact discretisation, fed with MEASURED signals (noise, delay optional)

Files needed in the same folder: offline_pipeline.py (controllers + cable model), arm26_torque_driven_2p5kg.osim (+ Geometry/ for display)

Examples
  python cosim_forward.py --plant analytic --payload 2.5                  # tested
  python cosim_forward.py --plant opensim  --payload 2.5 --selftest       # check plumbing first   (untested here: see [verify])
  python cosim_forward.py --plant opensim  --payload 2.5 --nominal 0 --out results/run1 --plot

Units: rad, rad/s, N m, kg, s.  + = flexion.   theta = 0: forearm hanging.
"""
import argparse
import csv
import math
import os

import numpy as np

from offline_pipeline import GC, AGC, Unassisted, CableActuator, m_e, l_c, l_e, l_w, I_FOREARM_ELBOW, g


# =====================================================================================================  reference motion
def make_reference(kind, T=1.0, amp=math.pi / 2, f=0.5):
    if kind == "cosine":                              # 0..amp..0, frequency f (my choice, range/period of Fig. 6a of the AGC paper)
        w = 2 * math.pi * f
        return lambda t: (0.5 * amp * (1 - math.cos(w * t)), 0.5 * amp * w * math.sin(w * t), 0.5 * amp * w * w * math.cos(w * t))
    if kind == "minjerk":                             # 0 -> amp in T seconds, then hold (earlier platform paper: 1 s fast, 2 s slow)
        def ref(t):
            s = min(max(t / T, 0.0), 1.0)
            th = amp * (10 * s**3 - 15 * s**4 + 6 * s**5)
            if t >= T:
                return th, 0.0, 0.0
            return th, amp / T * (30 * s**2 - 60 * s**3 + 30 * s**4), amp / T**2 * (60 * s - 180 * s**2 + 120 * s**3)
        return ref
    raise ValueError(kind)


# =====================================================================================================  plants
class AnalyticPlant:
    """I th'' + b th' + sigma g sin(th) = tau_h + tau_a   (RK4 with sub-steps, zero-order hold on the torques)."""
    def __init__(self, I, sigma, b=0.0, th0=0.0, thd0=0.0):
        self.I, self.sigma, self.b = I, sigma, b
        self.x = np.array([th0, thd0], float)
        self.t = 0.0
        self.tau = 0.0

    def _acc(self, x, tau):
        return (tau - self.b * x[1] - self.sigma * g * math.sin(x[0])) / self.I

    def measure(self):
        return self.t, self.x[0], self.x[1], self._acc(self.x, self.tau)

    def apply(self, tau_h, tau_a):
        self.tau = tau_h + tau_a

    def advance(self, dt):
        n = max(1, int(math.ceil(dt / 1e-3)))
        h = dt / n
        f = lambda x: np.array([x[1], self._acc(x, self.tau)])
        for _ in range(n):
            k1 = f(self.x); k2 = f(self.x + 0.5 * h * k1); k3 = f(self.x + 0.5 * h * k2); k4 = f(self.x + h * k3)
            self.x = self.x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt


class AnalyticID:
    def __init__(self, I, sigma, b=0.0):
        self.I, self.sigma, self.b = I, sigma, b

    def __call__(self, th, thd, thdd):
        return self.I * thdd + self.b * thd + self.sigma * g * math.sin(th)


class OpenSimPlant:
    """OpenSim torque-driven plant. Every OpenSim call marked [verify] should be checked against your version."""
    def __init__(self, model_path, payload_kg=None, accuracy=1e-6, th0=0.0, thd0=0.0):
        import opensim as osim
        self.osim = osim
        model = osim.Model(model_path)
        if payload_kg is not None:
            model.getBodySet().get("payload").setMass(float(payload_kg))                      # [verify] Body.setMass
        brain = osim.PrescribedController()
        brain.setName("python_inputs")
        for name in ("human_elbow_torque", "exo_elbow_torque"):                                 # order = control index 0, 1
            brain.addActuator(osim.CoordinateActuator.safeDownCast(model.getForceSet().get(name)))   # [verify]
        brain.prescribeControlForActuator(0, osim.Constant(0.0))                                # [verify] int-index overload
        brain.prescribeControlForActuator(1, osim.Constant(0.0))
        model.addController(brain)
        self.model = model
        self.state = model.initSystem()
        funcs = osim.PrescribedController.safeDownCast(model.getControllerSet().get(0)).get_ControlFunctions()   # [verify]
        self.f_h = osim.Constant.safeDownCast(funcs.get(0))
        self.f_x = osim.Constant.safeDownCast(funcs.get(1))
        self.elbow = model.getCoordinateSet().get("r_elbow_flex")
        self.elbow.setValue(self.state, float(th0))
        self.elbow.setSpeedValue(self.state, float(thd0))
        self.manager = osim.Manager(model)
        self.manager.setIntegratorAccuracy(accuracy)                                            # [verify]
        self.state.setTime(0.0)
        self.manager.initialize(self.state)
        self.t = 0.0

    def measure(self):
        self.model.realizeAcceleration(self.state)
        return (self.t, self.elbow.getValue(self.state), self.elbow.getSpeedValue(self.state),
                self.elbow.getAccelerationValue(self.state))

    def apply(self, tau_h, tau_a):
        self.f_h.setValue(float(tau_h))
        self.f_x.setValue(float(tau_a))

    def advance(self, dt):
        self.state = self.manager.integrate(self.t + dt)                                         # [verify] returns the State
        self.t += dt


class OpenSimID:
    """Human's internal model and truth source for sigma_eff and I_eff: OpenSim inverse dynamics on a separate model instance."""
    def __init__(self, model_path, payload_kg=None):
        import opensim as osim
        self.osim = osim
        self.model = osim.Model(model_path)
        if payload_kg is not None:
            self.model.getBodySet().get("payload").setMass(float(payload_kg))                    # [verify]
        self.state = self.model.initSystem()
        for i in range(self.model.getForceSet().getSize()):
            self.model.updForceSet().get(i).setAppliesForce(self.state, False)                  # [verify]
        self.solver = osim.InverseDynamicsSolver(self.model)
        self.elbow = self.model.getCoordinateSet().get("r_elbow_flex")
        self.iq = self.elbow.getMobilizerQIndex()                                                # [verify]

    def __call__(self, th, thd, thdd):
        self.elbow.setValue(self.state, th)
        self.elbow.setSpeedValue(self.state, thd)
        self.model.realizeVelocity(self.state)
        udot = self.osim.Vector(self.state.getNU(), 0.0)
        udot.set(self.iq, thdd)
        return self.solver.solve(self.state, udot).get(self.iq)

    def effective_params(self):
        I = self(0.0, 0.0, 1.0)                           # at theta = 0 gravity torque vanishes -> pure inertia
        sigma = self(math.pi / 2, 0.0, 0.0) / g           # static hold torque at 90 deg = sigma_eff * g
        return I, sigma


# =====================================================================================================  human model H2a
class HumanH2a:
    """Computed-torque feedback human. knows_exo: 'current' (compensates the exosuit command of this very tick; the
    assumption of the earlier platform paper), 'previous' (one control period late: more realistic), 'none'."""
    def __init__(self, idm, kp=400.0, kv=40.0, knows_exo="current"):
        self.idm, self.kp, self.kv, self.mode = idm, kp, kv, knows_exo

    def command(self, ref, th, thd, tau_a_now, tau_a_prev):
        th_r, thd_r, thdd_r = ref
        a_des = thdd_r + self.kv * (thd_r - thd) + self.kp * (th_r - th)
        known = {"current": tau_a_now, "previous": tau_a_prev, "none": 0.0}[self.mode]
        return self.idm(th, thd, a_des) - known


# =====================================================================================================  one run
def simulate(plant, human, controller, act, ref_fn, t_end, dt, ideal_exo=False, noise_acc=0.0, noise_tau=0.0,
             noise_th=0.0, delay=0.0, tau_h_gain_err=1.0, mu_true=None, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(t_end / dt))
    nd = int(round(delay / dt))
    cols = ["t", "theta", "theta_dot", "theta_ddot", "theta_ref", "err", "tau_h", "tau_a", "tau_m", "sigma_hat", "tau_t_meas"]
    L = {c: np.zeros(n) for c in cols}
    tau_h_prev = tau_a_prev = 0.0
    buf = []
    for k in range(n):
        t, th, thd, thdd = plant.measure()          # thdd is produced by the torques still applied from the previous period
        ref = ref_fn(t)

        # --- adaptation: consistent pair (Z at t_k, torque applied during the previous period), corrupted like a sensor would be
        tau_t_m = np.nan
        if k > 0:
            th_m = th + noise_th * rng.standard_normal()
            Z = np.array([thdd + noise_acc * rng.standard_normal(), thd, g * math.sin(th_m)])
            tau_t_m = tau_a_prev + tau_h_gain_err * tau_h_prev + noise_tau * rng.standard_normal()
            buf.append((Z, tau_t_m))
            Zd, taud = buf[max(0, len(buf) - 1 - nd)]
            controller.update(Zd, taud, dt)

        # --- exosuit command
        Y = g * math.sin(th)
        tau_a_des = max(controller.sigma_hat() * Y, 0.0)
        if ideal_exo:
            tau_m, tau_a = float("nan"), tau_a_des
        else:
            tau_m = act.command(tau_a_des, th, thd)
            tau_a = act.deliver(tau_m, th, thd, mu_true)

        # --- human command, then apply both and advance the plant one control period (zero-order hold)
        tau_h = human.command(ref, th, thd, tau_a, tau_a_prev)
        plant.apply(tau_h, tau_a)
        for c, v in zip(cols, (t, th, thd, thdd, ref[0], th - ref[0], tau_h, tau_a, tau_m, controller.sigma_hat(), tau_t_m)):
            L[c][k] = v
        plant.advance(dt)
        tau_h_prev, tau_a_prev = tau_h, tau_a
    return L


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2)))


# =====================================================================================================  self-test (OpenSim plumbing)
def selftest(args, idm, I_eff, sigma_eff):
    print("\n== self-test of the OpenSim channels ==")
    ok = True
    p = OpenSimPlant(args.model, args.payload, args.accuracy)
    for name, (h, x) in (("human channel", (2.0, 0.0)), ("exo channel", (0.0, 2.0))):
        p = OpenSimPlant(args.model, args.payload, args.accuracy)
        p.apply(h, x)
        a = p.measure()[3]
        exp = 2.0 / I_eff
        flag = abs(a - exp) < 0.02 * abs(exp)
        ok &= flag
        print(f"  {name}: 2 N m at theta=0 -> acceleration {a:+.4f} rad/s^2, expected {exp:+.4f}   {'OK' if flag else 'MISMATCH'}")
    p = OpenSimPlant(args.model, args.payload, args.accuracy, th0=math.pi / 2)
    a = p.measure()[3]
    exp = -sigma_eff * g / I_eff
    flag = abs(a - exp) < 0.02 * abs(exp)
    ok &= flag
    print(f"  gravity: free arm at 90 deg -> acceleration {a:+.4f} rad/s^2, expected {exp:+.4f}   {'OK' if flag else 'MISMATCH'}")
    p = OpenSimPlant(args.model, args.payload, args.accuracy)
    p.apply(2.0, 0.0)
    for _ in range(int(0.2 / 0.005)):
        p.advance(0.005)
    t, th, thd, _ = p.measure()
    exp_thd = 2.0 / I_eff * 0.2
    flag = abs(thd - exp_thd) < 0.03 * exp_thd
    ok &= flag
    print(f"  integration: after 0.2 s of 2 N m, velocity {thd:.4f} rad/s, expected about {exp_thd:.4f}   {'OK' if flag else 'CHECK'}")
    print("  SELF-TEST", "PASSED" if ok else "FAILED: fix the plumbing before running comparisons")
    return ok


# =====================================================================================================  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", choices=["opensim", "analytic"], default="analytic")
    ap.add_argument("--model", default="arm26_torque_driven_2p5kg.osim")
    ap.add_argument("--payload", type=float, default=2.5, help="true payload (kg)")
    ap.add_argument("--nominal", type=float, default=0.0, help="payload the nominal GC was tuned for (kg)")
    ap.add_argument("--traj", choices=["cosine", "minjerk"], default="cosine")
    ap.add_argument("--T", type=float, default=1.0, help="minjerk duration (s); 1 = fast, 2 = slow in the earlier paper")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.005, help="control period (s)")
    ap.add_argument("--accuracy", type=float, default=1e-6)
    ap.add_argument("--kp", type=float, default=400.0)
    ap.add_argument("--kv", type=float, default=40.0)
    ap.add_argument("--human-knows-exo", choices=["current", "previous", "none"], default="current")
    ap.add_argument("--gamma", type=float, default=0.025)
    ap.add_argument("--ideal-exo", action="store_true", help="bypass the cable model: tau_a = sigma_hat*Y exactly")
    ap.add_argument("--noise-acc", type=float, default=0.0)
    ap.add_argument("--noise-tau", type=float, default=0.0)
    ap.add_argument("--noise-th", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--tau-h-gain-err", type=float, default=1.0)
    ap.add_argument("--mu-true", type=float, default=None, help="true friction coefficient if different from the model's 0.07")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    # ---- truth model parameters and the human's internal model
    if a.plant == "opensim":
        idm = OpenSimID(a.model, a.payload)
        I_eff, sigma_eff = idm.effective_params()
        make_plant = lambda: OpenSimPlant(a.model, a.payload, a.accuracy)
        idm_nom = OpenSimID(a.model, a.nominal)
        sigma_nom = idm_nom.effective_params()[1]
    else:
        d = l_e + l_w
        I_eff, sigma_eff = I_FOREARM_ELBOW + a.payload * d**2, m_e * l_c + a.payload * d
        sigma_nom = m_e * l_c + a.nominal * d
        idm = AnalyticID(I_eff, sigma_eff)
        make_plant = lambda: AnalyticPlant(I_eff, sigma_eff)
    print(f"plant={a.plant}  payload={a.payload} kg  I_eff={I_eff:.4f} kg m^2  sigma_eff={sigma_eff:.4f} kg m  "
          f"(nominal GC sigma={sigma_nom:.4f})")
    if a.selftest:
        if a.plant != "opensim":
            raise SystemExit("--selftest needs --plant opensim")
        if not selftest(a, idm, I_eff, sigma_eff):
            raise SystemExit(1)

    ref_fn = make_reference(a.traj, T=a.T)
    t_end = a.t_end if a.t_end is not None else (6.0 if a.traj == "cosine" else a.T + 1.0)
    act = CableActuator()
    cases = {"unassisted": Unassisted(), "GC true": GC(sigma_eff), f"GC nominal ({a.nominal} kg)": GC(sigma_nom),
             "AGC": AGC(gamma=a.gamma)}
    runs = {}
    for name, ctrl in cases.items():
        human = HumanH2a(idm, a.kp, a.kv, a.human_knows_exo)
        runs[name] = simulate(make_plant(), human, ctrl, act, ref_fn, t_end, a.dt, a.ideal_exo, a.noise_acc, a.noise_tau,
                              a.noise_th, a.delay, a.tau_h_gain_err, a.mu_true)

    base = rms(runs["unassisted"]["tau_h"])
    print(f"\n{'controller':<24}{'RMS tau_h':>10}{'reduction':>11}{'peak tau_h':>12}{'min tau_h':>11}{'RMS err (rad)':>15}{'max |err|':>11}{'sigma_hat end':>15}")
    for name, L in runs.items():
        r = rms(L["tau_h"])
        print(f"{name:<24}{r:10.3f}{100*(1-r/base):10.1f}%{np.max(L['tau_h']):12.2f}{np.min(L['tau_h']):11.2f}"
              f"{rms(L['err']):15.5f}{np.max(np.abs(L['err'])):11.5f}{L['sigma_hat'][-1]:15.3f}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for name, L in runs.items():
            fn = os.path.join(a.out, name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv")
            with open(fn, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(L.keys()))
                w.writerows(zip(*L.values()))
        with open(os.path.join(a.out, "config.txt"), "w") as f:
            f.write(repr(vars(a)))
        print("saved CSVs to", a.out)
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        for name, L in runs.items():
            ax[1].plot(L["t"], L["tau_h"], label=name)
            ax[2].plot(L["t"], L["tau_a"], label=name)
            ax[3].plot(L["t"], L["sigma_hat"], label=name)
        L0 = runs["unassisted"]
        ax[0].plot(L0["t"], L0["theta_ref"], "k--", label="reference")
        ax[0].plot(L0["t"], L0["theta"], label="unassisted")
        ax[0].plot(runs["AGC"]["t"], runs["AGC"]["theta"], label="AGC")
        ax[3].axhline(sigma_eff, color="k", ls=":")
        for x, yl in zip(ax, ("theta (rad)", "tau_h (N m)", "tau_a (N m)", "sigma_hat")):
            x.set_ylabel(yl); x.legend(fontsize=7); x.grid(alpha=0.3)
        ax[-1].set_xlabel("t (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out or ".", "cosim_forward.png"), dpi=130)
        print("saved plot")


if __name__ == "__main__":
    main()
