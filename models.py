"""Deterministic reaction + electrolyzer models. No LLM in here: every number must be reproducible.

Species order everywhere: [CO2, H2, CH4, H2O, CO]
R1: CO2 + 4H2 -> CH4 + 2H2O      (Sabatier)
R2: CO2 +  H2 -> CO  +  H2O      (RWGS, the selectivity killer)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

R = 8.314462618            # J/mol/K
FARADAY = 96485.33212      # C/mol
M_H2, M_CH4 = 2.01588, 16.0425   # g/mol
DH1, DH2 = -165.0e3, +41.2e3     # J/mol, 298 K standard enthalpies (R1, R2)
HHV_H2 = 39.4              # kWh/kg (same basis as the deck, slide 10)

# Koschany, Schlereth & Hinrichs (2016) Appl. Catal. B 181, 504 -- Ni/Al(O)x, CO2 methanation.
# LHHW rate law. VERIFY every number against Table 3 of the paper before you cite it.
KIN_DEFAULT = dict(k_ref=3.46e-4, Ea=77.5e3, T_ref=555.0,
                   K_OH=0.5, dH_OH=22.4e3,
                   K_H2=0.44, dH_H2=-6.2e3,
                   K_mix=0.88, dH_mix=-10.0e3)


# ---------------------------------------------------------------- thermodynamics
def K_sabatier(T: float) -> float:
    """Equilibrium constant of R1 [bar^-2] (Koschany 2016 correlation)."""
    return 137.0 * T ** -3.998 * np.exp(158.7e3 / (R * T))


def K_rwgs(T: float) -> float:
    """Equilibrium constant of R2 = 1/K_WGS, K_WGS = exp(4400/T - 4.036) (Xu & Froment 1989)."""
    return float(np.exp(-(4400.0 / T - 4.036)))


def _pp(x1, x2, ratio, P):
    n = np.array([1 - x1 - x2, ratio - 4 * x1 - x2, x1, 2 * x1 + x2, x2])
    return n, n / n.sum() * P


def equilibrium(T_C: float, P: float, ratio: float = 4.0) -> dict:
    """Two-reaction equilibrium (ideal gas) per 1 mol CO2 fed. Returns extents, X_CO2, S_CH4, yields."""
    T = T_C + 273.15
    K1, K2 = K_sabatier(T), K_rwgs(T)

    def x2_of(x1):
        hi = min(1 - x1, ratio - 4 * x1) * (1 - 1e-10)
        lo = 1e-14

        def g(x2):
            _, p = _pp(x1, x2, ratio, P)
            return np.log(p[4] * p[3]) - np.log(p[0] * p[1]) - np.log(K2)
        if hi <= lo or g(lo) >= 0:
            return lo
        if g(hi) <= 0:
            return hi
        return brentq(g, lo, hi, xtol=1e-14)

    def h(x1):
        _, p = _pp(x1, x2_of(x1), ratio, P)
        return np.log(p[2] * p[3] ** 2) - np.log(K1 * p[0] * p[1] ** 4)

    xmax = min(1.0, ratio / 4) * (1 - 1e-9)
    lo = 1e-12
    x1 = lo if h(lo) >= 0 else (xmax if h(xmax) <= 0 else brentq(h, lo, xmax, xtol=1e-14))
    x2 = x2_of(x1)
    n, _ = _pp(x1, x2, ratio, P)
    y = dict(zip(["CO2", "H2", "CH4", "H2O", "CO"], n / n.sum()))
    return dict(T_C=T_C, P=P, ratio=ratio, x1=x1, x2=x2, X_CO2=x1 + x2,
                S_CH4=x1 / (x1 + x2) if (x1 + x2) > 0 else np.nan, y_wet=y,
                heat_kJ_per_mol_CO2=(x1 * DH1 + x2 * DH2) / 1e3)


def selectivity_grid(Ts, Ps, ratios) -> pd.DataFrame:
    rows = []
    for r in ratios:
        for P in Ps:
            for T in Ts:
                e = equilibrium(T, P, r)
                rows.append(dict(T_C=T, P_bar=P, ratio=r, X_CO2=e["X_CO2"], S_CH4=e["S_CH4"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- kinetics
def rate(T: float, p, kin: dict, K1: float) -> float:
    """Koschany LHHW rate, mol/(g_cat s). p = [CO2, H2, CH4, H2O] partial pressures in bar."""
    pCO2, pH2, pCH4, pH2O = [max(v, 1e-12) for v in p]
    f = lambda K, dH: K * np.exp(-dH / R * (1 / T - 1 / kin["T_ref"]))
    k = kin["k_ref"] * np.exp(-kin["Ea"] / R * (1 / T - 1 / kin["T_ref"]))
    den = (1 + f(kin["K_OH"], kin["dH_OH"]) * pH2O / np.sqrt(pH2)
           + f(kin["K_H2"], kin["dH_H2"]) * np.sqrt(pH2)
           + f(kin["K_mix"], kin["dH_mix"]) * np.sqrt(pCO2))
    beta = pCH4 * pH2O ** 2 / (K1 * pCO2 * pH2 ** 4)
    return k * np.sqrt(pH2 * pCO2) * (1 - beta) / den ** 2


def pfr(T_C, P, ratio, F_CO2, W_max, kin, n=200):
    """Isothermal, isobaric PFR (CO2 methanation only). F_CO2 mol/s, W in g. Returns W, X_CO2."""
    T = T_C + 273.15
    K1 = K_sabatier(T)
    xcap = min(1.0, ratio / 4) - 1e-9

    def f(W, y):
        x = min(max(y[0], 0.0), xcap)
        n_ = np.array([1 - x, ratio - 4 * x, x, 2 * x])
        return [rate(T, n_ / (1 + ratio - 2 * x) * P, kin, K1) / F_CO2]
    Ws = np.linspace(0, W_max, n)
    sol = solve_ivp(f, (0, W_max), [0.0], t_eval=Ws, method="LSODA", rtol=1e-8, atol=1e-12)
    return Ws, sol.y[0]


def arrhenius_fit(T_lo_C, T_hi_C, P, ratio, kin):
    """Effective k = A exp(-Ea/RT) [mol/(g s bar)] at the *feed* composition (far from equilibrium),
    i.e. r = k * pH2^0.5 * pCO2^0.5. Use this for DWSIM's Arrhenius reaction type."""
    Ts = np.linspace(T_lo_C, T_hi_C, 15) + 273.15
    pf = np.array([1.0, ratio, 0, 0]) / (1 + ratio) * P
    k = np.array([rate(T, pf, kin, K_sabatier(T)) / np.sqrt(pf[0] * pf[1]) for T in Ts])
    slope, intercept = np.polyfit(1 / Ts, np.log(k), 1)
    return dict(A=float(np.exp(intercept)), Ea_kJ_mol=float(-slope * R / 1e3),
                T_range_C=[T_lo_C, T_hi_C], P_bar=P, ratio=ratio)


# ---------------------------------------------------------------- electrolyzer
# PLACEHOLDER parameter sets: order-of-magnitude typical values, NOT sourced. Replace + cite.
ELEC_PRESETS = {
    "PEM (~80 C)":      dict(T=353.15, V_rev=1.18, j0=1e-4, alpha=0.5, asr=0.10, f1=250.0, f2=0.99),
    "Alkaline (~80 C)": dict(T=353.15, V_rev=1.18, j0=1e-4, alpha=0.5, asr=0.25, f1=150.0, f2=0.985),
    "SOEC (~800 C)":    dict(T=1073.15, V_rev=1.00, j0=0.2, alpha=0.5, asr=0.25, f1=1.0, f2=1.0),
}


def electrolyzer(j, p: dict) -> pd.DataFrame:
    """Polarization V(j) = V_rev + Tafel(asinh) + j*ASR ;  Faradaic eff (Ulleberg): f2*j^2/(f1+j^2), j in mA/cm2.
    j in A/cm2. Energy per kg H2 = 2F/M * V / eta_F."""
    j = np.maximum(np.asarray(j, float), 1e-3)
    eta_act = R * p["T"] / (p["alpha"] * FARADAY) * np.arcsinh(j / (2 * p["j0"]))
    V = p["V_rev"] + eta_act + p["asr"] * j
    jm = j * 1e3
    eta_F = p["f2"] * jm ** 2 / (p["f1"] + jm ** 2)
    E = 2 * FARADAY / (M_H2 * 1e-3) / 3.6e6 * V / eta_F          # kWh/kg H2
    return pd.DataFrame(dict(j=j, V=V, eta_F=eta_F, kWh_per_kg_H2=E,
                             eta_HHV=HHV_H2 / E, kWh_per_kg_CH4=E * 4 * M_H2 / M_CH4))