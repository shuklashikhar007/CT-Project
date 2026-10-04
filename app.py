import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models as m
import llm as L

st.set_page_config(page_title="Sabatier Reaction Dashboard", layout="wide")
st.title("Mars ISRU: Reaction Engineering Dashboard")
st.caption("Physics = deterministic models. LLM = literature extraction (grounded) + analyst. Nothing the LLM says feeds the models.")

DECK_T = [200, 250, 300, 350, 400, 450, 500, 550, 600]
DECK_S = [58, 82, 95, 98, 97, 90, 74, 58, 45]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Operating point")
    T = st.slider("Reactor T (C)", 200, 600, 350, 5)
    P = st.slider("Pressure (bar)", 1.0, 30.0, 5.0, 0.5)
    ratio = st.slider("H2:CO2 feed ratio", 3.0, 5.0, 4.0, 0.05)
    F_CO2 = st.number_input("CO2 feed (mol/s)", 0.001, 100.0, 0.2778, format="%.4f",
                            help="0.2778 mol/s = 1 kmol/h CO2 (= 4 kmol/h H2, the DWSIM POC basis)")
    W = st.number_input("Catalyst mass (g)", 1.0, 1e6, 1500.0, 100.0)

    with st.expander("Rate-law parameters (Koschany 2016 defaults)"):
        st.warning("Defaults typed from memory of the paper. Check each against its table, then tick 'verified'.")
        kin = dict(m.KIN_DEFAULT)
        c1, c2 = st.columns(2)
        kin["k_ref"] = c1.number_input("k_ref mol/(bar g s)", value=kin["k_ref"], format="%.3e")
        kin["T_ref"] = c2.number_input("T_ref (K)", value=kin["T_ref"])
        kin["Ea"] = c1.number_input("Ea (kJ/mol)", value=kin["Ea"] / 1e3) * 1e3
        for key, lab in [("K_OH", "K_OH"), ("K_H2", "K_H2"), ("K_mix", "K_mix")]:
            kin[key] = c1.number_input(f"{lab} (bar^-0.5)", value=kin[key], format="%.3f")
            kin["dH_" + key[2:]] = c2.number_input(f"dH_{key[2:]} (kJ/mol)", value=kin["dH_" + key[2:]] / 1e3) * 1e3
        kin_src = st.text_input("Source (paper, table)", "Koschany et al. 2016, Appl. Catal. B 181, 504")
        kin_ok = st.checkbox("Verified against source")

    st.header("LLM")
    prov = st.selectbox("Provider", list(L.PROVIDERS))
    model_name = st.text_input("Model", L.PROVIDERS[prov][2])
    api_key = st.text_input("API key", type="password", help="Or set the env var before launching.")

    def llm():
        return L.get_llm(prov, model_name, api_key or None)

tabs = st.tabs(["1 Reaction", "2 Selectivity envelope", "3 Electrolyzer", "4 Literature to parameters", "5 Hand-off + analyst"])

eq = m.equilibrium(T, P, ratio)
Ws, Xk = m.pfr(T, P, ratio, F_CO2, W, kin)

# ------------------------------------------------------------------ tab 1
with tabs[0]:
    a, b, c, d, e = st.columns(5)
    a.metric("Equilibrium X(CO2)", f"{eq['X_CO2']:.3f}")
    b.metric("Equilibrium S(CH4)", f"{eq['S_CH4']:.4f}")
    c.metric("Kinetic X(CO2) @ W", f"{Xk[-1]:.3f}")
    d.metric("CH4 out (kg/day)", f"{F_CO2 * Xk[-1] * eq['S_CH4'] * m.M_CH4 * 86400 / 1e3:.1f}")
    duty = F_CO2 * Xk[-1] * (eq["x1"] * m.DH1 + eq["x2"] * m.DH2) / (eq["x1"] + eq["x2"]) / 1e3
    e.metric("Reactor duty (kW)", f"{duty:.1f}", help="Using 298 K standard enthalpies (-165 / +41.2 kJ/mol); +-10% T-correction.")

    col1, col2 = st.columns(2)
    f1 = go.Figure()
    f1.add_scatter(x=Ws, y=Xk, name="Kinetic PFR (CO2 methanation)")
    f1.add_hline(y=eq["X_CO2"], line_dash="dash", annotation_text="equilibrium")
    f1.update_layout(title=f"Conversion vs catalyst mass ({T} C, {P} bar)", xaxis_title="Catalyst (g)",
                     yaxis_title="X(CO2)", yaxis_range=[0, 1.02], height=380)
    col1.plotly_chart(f1, width="stretch")

    Tg = np.arange(200, 605, 10)
    eqs = [m.equilibrium(t, P, ratio) for t in Tg]
    f2 = go.Figure()
    f2.add_scatter(x=Tg, y=[x["S_CH4"] * 100 for x in eqs], name="Equilibrium selectivity S(CH4) %")
    f2.add_scatter(x=Tg, y=[x["X_CO2"] * 100 for x in eqs], name="Equilibrium conversion X(CO2) %")
    f2.add_scatter(x=DECK_T, y=DECK_S, mode="markers+lines", name="Slide 5 curve (illustrative)",
                   line=dict(dash="dot", color="grey"))
    f2.add_hline(y=95, line_dash="dash", line_color="red")
    f2.update_layout(title=f"Selectivity vs T at {P} bar, H2:CO2 = {ratio}", xaxis_title="T (C)", yaxis_title="%", height=380)
    col2.plotly_chart(f2, width="stretch")
    st.caption("Slide 5's low-T drop (58% at 200 C) is a *rate* limit. It lowers conversion, not selectivity. "
               "Thermodynamically, S(CH4) stays ~100% until RWGS takes over above ~450-500 C.")
    st.dataframe(pd.Series(eq["y_wet"], name="equilibrium outlet mole fraction (wet)").to_frame().T, hide_index=True)

# ------------------------------------------------------------------ tab 2
with tabs[1]:
    st.subheader("Does S(CH4) > threshold hold across the intended envelope?")
    c1, c2, c3 = st.columns(3)
    Tr = c1.slider("T range (C)", 200, 600, (300, 400), 10)
    Pr = c2.slider("P range (bar)", 1.0, 30.0, (1.0, 10.0), 0.5)
    Rr = c3.slider("H2:CO2 range", 3.0, 5.0, (3.5, 4.5), 0.1)
    thr = st.number_input("Selectivity threshold (%)", 50.0, 100.0, 95.0)

    @st.cache_data(show_spinner=False)
    def grid(Tr, Pr, Rr):
        Ts = np.linspace(Tr[0], Tr[1], 9)
        Ps = np.unique(np.round(np.geomspace(Pr[0], Pr[1], 6), 2)) if Pr[1] > Pr[0] else [Pr[0]]
        Rs = sorted({Rr[0], 4.0, Rr[1]}) if Rr[0] <= 4.0 <= Rr[1] else [Rr[0], Rr[1]]
        return m.selectivity_grid(Ts, Ps, Rs)
    G = grid(Tr, Pr, Rr)
    G["S_pct"] = G.S_CH4 * 100
    ok = (G.S_pct >= thr).mean()
    worst = G.loc[G.S_pct.idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.metric("Envelope points passing", f"{ok * 100:.0f}%")
    k2.metric("Worst-case S(CH4)", f"{worst.S_pct:.2f}%")
    k3.metric("Worst at", f"{worst.T_C:.0f} C, {worst.P_bar} bar, {worst.ratio}")
    (st.success if ok == 1 else st.error)(
        f"Equilibrium selectivity {'holds' if ok == 1 else 'FAILS'} across the envelope. ")
    rsel = st.selectbox("Show H2:CO2 =", sorted(G.ratio.unique()), index=sorted(G.ratio.unique()).index(4.0) if 4.0 in set(G.ratio) else 0)
    H = G[G.ratio == rsel].pivot(index="P_bar", columns="T_C", values="S_pct")
    hm = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index.astype(str), colorscale="RdYlGn", zmin=min(80, H.values.min()), zmax=100,
                              text=np.round(H.values, 2), texttemplate="%{text}", colorbar_title="S %"))
    hm.update_layout(xaxis_title="T (C)", yaxis_title="P (bar)", height=380)
    st.plotly_chart(hm, width="stretch")

# ------------------------------------------------------------------ tab 3
with tabs[2]:
    st.warning("Preset values are typical-magnitude placeholders, NOT sourced. Replace them with numbers from a cited paper (tab 4).")
    pname = st.selectbox("Electrolyzer type", list(m.ELEC_PRESETS))
    p0 = m.ELEC_PRESETS[pname]
    cols = st.columns(7)
    ep = dict(T=cols[0].number_input("T (K)", value=p0["T"]),
              V_rev=cols[1].number_input("V_rev (V)", value=p0["V_rev"]),
              j0=cols[2].number_input("j0 (A/cm2)", value=p0["j0"], format="%.2e"),
              alpha=cols[3].number_input("alpha", value=p0["alpha"]),
              asr=cols[4].number_input("ASR (ohm cm2)", value=p0["asr"]),
              f1=cols[5].number_input("f1 (mA2/cm4)", value=p0["f1"]),
              f2=cols[6].number_input("f2", value=p0["f2"]))
    el_src = st.text_input("Electrolyzer source (paper, table)", "")
    el_ok = st.checkbox("Electrolyzer parameters verified against source")
    jop = st.slider("Operating current density j (A/cm2)", 0.05, 3.0, 1.0, 0.05)
    cm = st.number_input("CH4 production target (kg/day)", 1.0, 1e5, 625.0)
    jg = np.linspace(0.02, 3.0, 150)
    D = m.electrolyzer(jg, ep)
    op = m.electrolyzer([jop], ep).iloc[0]
    q = st.columns(5)
    q[0].metric("Cell voltage", f"{op.V:.3f} V")
    q[1].metric("Faradaic eff.", f"{op.eta_F * 100:.1f}%")
    q[2].metric("kWh/kg H2", f"{op.kWh_per_kg_H2:.1f}", help="Deck uses 60.6 (= 39.4 / 0.65)")
    q[3].metric("Efficiency (HHV)", f"{op.eta_HHV * 100:.0f}%", help="Deck assumes 65%. SOEC >100% is expected: steam heat is not counted as electricity.")
    q[4].metric("Plant power", f"{op.kWh_per_kg_CH4 * cm / 24:.0f} kW", help="Electrolysis only; deck: 772 kW")
    cA, cB = st.columns(2)
    fa = go.Figure(); fa.add_scatter(x=D.j, y=D.V, name="V(j)")
    fa.add_vline(x=jop, line_dash="dot"); fa.update_layout(title="Polarization curve", xaxis_title="j (A/cm2)", yaxis_title="V", height=350)
    cA.plotly_chart(fa, width="stretch")
    fb = go.Figure()
    fb.add_scatter(x=D.j, y=D.eta_HHV * 100, name="HHV efficiency %")
    fb.add_scatter(x=D.j, y=D.eta_F * 100, name="Faradaic efficiency %")
    fb.add_hline(y=65, line_dash="dash", annotation_text="deck: 65%")
    fb.update_layout(title="Efficiency curves", xaxis_title="j (A/cm2)", yaxis_title="%", height=350)
    cB.plotly_chart(fb, width="stretch")

# ------------------------------------------------------------------ tab 4
with tabs[3]:
    st.subheader("Paste or upload a paper; extract candidate parameters")
    st.caption("Each row must carry a verbatim quote. 'grounded' = quote found in your text AND contains the number. "
               "Ungrounded rows are likely hallucinated: ignore them. Always eyeball grounded ones too (units, catalyst, T range).")
    target = st.selectbox("Extract", list(L.TARGETS))
    up = st.file_uploader("PDF (optional)", type="pdf")
    txt = ""
    if up:
        from pypdf import PdfReader
        txt = "\n".join((pg.extract_text() or "") for pg in PdfReader(up).pages)
    txt = st.text_area("Paper text", txt, height=220)
    if st.button("Extract parameters", disabled=not txt.strip()):
        try:
            with st.spinner("Asking the model..."):
                st.session_state["extracted"] = L.extract_params(llm(), txt, target)
        except Exception as ex:
            st.error(f"LLM call failed: {ex}")
    if "extracted" in st.session_state:
        df = st.session_state["extracted"]
        st.dataframe(df, width="stretch", hide_index=True)
        st.info("Copy verified numbers into the sidebar (kinetics) or tab 3 (electrolyzer), set the source, tick 'verified'.")

# ------------------------------------------------------------------ tab 5
with tabs[4]:
    fit = m.arrhenius_fit(Tr[0], Tr[1], P, ratio, kin)
    op = m.electrolyzer([jop], ep).iloc[0]
    pkg = dict(
        status=dict(kinetics_verified=kin_ok, kinetics_source=kin_src,
                    electrolyzer_verified=el_ok, electrolyzer_source=el_src or "NOT SET"),
        reactor=dict(catalyst="Ni/Al2O3", T_C=T, P_bar=P, H2_CO2=ratio,
                     equilibrium=dict(X_CO2=eq["X_CO2"], S_CH4=eq["S_CH4"], extent_R1=eq["x1"], extent_R2=eq["x2"],
                                      outlet_mole_frac_wet=eq["y_wet"]),
                     kinetic_X_CO2=float(Xk[-1]), catalyst_g=W, F_CO2_mol_s=F_CO2,
                     rate_law="Koschany LHHW, r=k*sqrt(pH2*pCO2)*(1-beta)/DEN^2 [mol/(g s), bar]", rate_params=kin,
                     arrhenius_fit_for_DWSIM=fit,
                     dwsim_conversion_reactor_hint=dict(conversion_CO2_R1=eq["x1"], conversion_CO2_R2=eq["x2"],
                                                        note="Prefer RGIBBS; use these only for RCONV")),
        selectivity_envelope=dict(T_C=list(Tr), P_bar=list(Pr), H2_CO2=list(Rr), threshold_pct=thr,
                                  fraction_passing=float(ok), worst_case_pct=float(worst.S_pct)),
        electrolyzer=dict(type=pname, params=ep, j_A_cm2=jop, V=float(op.V), eta_F=float(op.eta_F),
                          kWh_per_kg_H2=float(op.kWh_per_kg_H2), eta_HHV=float(op.eta_HHV)),
    )
    if not (kin_ok and el_ok):
        st.error("Not ready to hand off: parameters not marked verified. Simulation Lead should treat this as DRAFT.")
    st.download_button("Download hand-off JSON", json.dumps(pkg, indent=2, default=float), "reaction_handoff.json", "application/json")
    with st.expander("Preview"):
        st.json(json.loads(json.dumps(pkg, default=float)))
    st.subheader("Analyst")
    qn = st.text_input("Ask about these results", "Does the >95% selectivity claim hold, and what is the weakest assumption?")
    if st.button("Ask"):
        try:
            with st.spinner("Thinking..."):
                st.markdown(L.explain(llm(), json.loads(json.dumps(pkg, default=float)), qn))
        except Exception as ex:
            st.error(f"LLM call failed: {ex}")