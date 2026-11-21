# Databricks notebook source
# MAGIC %md
# MAGIC # Avaliacao da aderencia do modelo as medicoes (metricas, recortes e hipoteses)
# MAGIC
# MAGIC Calcula, a partir da camada gold, as metricas da Secao 4.3 do TCC:
# MAGIC - **razao de somas** (soma medida / soma esperada), metrica principal, no
# MAGIC   mesmo formato do PR da IEC 61724-1, com a sensibilidade ao modelo de
# MAGIC   transposicao, ao modelo de temperatura e a fonte de irradiancia;
# MAGIC - **MBE e RMSE**, absolutos e normalizados pela media medida, em base
# MAGIC   horaria, diaria e mensal; **correlacao** com e sem as horas noturnas;
# MAGIC - **recortes** por hora do dia, estacao meteorologica (dez-fev verao,
# MAGIC   mar-mai outono, jun-ago inverno, set-nov primavera) e faixa de
# MAGIC   irradiancia no plano, com n e intervalo de confianca de 95% por
# MAGIC   bootstrap em blocos diarios (1.000 reamostragens);
# MAGIC - **validacao da irradiancia** do NASA POWER contra o INMET A612;
# MAGIC - **alinhamento sub-horario** em passos de 5 minutos;
# MAGIC - testes das hipoteses explicativas: *clipping*, temperatura (proxy x NOCT),
# MAGIC   sujidade (dias desde a ultima chuva) e disponibilidade diurna;
# MAGIC - verificacoes complementares citadas no texto (convencao de tempo das
# MAGIC   series, verificacao das fontes, efeito das decisoes de preparacao,
# MAGIC   quarentena diurna e noturna, estados do rastreador, PR e Atlas).
# MAGIC
# MAGIC Todas as tabelas de saida comecam com `avaliacao_` na gold e alimentam as
# MAGIC figuras do notebook 10 e os numeros do texto.

# COMMAND ----------

# MAGIC %pip install pvlib==0.11.2 --quiet

# COMMAND ----------

import json, numpy as np, pandas as pd, pvlib
GOLD = "fotovoltaica_ifes.gold"
rng = np.random.default_rng(20261120)
EST = {12: "verao", 1: "verao", 2: "verao", 3: "outono", 4: "outono", 5: "outono",
       6: "inverno", 7: "inverno", 8: "inverno", 9: "primavera", 10: "primavera", 11: "primavera"}
saida = {}

ch = spark.table(f"{GOLD}.fato_energia_canal_hora").toPandas()
ch["hora_local"] = pd.to_datetime(ch.hora_local)
eq = spark.table(f"{GOLD}.dim_equipamento").toPandas()[["id_equipamento", "estrutura"]]
ch = ch.merge(eq, on="id_equipamento")
inc = ch[ch.incluir_aderencia].copy()
inc["data"] = inc.hora_local.dt.normalize()
inc["hl"] = inc.hora_local.dt.hour
inc["estacao"] = inc.hora_local.dt.month.map(EST)


def razao_ic(df, col_esp="energia_esperada_wh", B=1000):
    """Razao de somas e IC 95% por bootstrap em blocos diarios."""
    d = df.groupby("data")[["energia_medida_wh", col_esp]].sum()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    m, e = d.energia_medida_wh.values, d[col_esp].values
    idx = rng.integers(0, len(d), (B, len(d)))
    boot = m[idx].sum(1) / e[idx].sum(1)
    return m.sum() / e.sum(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(df), len(d)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Razao de somas global e sensibilidade

# COMMAND ----------

variantes = {"perez_noct (principal)": "esperada_perez_noct_wh", "haydavies_noct": "esperada_haydavies_noct_wh",
             "isotropico_noct": "esperada_isotropico_noct_wh", "perez_proxy_microinversor": "esperada_perez_proxy_wh",
             "inmet_perez_noct": "esperada_inmet_noct_wh"}
linhas = []
for nome, c in variantes.items():
    for est_nome, sub in [("todas", inc), ("fixa", inc[inc.estrutura == "fixa"]), ("rastreador", inc[inc.estrutura == "rastreador"])]:
        s = sub[sub[c].notna()]
        r, lo, hi, n, nd = razao_ic(s, c)
        linhas.append((nome, est_nome, r, lo, hi, n, nd))
sens = pd.DataFrame(linhas, columns=["variante", "estrutura", "razao_somas", "ic95_inf", "ic95_sup", "n_canal_horas", "n_dias"])
spark.createDataFrame(sens).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_sensibilidade")
display(sens.round(4))
saida["sensibilidade"] = sens.round(4).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) MBE, RMSE e correlacao (base horaria, diaria e mensal, nivel usina)
# MAGIC Base horaria: horas da usina pareadas (`flag_hora_pareada_usina`, todos os
# MAGIC canais em operacao incluidos). Base diaria: dias em que todas as horas com
# MAGIC sol tiveram todos os canais em operacao incluidos. Base mensal: soma dos
# MAGIC canal-horas incluidos.

# COMMAND ----------

uh = spark.table(f"{GOLD}.gold_incidencia_usina_hora").toPandas()
uh["hora_local"] = pd.to_datetime(uh.hora_local)


def erros(m, e, base):
    d = m - e
    return {"base": base, "n": int(len(m)), "media_medida_wh": float(m.mean()), "mbe_wh": float(d.mean()),
            "rmse_wh": float(np.sqrt((d ** 2).mean())), "nmbe_pct": float(d.mean() / m.mean() * 100),
            "nrmse_pct": float(np.sqrt((d ** 2).mean()) / m.mean() * 100),
            "correlacao": float(np.corrcoef(m, e)[0, 1]), "razao_somas": float(m.sum() / e.sum())}

par = uh[uh.flag_hora_pareada_usina]
met = [erros(par.energia_medida_wh.values, par.energia_esperada_wh.values, "horaria (horas pareadas, irradiancia no plano >= 50)")]
# dia completo: nenhuma hora diurna (poa >= 50) com canal em operacao faltando
uh["data"] = uh.hora_local.dt.normalize()
diur = uh[(uh.poa_fixa_perez_wh_m2 >= 50)]
ok_dia = diur.groupby("data").apply(lambda g: bool((g.n_canais_incluidos == g.n_canais_em_operacao).all() and (g.n_canais_incluidos > 0).all()))
dd = uh[uh.data.isin(ok_dia[ok_dia].index)].groupby("data")[["energia_medida_wh", "energia_esperada_wh"]].sum()
met.append(erros(dd.energia_medida_wh.values, dd.energia_esperada_wh.values, "diaria (dias completos)"))
mm = spark.table(f"{GOLD}.gold_incidencia_usina_mes").toPandas().dropna(subset=["energia_medida_kwh"])
mm = mm[mm.canal_horas_incluidas >= 100]
met.append(erros(mm.energia_medida_kwh.values * 1000, mm.energia_esperada_kwh.values * 1000, "mensal (meses com >= 100 canal-horas)"))
# correlacoes hora a hora: com e sem a noite (potencia total x irradiancia)
pot = spark.sql(f"""SELECT date_trunc('hour', instante_coleta_local) hora_local, avg(potencia_w) * count(DISTINCT id_equipamento) p,
                           count(*) n FROM {GOLD}.fato_potencia_5min GROUP BY 1""").toPandas()
irr = spark.table(f"{GOLD}.fato_irradiancia_hora").toPandas()
pz = pot.merge(irr, on="hora_local").merge(spark.table(f"{GOLD}.fato_irradiancia_plano_hora").select("hora_local", "poa_fixa_perez_wh_m2", "zenite_solar_graus").toPandas(), on="hora_local")
corr = {"r_potencia_ghi_24h": float(pz[["p", "ghi_wh_m2"]].corr().iloc[0, 1]),
        "r_potencia_ghi_diurno_ghi_maior_50": float(pz[pz.ghi_wh_m2 > 50][["p", "ghi_wh_m2"]].corr().iloc[0, 1]),
        "r_potencia_poa_diurno_poa_maior_50": float(pz[pz.poa_fixa_perez_wh_m2 > 50][["p", "poa_fixa_perez_wh_m2"]].corr().iloc[0, 1]),
        "r_medida_esperada_horas_pareadas": met[0]["correlacao"], "n_horas_pareadas": met[0]["n"]}
erros_df = pd.DataFrame(met)
spark.createDataFrame(erros_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_erros")
display(erros_df.round(3))
saida["erros"] = erros_df.round(4).to_dict("records")
saida["correlacoes"] = {k: round(v, 4) for k, v in corr.items()}
print(saida["correlacoes"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Recortes com n e IC 95% (bootstrap por dia)

# COMMAND ----------

inc["faixa_irradiancia"] = pd.cut(inc.poa_wh_m2, [50, 200, 400, 600, 800, 2000],
                                  labels=["050-200", "200-400", "400-600", "600-800", "800+"]).astype(str)
rec = []
for recorte, col in [("hora_do_dia", "hl"), ("estacao", "estacao"), ("faixa_irradiancia_plano", "faixa_irradiancia")]:
    for est_nome, sub0 in [("todas", inc), ("fixa", inc[inc.estrutura == "fixa"])]:
        for k, sub in sub0.groupby(col):
            r, lo, hi, n, nd = razao_ic(sub)
            rec.append((recorte, est_nome, str(k), r, lo, hi, n, nd))
rec = pd.DataFrame(rec, columns=["recorte", "estrutura", "categoria", "razao_somas", "ic95_inf", "ic95_sup", "n_canal_horas", "n_dias"])
spark.createDataFrame(rec).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_recortes")
display(rec.round(3))
saida["recortes"] = rec.round(4).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Validacao da irradiancia: NASA POWER x INMET A612
# MAGIC Horas com as duas fontes acima de 50 Wh/m²; dias com todas as horas de sol
# MAGIC registradas no INMET; mensal pela media diaria desses dias. A
# MAGIC climatologia do Atlas (global horizontal) entra como terceira referencia.

# COMMAND ----------

v = irr.dropna(subset=["ghi_inmet_wh_m2"]).copy()
v["hora_local"] = pd.to_datetime(v.hora_local)
vh = v[(v.ghi_wh_m2 > 50) & (v.ghi_inmet_wh_m2 > 50)]
val = [erros(vh.ghi_inmet_wh_m2.values, vh.ghi_wh_m2.values, "NASA x INMET horaria (ambas > 50 Wh/m2)")]
ir2 = irr.copy(); ir2["hora_local"] = pd.to_datetime(ir2.hora_local); ir2["data"] = ir2.hora_local.dt.normalize()
sol = ir2[ir2.ghi_ceu_claro_wh_m2 > 0]
g = sol.groupby("data").agg(nasa=("ghi_wh_m2", "sum"), inmet=("ghi_inmet_wh_m2", "sum"), n=("ghi_wh_m2", "size"), ni=("ghi_inmet_wh_m2", "count"))
g = g[g.ni == g.n]
val.append(erros(g.inmet.values, g.nasa.values, "NASA x INMET diaria (dias completos no INMET)"))
val_df = pd.DataFrame(val)
gm = g.groupby(g.index.to_period("M")).agg(nasa=("nasa", "mean"), inmet=("inmet", "mean"), dias=("n", "size"))
gm["razao_nasa_inmet"] = gm.nasa / gm.inmet
atl = spark.sql(f"SELECT mes, valor_kwh_m2_dia FROM {GOLD}.ref_climatologia_mes WHERE fonte = 'atlas_labren_2017' AND camada = 'global_horizontal'").toPandas().set_index("mes")
gm["atlas_kwh_m2_dia"] = [atl.loc[p.month, "valor_kwh_m2_dia"] for p in gm.index]
gm = gm.reset_index().rename(columns={"data": "mes_referencia"})
gm["mes_referencia"] = gm.mes_referencia.astype(str)
gm[["nasa", "inmet"]] = gm[["nasa", "inmet"]] / 1000
spark.createDataFrame(gm).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_nasa_inmet_mes")
spark.createDataFrame(val_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_nasa_inmet")
display(val_df.round(3)); display(gm.round(3))
saida["nasa_inmet"] = val_df.round(4).to_dict("records")
saida["nasa_inmet_mes"] = gm.round(3).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Alinhamento temporal
# MAGIC (a) Deslocamento da serie de 5 min da usina em passos de 5 min (-30 a +30
# MAGIC min), reagregada por hora e comparada com a energia esperada horaria (que
# MAGIC segue a convencao do NASA POWER, inicio da hora): se houvesse descompasso
# MAGIC sub-horario, o erro seria minimo fora do deslocamento zero. (b) Em dias
# MAGIC claros, centroide da geracao dos modulos fixos comparado com o meio-dia
# MAGIC solar calculado; e inicio/fim da geracao comparados com nascer/por do sol.

# COMMAND ----------

p5 = spark.sql(f"""SELECT f.id_equipamento, f.instante_coleta_local t, f.potencia_w
                   FROM {GOLD}.fato_potencia_5min f JOIN {GOLD}.dim_equipamento e USING (id_equipamento)
                   WHERE e.estrutura = 'fixa'""").toPandas()
p5["t"] = pd.to_datetime(p5.t)
esp_fix = ch[(ch.estrutura == "fixa") & ch.incluir_aderencia][["id_equipamento", "hora_local", "energia_esperada_wh"]]
ali = []
for k in range(-6, 7):
    q = p5.assign(h=(p5.t + pd.Timedelta(minutes=5 * k)).dt.floor("h"))
    hh = q.groupby(["id_equipamento", "h"]).agg(n=("potencia_w", "size"), m=("potencia_w", "mean")).reset_index()
    hh = hh[hh.n == 12].merge(esp_fix, left_on=["id_equipamento", "h"], right_on=["id_equipamento", "hora_local"])
    d = hh.m - hh.energia_esperada_wh
    hh["hl"] = hh.h.dt.hour
    manha = hh[hh.hl < 10]; tarde = hh[hh.hl >= 14]
    ali.append((5 * k, float(np.sqrt((d ** 2).mean())), float(np.corrcoef(hh.m, hh.energia_esperada_wh)[0, 1]),
                float(manha.m.sum() / manha.energia_esperada_wh.sum()), float(tarde.m.sum() / tarde.energia_esperada_wh.sum()), len(hh)))
ali = pd.DataFrame(ali, columns=["deslocamento_min", "rmse_wh", "correlacao", "razao_manha_ate_10h", "razao_tarde_desde_14h", "n"])
spark.createDataFrame(ali).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_alinhamento")
display(ali.round(4))
saida["alinhamento"] = ali.round(4).to_dict("records")

U = spark.table(f"{GOLD}.dim_usina").toPandas().iloc[0]
loc = pvlib.location.Location(U.latitude, U.longitude, "Etc/GMT+3", U.altitude_m)
dia_claro = ir2.groupby("data").apply(lambda x: x.ghi_wh_m2.sum() / max(x.ghi_ceu_claro_wh_m2.sum(), 1))
claros = set(dia_claro[dia_claro >= 0.75].index)
fx = p5.groupby("t").potencia_w.sum()
cen = []
for d, s in fx.groupby(fx.index.normalize()):
    if d not in claros or s.sum() < 2000:
        continue
    tt = pd.date_range(d, periods=1440, freq="min", tz="Etc/GMT+3")
    el = loc.get_solarposition(tt).apparent_elevation
    meio_dia = el.idxmax(); nasc = el[el > 0].index.min(); por = el[el > 0].index.max()
    h = s.index.hour + s.index.minute / 60
    cen_h = (h * s.clip(lower=0)).sum() / s.clip(lower=0).sum()
    ger = s[s > 5]
    cen.append((d.date(), (cen_h - (meio_dia.hour + meio_dia.minute / 60)) * 60,
                (ger.index.min().hour * 60 + ger.index.min().minute) - (nasc.hour * 60 + nasc.minute),
                (por.hour * 60 + por.minute) - (ger.index.max().hour * 60 + ger.index.max().minute)))
cen = pd.DataFrame(cen, columns=["data", "centroide_menos_meio_dia_solar_min", "inicio_geracao_menos_nascer_min", "por_do_sol_menos_fim_geracao_min"])
spark.createDataFrame(cen).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_centroide_dias_claros")
saida["centroide"] = {"dias_claros": len(cen), **cen.drop(columns="data").median().round(1).to_dict()}
print(saida["centroide"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Hipoteses explicativas
# MAGIC - **Clipping**: proporcao de leituras no teto de saida do canal.
# MAGIC - **Temperatura**: temperatura de celula (NOCT, com a T2M) x temperatura
# MAGIC   reportada pelo microinversor, por estacao, nas horas com sol.
# MAGIC - **Sujidade**: razao diaria (fixos) em funcao dos dias desde a ultima
# MAGIC   chuva >= 5 mm no INMET; se a sujeira pesasse, a razao cairia com os dias secos.

# COMMAND ----------

clip = spark.sql(f"""SELECT count_if(flag_potencia_no_teto) no_teto, count_if(potencia_w > 0) com_geracao,
                     count_if(flag_potencia_no_teto) / count_if(potencia_w > 0) AS fracao
                     FROM {GOLD}.fato_potencia_5min WHERE NOT flag_canal_sem_modulo""").toPandas().iloc[0].to_dict()
saida["clipping"] = {k: float(v) for k, v in clip.items()}

tt = inc[inc.temperatura_microinversor_c.notna()]
temp = tt.groupby("estacao").agg(t_microinversor=("temperatura_microinversor_c", "mean"), t_celula_noct=("temperatura_celula_noct_c", "mean"),
                                 t_ar=("temperatura_ar_c", "mean"), n=("hl", "size"))
temp.loc["todas"] = [tt.temperatura_microinversor_c.mean(), tt.temperatura_celula_noct_c.mean(), tt.temperatura_ar_c.mean(), len(tt)]
temp["r_proxy_noct"] = np.nan
temp.loc["todas", "r_proxy_noct"] = tt[["temperatura_microinversor_c", "temperatura_celula_noct_c"]].corr().iloc[0, 1]
temp = temp.reset_index()
spark.createDataFrame(temp).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_temperatura")
display(temp.round(2))
saida["temperatura"] = temp.round(3).to_dict("records")

chuva = ir2.groupby("data").precipitacao_inmet_mm.sum(min_count=1)
fxd = inc[inc.estrutura == "fixa"].groupby("data")[["energia_medida_wh", "energia_esperada_wh"]].sum()
fxd = fxd[fxd.energia_esperada_wh > 1000]
ultima = None; dias_sec = {}
for d in pd.date_range(chuva.index.min(), chuva.index.max(), freq="D"):
    if chuva.get(d, np.nan) >= 5:
        ultima = d
    dias_sec[d] = (d - ultima).days if ultima is not None else np.nan
fxd["dias_desde_chuva"] = [dias_sec.get(d, np.nan) for d in fxd.index]
fxd = fxd.dropna()
fxd["classe"] = pd.cut(fxd.dias_desde_chuva, [-1, 2, 7, 14, 1000], labels=["0-2", "3-7", "8-14", "15+"]).astype(str)
suj = fxd.groupby("classe").apply(lambda g: pd.Series({"razao_somas": g.energia_medida_wh.sum() / g.energia_esperada_wh.sum(), "n_dias": len(g)})).reset_index()
suj["r_spearman_razao_diaria_x_dias_secos"] = float(pd.Series(fxd.energia_medida_wh / fxd.energia_esperada_wh).corr(fxd.dias_desde_chuva, method="spearman"))
spark.createDataFrame(suj).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_sujidade")
display(suj.round(3))
saida["sujidade"] = suj.round(4).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Completude e disponibilidade restritas ao periodo diurno

# COMMAND ----------

disp = spark.sql(f"""
WITH sol AS (SELECT hora_local FROM {GOLD}.fato_irradiancia_plano_hora WHERE zenite_solar_graus < 90),
dias AS (SELECT DISTINCT date(instante_coleta_local) d FROM {GOLD}.fato_potencia_5min),
periodo AS (SELECT count(*) AS horas_sol FROM sol WHERE date(hora_local) BETWEEN (SELECT min(d) FROM dias) AND (SELECT max(d) FROM dias))
SELECT f.id_microinversor, count(*) AS leituras_diurnas, (SELECT horas_sol FROM periodo) * 12 AS leituras_diurnas_possiveis,
       count(*) / ((SELECT horas_sol FROM periodo) * 12) AS disponibilidade_diurna
FROM {GOLD}.fato_potencia_5min f JOIN sol ON date_trunc('hour', f.instante_coleta_local) = sol.hora_local
WHERE f.canal = 'A' GROUP BY 1 ORDER BY disponibilidade_diurna DESC
""").toPandas()
spark.createDataFrame(disp).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_disponibilidade_diurna")
display(disp)
saida["disponibilidade_diurna"] = disp.round(4).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Verificacoes complementares citadas no texto
# MAGIC Grava em `avaliacao_verificacoes` (grupo, indicador, valor, n, definicao) os
# MAGIC numeros do texto que nao estao nas tabelas anteriores, cada um com a sua
# MAGIC definicao: convencao de tempo das series horarias, verificacao preliminar das
# MAGIC fontes com o canal A, correlacoes entre potencia e irradiacao, efeito das
# MAGIC decisoes de preparacao sobre a razao, quarentena diurna e noturna, estados do
# MAGIC rastreador por periodo, razao de desempenho (PR) e coerencia entre o Atlas e a
# MAGIC climatologia do NASA POWER.

# COMMAND ----------

V = []
def reg(grupo, indicador, valor, n, definicao):
    V.append({"grupo": grupo, "indicador": indicador, "valor": float(valor), "n": int(n), "definicao": definicao})

# 8.1 Convencao de tempo: NASA POWER (ceu claro) e INMET (dias claros) contra o ceu claro de Ineichen (pvlib)
U = spark.table(f"{GOLD}.dim_usina").toPandas().iloc[0]
ir = spark.table(f"{GOLD}.fato_irradiancia_hora").select("hora_utc", "hora_local", "ghi_ceu_claro_wh_m2", "ghi_inmet_wh_m2").toPandas()
ir["hora_utc"] = pd.to_datetime(ir.hora_utc, utc=True)
ir["hora_local"] = pd.to_datetime(ir.hora_local)
loc = pvlib.location.Location(U.latitude, U.longitude, altitude=U.altitude_m)
t1 = pd.date_range(ir.hora_utc.min() - pd.Timedelta("3h"), ir.hora_utc.max() + pd.Timedelta("3h"), freq="1min", tz="UTC")
ineichen = loc.get_clearsky(t1, model="ineichen")["ghi"].resample("1h", label="left", closed="left").mean()  # media em [h, h+1)
conv = {"inicio": 0, "fim": 1}  # rotulo h = inicio [h, h+1) ou fim [h-1, h) do intervalo
ns = ir.set_index("hora_utc").ghi_ceu_claro_wh_m2
for nome, d in conv.items():
    ref = ineichen.shift(d).reindex(ns.index)
    ok = ns.notna() & ref.notna() & ((ns > 0) | (ref > 0))
    reg("convencao_tempo", f"nasa_ceu_claro_rotulo_{nome}_r", np.corrcoef(ns[ok], ref[ok])[0, 1], ok.sum(),
        f"Correlacao horaria entre a GHI de ceu claro do NASA POWER (UTC) e a media de Ineichen (pvlib) supondo o rotulo como {nome} do intervalo")
    reg("convencao_tempo", f"nasa_ceu_claro_rotulo_{nome}_rmse_wh", np.sqrt(((ns[ok] - ref[ok]) ** 2).mean()), ok.sum(), "RMSE horario da mesma comparacao (Wh/m2)")
# INMET: a gold guarda a serie ja rotulada pelo inicio da hora; o rotulo original da estacao e o da gold + 1 h
inm = ir.set_index("hora_utc").ghi_inmet_wh_m2
inm_orig = pd.Series(inm.values, index=inm.index + pd.Timedelta("1h"))
dia_loc = ir.set_index("hora_utc").hora_local.dt.date
ref0 = ineichen.reindex(inm.index)
tab = pd.DataFrame({"g": inm, "c": ref0, "dia": dia_loc})
sol_h = tab[tab.c > 0]
dd = sol_h.groupby("dia").agg(g=("g", "sum"), c=("c", "sum"), falta=("g", lambda s: s.isna().sum()))
claros = dd[(dd.falta == 0) & (dd.g >= 0.75 * dd.c)].index
for nome, d in conv.items():
    ref = ineichen.shift(d).reindex(inm_orig.index)
    m = pd.Series(inm_orig.index.tz_convert("America/Sao_Paulo").date, index=inm_orig.index).isin(claros) & inm_orig.notna() & ((inm_orig > 0) | (ref > 0))
    reg("convencao_tempo", f"inmet_dias_claros_rotulo_original_{nome}_r", np.corrcoef(inm_orig[m], ref[m])[0, 1], len(claros),
        f"Correlacao horaria entre a radiacao global do INMET A612 com o rotulo original e Ineichen, supondo o rotulo como {nome} do intervalo; n = dias claros (todas as horas com sol com dado e soma diaria >= 75% de Ineichen)")

# 8.2 Verificacao preliminar das fontes: energia do canal A x GHI do NASA POWER, sem exigir horas completas
pa = spark.sql(f"""SELECT date_trunc('hour', instante_coleta_local) hora_local, sum(potencia_w) * 5 / 60 e_wh, count(*) n
                   FROM {GOLD}.fato_potencia_5min WHERE canal = 'A' GROUP BY 1""").toPandas()
pa["hora_local"] = pd.to_datetime(pa.hora_local)
gh = spark.table(f"{GOLD}.fato_irradiancia_hora").select("hora_local", "ghi_wh_m2").toPandas(); gh["hora_local"] = pd.to_datetime(gh.hora_local)
zn = spark.table(f"{GOLD}.fato_irradiancia_plano_hora").select("hora_local", "zenite_solar_graus").toPandas(); zn["hora_local"] = pd.to_datetime(zn.hora_local)
for nome, desl in [("utc-3", 0), ("utc-2", -1), ("utc-4", 1)]:
    g2 = gh.assign(hora_local=gh.hora_local + pd.Timedelta(hours=desl))
    x = pa.merge(g2, on="hora_local"); x = x[x.ghi_wh_m2 > 0]
    reg("verificacao_fontes", f"r_horaria_canal_a_ghi_{nome}", np.corrcoef(x.e_wh, x.ghi_wh_m2)[0, 1], len(x),
        f"Correlacao horaria entre a energia do canal A dos 4 microinversores (leituras disponiveis, sem exigir hora completa) e a GHI do NASA POWER, horas com GHI > 0, medicoes tratadas como {nome}")
x = pa.merge(gh, on="hora_local").merge(zn, on="hora_local"); x["dia"] = x.hora_local.dt.date
pos = zn[zn.zenite_solar_graus < 90].assign(dia=zn.hora_local.dt.date).groupby("dia").size() * 12 * 4
gd = gh.assign(dia=gh.hora_local.dt.date).groupby("dia").ghi_wh_m2.sum()
xd = x[x.zenite_solar_graus < 90].groupby("dia").agg(e_wh=("e_wh", "sum"), n=("n", "sum")).join(gd).join(pos.rename("pos"))
xd = xd[xd.n >= 0.9 * xd.pos]
reg("verificacao_fontes", "r_diaria_canal_a_ghi", np.corrcoef(xd.e_wh, xd.ghi_wh_m2)[0, 1], len(xd),
    "Correlacao diaria entre a energia do canal A e a GHI diaria do NASA POWER, dias com pelo menos 90% das leituras diurnas do canal A")

# 8.3 Correlacoes entre potencia total e irradiacao (Secao 4.3.3)
for k, v in corr.items():
    if k.startswith("r_potencia"):
        reg("correlacoes", k, v, 0, "Correlacao horaria entre a potencia total (media das leituras x numero de equipamentos) e a irradiacao; ver secao 1")

# 8.4 Efeito das decisoes de preparacao sobre a razao de somas
hc = ch[ch.hora_completa].copy()
ia = hc.incluir_aderencia
esp_ghi = hc.energia_esperada_wh * hc.ghi_wh_m2 / hc.poa_wh_m2.where(hc.poa_wh_m2 > 0)
canal_a = hc.id_equipamento.str.endswith("-A")
rast = hc.id_equipamento.str.startswith("403000261021")
sem_trat = (hc.poa_wh_m2 >= 50) & ~hc.canal_sem_geracao_no_dia
reg("decisoes_preparacao", "razao_modelo_principal", hc.energia_medida_wh[ia].sum() / hc.energia_esperada_wh[ia].sum(), ia.sum(), "Razao de somas nas canal-horas incluidas")
reg("decisoes_preparacao", "razao_so_canal_a_medido", hc.energia_medida_wh[ia & canal_a].sum() / hc.energia_esperada_wh[ia].sum(), ia.sum(), "Energia medida so do canal A; esperada de todos os modulos")
reg("decisoes_preparacao", "razao_ghi_no_lugar_da_poa", hc.energia_medida_wh[ia].sum() / esp_ghi[ia].sum(), ia.sum(), "Esperada reescalada por GHI/POA (temperatura de celula mantida)")
reg("decisoes_preparacao", "razao_rastreador_principal", hc.energia_medida_wh[ia & rast].sum() / hc.energia_esperada_wh[ia & rast].sum(), (ia & rast).sum(), "Canais do rastreador, modelo principal")
reg("decisoes_preparacao", "razao_rastreador_sem_excluir_travado", hc.energia_medida_wh[sem_trat & rast].sum() / hc.energia_esperada_wh[sem_trat & rast].sum(), (sem_trat & rast).sum(),
    "Canais do rastreador incluindo os dias travados ou indeterminados (hora completa, POA >= 50, canal gerando)")

# 8.5 Quarentena diurna e noturna (zenite no meio da hora; horas fora da tabela horaria contam como noite)
qd = spark.sql(f"""SELECT CASE WHEN z.zenite_solar_graus < 90 THEN 'dia' ELSE 'noite' END periodo, count(*) n
                  FROM fotovoltaica_ifes.silver.usina_geracao_quarentena q
                  LEFT JOIN {GOLD}.fato_irradiancia_plano_hora z ON date_trunc('hour', q.instante_coleta_local) = z.hora_local
                  WHERE q.motivo_quarentena = 'equipamento_sem_leitura' GROUP BY 1""").toPandas().set_index("periodo").n
for p_, n_ in qd.items():
    reg("quarentena", f"equipamento_sem_leitura_{p_}", n_, n_, "Linhas em quarentena por falta de potencia, por periodo do dia")

# 8.6 Estados do rastreador classificados (sem heranca), por periodo
er = spark.table(f"{GOLD}.estado_rastreador_dia").toPandas(); er = er[~er.estado_inferido]; er["data"] = pd.to_datetime(er.data)
def conta(est, ini=None, fim=None):
    s = er[er.estado == est]
    if ini: s = s[(s.data >= ini) & (s.data <= fim)]
    return len(s)
for est, ini, fim in [("parado_referencia", None, None), ("parado_referencia", "2020-11-01", "2021-02-28"),
                      ("rastreando", None, None), ("rastreando", "2021-03-01", "2021-08-31"),
                      ("travado_oeste", None, None), ("travado_oeste", "2021-12-01", "2022-02-28"), ("travado_oeste", "2021-05-01", "2021-07-31"),
                      ("rastreando", "2021-09-01", "2021-11-30"), ("parado_referencia", "2021-09-01", "2021-11-30"), ("travado_oeste", "2021-09-01", "2021-11-30")]:
    n_ = conta(est, ini, fim)
    reg("rastreador", f"{est}_{ini or 'todo'}_{fim or 'periodo'}", n_, n_, "Dias classificados no estado (sem dias herdados)")

# 8.7 Razao de desempenho (IEC 61724-1) nas canal-horas incluidas
de = spark.table(f"{GOLD}.dim_equipamento").select("id_equipamento", "potencia_nominal_modulo_wp", "coef_temp_potencia_pct_c").toPandas()
pr = inc.merge(de, on="id_equipamento")
pr["yr"] = pr.poa_wh_m2 / 1000 * pr.potencia_nominal_modulo_wp
pr["yr_t"] = pr.yr * (1 + pr.coef_temp_potencia_pct_c / 100 * (pr.temperatura_celula_noct_c - 25))
for nome, s in [("todas", pr), ("fixa", pr[pr.estrutura == "fixa"]), ("rastreador", pr[pr.estrutura == "rastreador"])]:
    reg("pr", f"pr_{nome}", s.energia_medida_wh.sum() / s.yr.sum(), len(s), "Energia medida / (H_POA / 1000 x Wp)")
    reg("pr", f"pr_corrigida_temperatura_{nome}", s.energia_medida_wh.sum() / s.yr_t.sum(), len(s), "PR com Y_r corrigido pela temperatura NOCT")

# 8.8 Atlas (LABREN) x climatologia do NASA POWER, GHI mensal
cl = spark.table(f"{GOLD}.ref_climatologia_mes").toPandas()
a = cl[(cl.fonte == "atlas_labren_2017") & (cl.camada == "global_horizontal")].set_index("mes").valor_kwh_m2_dia
nclim = cl[(cl.fonte == "nasa_power_2001_2020") & (cl.camada == "global_horizontal")].set_index("mes").valor_kwh_m2_dia
reg("atlas", "dif_mensal_maxima_pct", ((a / nclim - 1).abs() * 100).max(), 12, "Maior diferenca mensal absoluta entre Atlas e climatologia NASA (%)")
reg("atlas", "dif_media_anual_pct", (a.mean() / nclim.mean() - 1) * 100, 12, "Diferenca entre as medias anuais (%)")

ver = pd.DataFrame(V)
spark.createDataFrame(ver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_verificacoes")
display(ver)
saida["verificacoes"] = ver.round(4).to_dict("records")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(saida, default=str, ensure_ascii=False))
