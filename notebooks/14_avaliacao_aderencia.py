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
# MAGIC   sujidade (dias desde a ultima chuva) e disponibilidade diurna.
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

dbutils.notebook.exit(json.dumps(saida, default=str, ensure_ascii=False))
