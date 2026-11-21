# Databricks notebook source
# MAGIC %md
# MAGIC # Camada gold - modelo de energia esperada e tabelas finais
# MAGIC
# MAGIC Modelo da Equacao 3.1 do TCC,
# MAGIC
# MAGIC `E_PV = E_sol x A x eta_PV x eta_PCU x [1 + PTC x (T_sup - T_std) / 100]`,
# MAGIC
# MAGIC com tres escolhas de entrada:
# MAGIC 1. **`E_sol` e a irradiancia no plano dos modulos**, nao a global
# MAGIC    horizontal. A GHI, a DNI e a DHI horarias do NASA POWER sao transpostas
# MAGIC    para o plano de cada estrutura (fixa: 15 graus voltada ao norte;
# MAGIC    rastreador de 1 eixo) pelo modelo de Perez, com Hay-Davies e isotropico
# MAGIC    como sensibilidade. A posicao solar e calculada no meio de cada hora.
# MAGIC 2. **O calculo e feito por canal**, com a area, a eficiencia, o coeficiente
# MAGIC    de temperatura e a NOCT do modulo daquele canal (`dim_equipamento`), e
# MAGIC    so nas horas em que o canal registrou as 12 leituras de 5 minutos
# MAGIC    (intervalos pareados). Canais sem modulo ficam de fora, assim como os
# MAGIC    dias em que um canal com modulo nao gerou nada enquanto o outro canal do
# MAGIC    mesmo microinversor gerava (modulo desconectado).
# MAGIC 3. **`T_sup` e a temperatura de celula pelo modelo NOCT**, com a temperatura
# MAGIC    do ar do NASA POWER. A temperatura reportada pelos microinversores entra
# MAGIC    como variante de sensibilidade.
# MAGIC
# MAGIC Saidas: `fato_irradiancia_plano_hora`, `fato_energia_canal_hora`,
# MAGIC `gold_incidencia_usina_hora`, `gold_incidencia_usina_mes` e
# MAGIC `energia_esperada_atlas_mes` (energia esperada pela climatologia do Atlas,
# MAGIC para comparabilidade com Prudencio, 2024).

# COMMAND ----------

# MAGIC %pip install pvlib==0.11.2 --quiet

# COMMAND ----------

import json, numpy as np, pandas as pd, pvlib
SILVER, GOLD = "fotovoltaica_ifes.silver", "fotovoltaica_ifes.gold"
U = spark.table(f"{GOLD}.dim_usina").toPandas().iloc[0]
EQ = spark.table(f"{GOLD}.dim_equipamento").toPandas().set_index("id_equipamento")
LIMIAR = float(spark.sql(f"SELECT valor FROM {SILVER}.config_parametros WHERE parametro = 'irradiancia_minima_aderencia_wh_m2'").collect()[0][0])
loc = pvlib.location.Location(U.latitude, U.longitude, "UTC", U.altitude_m)
print(U[["latitude", "longitude", "inclinacao_modulos_graus", "azimute_modulos_graus", "rastreador_angulo_maximo_graus",
         "modelo_transposicao", "modelo_temperatura_celula"]].to_dict(), "| limiar:", LIMIAR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Irradiancia no plano dos modulos (`fato_irradiancia_plano_hora`)

# COMMAND ----------

irr = spark.table(f"{GOLD}.fato_irradiancia_hora").toPandas().sort_values("hora_utc")
t_ini = pd.to_datetime(irr.hora_utc).dt.tz_localize("UTC")
meio = pd.DatetimeIndex(t_ini + pd.Timedelta(minutes=30))
sp = loc.get_solarposition(meio)
ext = pvlib.irradiance.get_extra_radiation(meio)
am = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith.values)
ghi, dni, dhi = (irr[c].fillna(0).clip(lower=0).values for c in ["ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2"])

geom = {"fixa": (pd.Series(U.inclinacao_modulos_graus, index=meio), pd.Series(U.azimute_modulos_graus, index=meio))}
tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_tilt=U.rastreador_inclinacao_eixo_graus,
                               axis_azimuth=U.rastreador_azimute_eixo_graus,
                               max_angle=U.rastreador_angulo_maximo_graus, backtrack=False)
geom["rastreador"] = (tr.surface_tilt.fillna(U.rastreador_inclinacao_eixo_graus),
                      tr.surface_azimuth.fillna(U.rastreador_azimute_eixo_graus))

plano = pd.DataFrame({"hora_utc": irr.hora_utc.values, "hora_local": irr.hora_local.values,
                      "zenite_solar_graus": sp.apparent_zenith.values, "azimute_solar_graus": sp.azimuth.values,
                      "angulo_rastreador_graus": tr.tracker_theta.values})
for est, (tilt, az) in geom.items():
    for mod, nome in [("perez", "perez"), ("haydavies", "haydavies"), ("isotropic", "isotropico")]:
        poa = pvlib.irradiance.get_total_irradiance(tilt, az, sp.apparent_zenith, sp.azimuth, dni, ghi, dhi,
                                                    dni_extra=ext, airmass=am, model=mod)["poa_global"]
        plano[f"poa_{est}_{nome}_wh_m2"] = np.where(sp.apparent_zenith.values < 90, poa.fillna(0).clip(lower=0).values, 0.0)
# Variante de validacao com irradiancia medida em solo (INMET A612): a GHI horaria
# e decomposta em DNI e DHI pelo modelo de Erbs e transposta pelo mesmo Perez.
ghi_i = irr.ghi_inmet_wh_m2.values
erbs = pvlib.irradiance.erbs(pd.Series(ghi_i, index=meio).fillna(0), sp.apparent_zenith, meio)
for est, (tilt, az) in geom.items():
    poa_i = pvlib.irradiance.get_total_irradiance(tilt, az, sp.apparent_zenith, sp.azimuth, erbs["dni"].fillna(0),
                                                  pd.Series(ghi_i, index=meio).fillna(0), erbs["dhi"].fillna(0),
                                                  dni_extra=ext, airmass=am, model="perez")["poa_global"]
    plano[f"poa_{est}_inmet_wh_m2"] = np.where(np.isnan(ghi_i) | (sp.apparent_zenith.values >= 90), np.nan,
                                               poa_i.fillna(0).clip(lower=0).values)
# Geometria fixa aplicada ao rastreador (dias em que ele ficou parado na posicao de referencia)
for mod in ["perez", "haydavies", "isotropico", "inmet"]:
    plano[f"poa_rastreador_parado_{mod}_wh_m2"] = plano[f"poa_fixa_{mod}_wh_m2"]
spark.createDataFrame(plano).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD}.fato_irradiancia_plano_hora")
spark.sql(f"COMMENT ON TABLE {GOLD}.fato_irradiancia_plano_hora IS 'Irradiancia horaria no plano dos modulos (fixa e rastreador), pelos modelos de Perez, Hay-Davies e isotropico, a partir do NASA POWER. Rotulo = inicio da hora.'")
print("Razao anual plano/horizontal (Perez):",
      round(plano.poa_fixa_perez_wh_m2.sum() / ghi.sum(), 3), "fixa |",
      round(plano.poa_rastreador_perez_wh_m2.sum() / ghi.sum(), 3), "rastreador")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Estado diario do rastreador (`estado_rastreador_dia`)
# MAGIC A razao entre a energia do canal A do rastreador (microinversor 1) e a do
# MAGIC canal A fixo do microinversor 2, normalizada pela razao do meio-dia
# MAGIC (11h-13h), indica o que o rastreador fez em cada dia: se ganha de manha
# MAGIC (8h-10h) e a tarde (14h-16h), esta rastreando; se a razao fica constante,
# MAGIC esta parado na posicao de referencia (equivale a um modulo fixo); se perde
# MAGIC de manha e ganha a tarde, esta travado voltado para oeste. Dias nublados
# MAGIC (pouca energia nas janelas) ou ambiguos herdam o estado do dia
# MAGIC classificado mais proximo, ate 7 dias. A janela da manha comeca as 8h
# MAGIC porque as 7h, no inverno, o sol baixo reduz a geracao do rastreador por
# MAGIC motivo diferente do estado do mecanismo.

# COMMAND ----------

est = spark.sql(f"""
WITH p AS (
  SELECT date(instante_coleta_local) AS data, hour(instante_coleta_local) AS h, id_equipamento, potencia_w
  FROM {GOLD}.fato_potencia_5min WHERE id_equipamento IN ('403000261021-A', '403000261629-A')),
j AS (
  SELECT data,
    sum(CASE WHEN h BETWEEN 8 AND 9 AND id_equipamento = '403000261021-A' THEN potencia_w END) rm,
    sum(CASE WHEN h BETWEEN 8 AND 9 AND id_equipamento = '403000261629-A' THEN potencia_w END) fm,
    sum(CASE WHEN h BETWEEN 11 AND 12 AND id_equipamento = '403000261021-A' THEN potencia_w END) r0,
    sum(CASE WHEN h BETWEEN 11 AND 12 AND id_equipamento = '403000261629-A' THEN potencia_w END) f0,
    sum(CASE WHEN h BETWEEN 14 AND 15 AND id_equipamento = '403000261021-A' THEN potencia_w END) ra,
    sum(CASE WHEN h BETWEEN 14 AND 15 AND id_equipamento = '403000261629-A' THEN potencia_w END) fa
  FROM p GROUP BY 1)
SELECT data, (rm / fm) / (r0 / f0) AS ganho_manha, (ra / fa) / (r0 / f0) AS ganho_tarde,
       least(fm, f0, fa) * 5 / 60 AS energia_minima_janela_wh
FROM j
""").toPandas()
def classifica(r):
    if pd.isna(r.ganho_manha) or pd.isna(r.ganho_tarde) or r.energia_minima_janela_wh < 60:
        return None
    m, t = r.ganho_manha, r.ganho_tarde
    if m > 1.10 and t > 1.10: return "rastreando"
    if 0.92 <= m <= 1.08 and 0.92 <= t <= 1.08: return "parado_referencia"
    if m < 0.85 and t > 1.10: return "travado_oeste"
    if m > 1.10 and t < 0.85: return "travado_leste"
    return None
est["data"] = pd.to_datetime(est.data)
est = est.set_index("data").reindex(pd.date_range(est.data.min(), est.data.max(), freq="D")).rename_axis("data").reset_index()
est["estado_classificado"] = est.apply(classifica, axis=1)
est["estado"] = est.estado_classificado.ffill(limit=7).combine_first(est.estado_classificado.bfill(limit=7))
est["estado"] = est.estado.fillna("indeterminado")
est["estado_inferido"] = est.estado_classificado.isna()
spark.createDataFrame(est.assign(data=est.data.dt.date)).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD}.estado_rastreador_dia")
spark.sql(f"COMMENT ON TABLE {GOLD}.estado_rastreador_dia IS 'Estado diario do rastreador inferido dos dados (rastreando, parado na referencia, travado a oeste/leste), pela razao rastreador/fixo de manha e a tarde normalizada pelo meio-dia.'")
print(est.groupby([est.data.dt.to_period("M"), "estado"]).size().unstack(fill_value=0).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Energia medida e esperada por canal e hora (`fato_energia_canal_hora`)
# MAGIC A energia medida na hora e a media das 12 leituras de potencia (W) da
# MAGIC hora, em Wh; horas com menos de 12 leituras ficam marcadas como
# MAGIC incompletas. Um canal com modulo e considerado sem geracao no dia quando
# MAGIC gerou menos de 5% do outro canal do mesmo microinversor (com este acima de
# MAGIC 100 Wh). Para os canais do rastreador, a geometria depende do estado do
# MAGIC dia: rastreamento de 1 eixo, posicao fixa de referencia, ou exclusao das
# MAGIC metricas quando travado (registrado no log de eventos).

# COMMAND ----------

med = spark.sql(f"""
WITH h AS (
  SELECT id_equipamento, date_trunc('hour', instante_coleta_local) AS hora_local,
         count(*) AS n_leituras_5min, avg(potencia_w) AS potencia_media_w,
         avg(temperatura_microinversor_c) AS temperatura_microinversor_c,
         max(CAST(flag_potencia_no_teto AS INT)) AS teve_leitura_no_teto,
         max(CAST(flag_horario_reporte_congelado AS INT)) AS teve_reporte_congelado
  FROM {GOLD}.fato_potencia_5min GROUP BY 1, 2),
d AS (
  SELECT id_equipamento, id_microinversor, date(instante_coleta_local) AS data, sum(potencia_w) * 5 / 60 AS energia_dia_wh
  FROM {GOLD}.fato_potencia_5min GROUP BY 1, 2, 3),
d2 AS (SELECT d.*, max(energia_dia_wh) OVER (PARTITION BY id_microinversor, data) AS energia_dia_max_microinversor_wh FROM d)
SELECT h.*, d2.energia_dia_wh,
       (d2.energia_dia_max_microinversor_wh > 100 AND d2.energia_dia_wh < 0.05 * d2.energia_dia_max_microinversor_wh) AS canal_sem_geracao_no_dia
FROM h JOIN d2 ON h.id_equipamento = d2.id_equipamento AND date(h.hora_local) = d2.data
""").toPandas()
med["hora_local"] = pd.to_datetime(med.hora_local)
pl = plano.assign(hora_local=pd.to_datetime(plano.hora_local))
pl["data"] = pl.hora_local.dt.normalize()
pl = pl.merge(est[["data", "estado"]].rename(columns={"estado": "estado_rastreador"}), on="data", how="left")
pl["estado_rastreador"] = pl.estado_rastreador.fillna("indeterminado")
temp_ar = irr.assign(hora_local=pd.to_datetime(irr.hora_local))[["hora_local", "temperatura_ar_c", "vento_2m_m_s", "ghi_wh_m2"]]
MODELOS = ["perez", "haydavies", "isotropico", "inmet"]

linhas = []
for eid, e in EQ.iterrows():
    if e.estrutura == "sem_modulo":
        continue
    x = pl.merge(temp_ar, on="hora_local").merge(med[med.id_equipamento == eid], on="hora_local", how="left")
    x["id_equipamento"] = eid
    if e.estrutura == "rastreador":
        parado = x.estado_rastreador.eq("parado_referencia")
        x["geometria"] = np.where(parado, "fixa_referencia", np.where(x.estado_rastreador.eq("rastreando"), "rastreamento", "travado_ou_indeterminado"))
        for mod in MODELOS:
            x[f"G_{mod}"] = np.where(parado, x[f"poa_rastreador_parado_{mod}_wh_m2"], x[f"poa_rastreador_{mod}_wh_m2"])
    else:
        x["geometria"] = "fixa"
        for mod in MODELOS:
            x[f"G_{mod}"] = x[f"poa_fixa_{mod}_wh_m2"]
    base_param = e.area_modulo_m2 * e.eficiencia_modulo * e.eficiencia_pcu
    for mod in MODELOS:
        G = x[f"G_{mod}"]
        tc_noct = x.temperatura_ar_c + (e.noct_c - 20) / 800 * G
        x[f"esperada_{mod}_noct_wh"] = G * base_param * (1 + e.coef_temp_potencia_pct_c * (tc_noct - 25) / 100)
        if mod == "perez":
            x["temperatura_celula_noct_c"] = tc_noct
            x["esperada_perez_proxy_wh"] = G * base_param * (1 + e.coef_temp_potencia_pct_c * (x.temperatura_microinversor_c - 25) / 100)
    x["poa_wh_m2"] = x["G_perez"]
    x["energia_esperada_wh"] = x["esperada_perez_noct_wh"]
    linhas.append(x)
f = pd.concat(linhas, ignore_index=True)
f["energia_medida_wh"] = f.potencia_media_w
f["hora_completa"] = f.n_leituras_5min.eq(12)
f["canal_sem_geracao_no_dia"] = f.canal_sem_geracao_no_dia.fillna(False).astype(bool)
f["rastreador_fora_de_operacao_normal"] = f.geometria.eq("travado_ou_indeterminado")
f["incluir_aderencia"] = f.hora_completa & ~f.canal_sem_geracao_no_dia & ~f.rastreador_fora_de_operacao_normal & (f.poa_wh_m2 >= LIMIAR)
cols = ["id_equipamento", "hora_local", "geometria", "estado_rastreador", "n_leituras_5min", "hora_completa",
        "canal_sem_geracao_no_dia", "rastreador_fora_de_operacao_normal", "incluir_aderencia",
        "energia_medida_wh", "energia_esperada_wh", "poa_wh_m2", "ghi_wh_m2", "temperatura_ar_c", "vento_2m_m_s",
        "temperatura_celula_noct_c", "temperatura_microinversor_c", "teve_leitura_no_teto", "teve_reporte_congelado"] + \
       [c for c in f if c.startswith("esperada_")]
f = f[cols]
f.insert(1, "id_usina", "ifes_campus_vitoria")
f.insert(3, "id_tempo", f.hora_local.dt.strftime("%Y%m%d%H").astype("int64"))
spark.createDataFrame(f).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.fato_energia_canal_hora")
spark.sql(f"COMMENT ON TABLE {GOLD}.fato_energia_canal_hora IS 'Energia medida e esperada (Equacao 3.1 no plano dos modulos) por canal com modulo e por hora. incluir_aderencia = hora com 12 leituras, canal gerando no dia, rastreador em operacao normal e irradiancia no plano >= limiar.'")
inc = f[f.incluir_aderencia]
print("canal-horas:", len(f), "| incluidas na aderencia:", len(inc),
      "| razao de somas (Perez, NOCT):", round(inc.energia_medida_wh.sum() / inc.energia_esperada_wh.sum(), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Tabela final por usina e hora (`gold_incidencia_usina_hora`)
# MAGIC Soma dos canais incluidos na aderencia. `flag_hora_pareada_usina` indica
# MAGIC as horas em que todos os canais com modulo em operacao naquele dia
# MAGIC entraram (comparacao da usina inteira, sem lacunas de equipamento).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.gold_incidencia_usina_hora
COMMENT 'Usina x hora: irradiancia, energia medida e esperada (soma dos canais incluidos na aderencia) e razao.'
AS
WITH c AS (
  SELECT id_usina, hora_local, id_tempo,
         count_if(NOT canal_sem_geracao_no_dia AND NOT rastreador_fora_de_operacao_normal) AS n_canais_em_operacao,
         count_if(incluir_aderencia) AS n_canais_incluidos,
         sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) AS energia_medida_wh,
         sum(CASE WHEN incluir_aderencia THEN energia_esperada_wh END) AS energia_esperada_wh,
         sum(CASE WHEN incluir_aderencia THEN esperada_haydavies_noct_wh END) AS esperada_haydavies_noct_wh,
         sum(CASE WHEN incluir_aderencia THEN esperada_isotropico_noct_wh END) AS esperada_isotropico_noct_wh,
         sum(CASE WHEN incluir_aderencia THEN esperada_perez_proxy_wh END) AS esperada_perez_proxy_wh,
         sum(CASE WHEN incluir_aderencia THEN esperada_inmet_noct_wh END) AS esperada_inmet_noct_wh,
         avg(temperatura_microinversor_c) AS temperatura_microinversor_c
  FROM {GOLD}.fato_energia_canal_hora GROUP BY 1, 2, 3)
SELECT c.*, c.n_canais_incluidos > 0 AND c.n_canais_incluidos = c.n_canais_em_operacao AS flag_hora_pareada_usina,
       try_divide(c.energia_medida_wh, c.energia_esperada_wh) AS razao_real_esperado,
       i.ghi_wh_m2, i.dni_wh_m2, i.dhi_wh_m2, i.ghi_ceu_claro_wh_m2, i.temperatura_ar_c, i.vento_2m_m_s,
       i.ghi_inmet_wh_m2, i.precipitacao_inmet_mm,
       p.poa_fixa_perez_wh_m2, p.poa_rastreador_perez_wh_m2,
       current_timestamp() AS atualizado_em
FROM c JOIN {GOLD}.fato_irradiancia_hora i USING (hora_local)
JOIN {GOLD}.fato_irradiancia_plano_hora p USING (hora_local)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Tabela final por usina e mes (`gold_incidencia_usina_mes`)
# MAGIC Razao de somas (soma medida / soma esperada), como o PR da IEC 61724-1, e
# MAGIC indicadores de cobertura calculados so sobre o periodo diurno (horas com
# MAGIC sol acima do horizonte), ja que o microinversor nao opera a noite.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.gold_incidencia_usina_mes
COMMENT 'Usina x mes: razao de somas medida/esperada, cobertura e disponibilidade diurnas e referencias de irradiacao.'
AS
WITH diurno AS (
  SELECT p.hora_local FROM {GOLD}.fato_irradiancia_plano_hora p WHERE p.zenite_solar_graus < 90),
canal_hora AS (
  SELECT f.*, year(f.hora_local) ano, month(f.hora_local) mes FROM {GOLD}.fato_energia_canal_hora f
  JOIN diurno USING (hora_local) WHERE NOT f.canal_sem_geracao_no_dia),
m AS (
  SELECT 'ifes_campus_vitoria' AS id_usina, ano, mes,
         sum(n_leituras_5min) AS leituras_diurnas_registradas,
         count(*) * 12 AS leituras_diurnas_possiveis,
         count_if(incluir_aderencia) AS canal_horas_incluidas,
         sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) / 1000 AS energia_medida_kwh,
         sum(CASE WHEN incluir_aderencia THEN energia_esperada_wh END) / 1000 AS energia_esperada_kwh,
         sum(CASE WHEN incluir_aderencia THEN esperada_haydavies_noct_wh END) / 1000 AS esperada_haydavies_kwh,
         sum(CASE WHEN incluir_aderencia THEN esperada_isotropico_noct_wh END) / 1000 AS esperada_isotropico_kwh,
         sum(CASE WHEN incluir_aderencia THEN esperada_perez_proxy_wh END) / 1000 AS esperada_perez_proxy_kwh,
         sum(CASE WHEN incluir_aderencia AND esperada_inmet_noct_wh IS NOT NULL THEN energia_medida_wh END) / 1000 AS energia_medida_horas_inmet_kwh,
         sum(CASE WHEN incluir_aderencia THEN esperada_inmet_noct_wh END) / 1000 AS esperada_inmet_kwh,
         count(DISTINCT CASE WHEN n_leituras_5min > 0 THEN date(hora_local) END) AS dias_com_leitura
  FROM canal_hora GROUP BY 1, 2, 3),
irr AS (
  SELECT year(hora_local) ano, month(hora_local) mes, sum(ghi_wh_m2) / 1000 / count(DISTINCT date(hora_local)) AS ghi_nasa_kwh_m2_dia,
         sum(ghi_inmet_wh_m2) / 1000 AS ghi_inmet_kwh_m2, sum(precipitacao_inmet_mm) AS precipitacao_inmet_mm
  FROM {GOLD}.fato_irradiancia_hora GROUP BY 1, 2),
ref AS (
  SELECT mes, max(CASE WHEN fonte = 'atlas_labren_2017' AND camada = 'global_horizontal' THEN valor_kwh_m2_dia END) AS ghi_atlas_kwh_m2_dia,
              max(CASE WHEN fonte = 'nasa_power_2001_2020' AND camada = 'global_horizontal' THEN valor_kwh_m2_dia END) AS ghi_nasa_climatologia_kwh_m2_dia
  FROM {GOLD}.ref_climatologia_mes GROUP BY 1)
SELECT m.*, day(last_day(make_date(m.ano, m.mes, 1))) AS dias_calendario,
       round(m.dias_com_leitura / day(last_day(make_date(m.ano, m.mes, 1))), 4) AS pct_dias_com_leitura,
       round(m.leituras_diurnas_registradas / m.leituras_diurnas_possiveis, 4) AS disponibilidade_diurna,
       round(try_divide(m.energia_medida_kwh, m.energia_esperada_kwh), 4) AS razao_real_esperado_mes,
       irr.ghi_nasa_kwh_m2_dia, irr.precipitacao_inmet_mm, ref.ghi_atlas_kwh_m2_dia, ref.ghi_nasa_climatologia_kwh_m2_dia,
       current_timestamp() AS atualizado_em
FROM m LEFT JOIN irr USING (ano, mes) LEFT JOIN ref USING (mes)
""")
display(spark.sql(f"SELECT ano, mes, canal_horas_incluidas, round(energia_medida_kwh,1) med, round(energia_esperada_kwh,1) esp, razao_real_esperado_mes, disponibilidade_diurna FROM {GOLD}.gold_incidencia_usina_mes ORDER BY 1, 2"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Energia esperada pela climatologia do Atlas (`energia_esperada_atlas_mes`)
# MAGIC Implementa a Secao 3.8.3 (comparabilidade com Prudencio, 2024): para cada
# MAGIC mes, a irradiacao global e a difusa diarias medias do Atlas (celula mais
# MAGIC proxima) sao distribuidas ao longo de um dia medio do mes pelas funcoes de
# MAGIC Collares-Pereira e Rabl (global) e Liu e Jordan (difusa), descritas em
# MAGIC Duffie, Beckman e Blair (2020), transpostas pelo mesmo modelo de Perez e
# MAGIC convertidas em energia pela mesma Equacao 3.1 de cada canal, com a
# MAGIC temperatura do ar da climatologia do NASA POWER. O resultado e a energia
# MAGIC diaria esperada de um mes tipico, comparada, canal a canal, com a energia
# MAGIC medida nos dias em que o canal teve todas as horas diurnas completas.

# COMMAND ----------

ref = spark.table(f"{GOLD}.ref_climatologia_mes").toPandas()
atl = ref[ref.fonte == "atlas_labren_2017"].pivot(index="mes", columns="camada", values="valor_kwh_m2_dia")
t2m = spark.sql(f"""SELECT * FROM {SILVER}.incidencia_nasa_climatologia WHERE parametro = 'T2M'""").toPandas().iloc[0]
DIA_MEDIO = {1: 17, 2: 16, 3: 16, 4: 15, 5: 15, 6: 11, 7: 17, 8: 16, 9: 15, 10: 15, 11: 14, 12: 10}  # Klein (1977)
MES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
res = []
for m in range(1, 13):
    dia = pd.Timestamp(2021, m, DIA_MEDIO[m])
    t = pd.date_range(dia, periods=24 * 60, freq="min", tz="Etc/GMT+3")
    s = loc.get_solarposition(t)
    dec = pvlib.solarposition.declination_spencer71(t.dayofyear)          # radianos
    ws = np.degrees(np.arccos(-np.tan(np.radians(U.latitude)) * np.tan(dec)))  # angulo horario do por do sol
    w = pvlib.solarposition.hour_angle(t, U.longitude, s.equation_of_time.values)
    a, b = 0.409 + 0.5016 * np.sin(np.radians(ws - 60)), 0.6609 - 0.4767 * np.sin(np.radians(ws - 60))
    den = np.sin(np.radians(ws)) - np.pi * ws / 180 * np.cos(np.radians(ws))
    rt = np.pi / 24 * (a + b * np.cos(np.radians(w))) * (np.cos(np.radians(w)) - np.cos(np.radians(ws))) / den
    rd = np.pi / 24 * (np.cos(np.radians(w)) - np.cos(np.radians(ws))) / den
    ghi_m = np.clip(rt, 0, None) * atl.loc[m, "global_horizontal"] * 1000   # W/m2 (Wh/m2 por hora, amostrado por minuto)
    dhi_m = np.minimum(np.clip(rd, 0, None) * atl.loc[m, "difusa"] * 1000, ghi_m)
    cz = np.cos(np.radians(s.apparent_zenith.clip(upper=89.9)))
    dni_m = np.where(s.apparent_zenith < 89, (ghi_m - dhi_m) / cz, 0)
    tra = pvlib.tracking.singleaxis(s.apparent_zenith, s.azimuth, axis_tilt=U.rastreador_inclinacao_eixo_graus,
                                    axis_azimuth=U.rastreador_azimute_eixo_graus, max_angle=U.rastreador_angulo_maximo_graus, backtrack=False)
    poa = {}
    for est, (tilt, az) in {"fixa": (U.inclinacao_modulos_graus, U.azimute_modulos_graus),
                            "rastreador": (tra.surface_tilt.fillna(U.rastreador_inclinacao_eixo_graus), tra.surface_azimuth.fillna(0))}.items():
        p = pvlib.irradiance.get_total_irradiance(tilt, az, s.apparent_zenith, s.azimuth, dni_m, ghi_m, dhi_m,
                                                  dni_extra=pvlib.irradiance.get_extra_radiation(t),
                                                  airmass=pvlib.atmosphere.get_relative_airmass(s.apparent_zenith), model="perez")["poa_global"]
        poa[est] = p.fillna(0).clip(lower=0).values
    ta = float(t2m[MES[m - 1]])
    for eid, e in EQ.iterrows():
        if e.estrutura == "sem_modulo":
            continue
        G = poa[e.estrutura]
        tc = ta + (e.noct_c - 20) / 800 * G
        p_w = G / 1000 * e.area_modulo_m2 * e.eficiencia_modulo * e.eficiencia_pcu * 1000 * (1 + e.coef_temp_potencia_pct_c * (tc - 25) / 100)
        res.append((m, eid, float(np.sum(np.clip(p_w, 0, None)) / 60), float(np.sum(G) / 60 / 1000),
                    float(atl.loc[m, "global_horizontal"]), float(atl.loc[m, "difusa"])))
atlas = pd.DataFrame(res, columns=["mes", "id_equipamento", "energia_esperada_atlas_wh_dia", "irradiacao_plano_atlas_kwh_m2_dia",
                                   "ghi_atlas_kwh_m2_dia", "dhi_atlas_kwh_m2_dia"])
spark.createDataFrame(atlas).createOrReplaceTempView("atlas_canal")

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.energia_esperada_atlas_mes
COMMENT 'Energia diaria esperada de um mes tipico pela climatologia do Atlas (Equacao 3.1 por canal, plano dos modulos) e energia diaria media medida nos dias completos.'
AS
WITH dia_canal AS (
  SELECT id_equipamento, date(hora_local) AS data, count_if(NOT hora_completa AND poa_wh_m2 >= {LIMIAR}) AS horas_diurnas_incompletas,
         sum(energia_medida_wh) AS energia_medida_wh,
         max(CAST(canal_sem_geracao_no_dia OR rastreador_fora_de_operacao_normal AS INT)) AS sem_geracao
  FROM {GOLD}.fato_energia_canal_hora GROUP BY 1, 2),
dia_flag AS (
  SELECT data, max(CASE WHEN sem_geracao = 0 THEN horas_diurnas_incompletas END) AS max_incompletas,
         count_if(sem_geracao = 0) AS n_canais
  FROM dia_canal GROUP BY 1),
base AS (
  SELECT dc.data, month(dc.data) AS mes, dc.energia_medida_wh, a.energia_esperada_atlas_wh_dia
  FROM dia_canal dc
  JOIN atlas_canal a ON a.mes = month(dc.data) AND a.id_equipamento = dc.id_equipamento
  WHERE dc.sem_geracao = 0 AND dc.horas_diurnas_incompletas = 0),
dia_ok AS (
  SELECT data, mes, sum(energia_medida_wh) AS medida_wh, sum(energia_esperada_atlas_wh_dia) AS esperada_wh
  FROM base GROUP BY 1, 2),
ghi AS (SELECT mes, first(ghi_atlas_kwh_m2_dia) AS ghi_atlas_kwh_m2_dia FROM atlas_canal GROUP BY 1)
SELECT d.mes, count(*) AS dias_com_canal_completo, avg(d.medida_wh) / 1000 AS energia_medida_kwh_dia,
       avg(d.esperada_wh) / 1000 AS energia_esperada_atlas_kwh_dia,
       try_divide(sum(d.medida_wh), sum(d.esperada_wh)) AS razao_real_esperado_atlas,
       max(g.ghi_atlas_kwh_m2_dia) AS ghi_atlas_kwh_m2_dia, current_timestamp() AS atualizado_em
FROM dia_ok d LEFT JOIN ghi g USING (mes) GROUP BY d.mes
""")
display(spark.table(f"{GOLD}.energia_esperada_atlas_mes").orderBy("mes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Chaves primarias e estrangeiras (informacionais)

# COMMAND ----------

for tab, cols in {"fato_irradiancia_plano_hora": ["hora_local"], "fato_energia_canal_hora": ["id_equipamento", "hora_local"],
                  "gold_incidencia_usina_hora": ["id_usina", "hora_local"], "gold_incidencia_usina_mes": ["id_usina", "ano", "mes"],
                  "energia_esperada_atlas_mes": ["mes"], "estado_rastreador_dia": ["data"]}.items():
    for c in cols:
        spark.sql(f"ALTER TABLE {GOLD}.{tab} ALTER COLUMN {c} SET NOT NULL")
pks = [("dim_usina", "pk_dim_usina", "id_usina"), ("dim_equipamento", "pk_dim_equipamento", "id_equipamento"),
       ("dim_tempo", "pk_dim_tempo", "id_tempo"), ("fato_potencia_5min", "pk_fato_potencia_5min", "id_equipamento, instante_coleta_local"),
       ("fato_irradiancia_hora", "pk_fato_irradiancia_hora", "id_usina, hora_local"),
       ("fato_irradiancia_plano_hora", "pk_fato_irradiancia_plano_hora", "hora_local"),
       ("fato_energia_canal_hora", "pk_fato_energia_canal_hora", "id_equipamento, hora_local"),
       ("ref_climatologia_mes", "pk_ref_climatologia_mes", "fonte, camada, mes"),
       ("gold_incidencia_usina_hora", "pk_gold_incidencia_usina_hora", "id_usina, hora_local"),
       ("gold_incidencia_usina_mes", "pk_gold_incidencia_usina_mes", "id_usina, ano, mes"),
       ("energia_esperada_atlas_mes", "pk_energia_esperada_atlas_mes", "mes"),
       ("estado_rastreador_dia", "pk_estado_rastreador_dia", "data")]
fks = [("dim_equipamento", "fk_equipamento_usina", "id_usina", "dim_usina (id_usina)"),
       ("fato_potencia_5min", "fk_potencia_equipamento", "id_equipamento", "dim_equipamento (id_equipamento)"),
       ("fato_potencia_5min", "fk_potencia_usina", "id_usina", "dim_usina (id_usina)"),
       ("fato_irradiancia_hora", "fk_irradiancia_usina", "id_usina", "dim_usina (id_usina)"),
       ("fato_energia_canal_hora", "fk_energia_canal_equipamento", "id_equipamento", "dim_equipamento (id_equipamento)"),
       ("fato_energia_canal_hora", "fk_energia_canal_tempo", "id_tempo", "dim_tempo (id_tempo)"),
       ("gold_incidencia_usina_hora", "fk_usina_hora_usina", "id_usina", "dim_usina (id_usina)"),
       ("gold_incidencia_usina_hora", "fk_usina_hora_tempo", "id_tempo", "dim_tempo (id_tempo)"),
       ("gold_incidencia_usina_mes", "fk_usina_mes_usina", "id_usina", "dim_usina (id_usina)")]
# remove todas as chaves existentes (estrangeiras antes das primarias) e recria (primarias antes das estrangeiras)
existentes = spark.sql("""SELECT table_name, constraint_name, constraint_type
                          FROM fotovoltaica_ifes.information_schema.table_constraints WHERE table_schema = 'gold'""").collect()
for tipo in ("FOREIGN KEY", "PRIMARY KEY"):
    for c_ in existentes:
        if c_["constraint_type"] == tipo:
            spark.sql(f"ALTER TABLE {GOLD}.{c_['table_name']} DROP CONSTRAINT IF EXISTS {c_['constraint_name']}")
for t, n, c in pks:
    spark.sql(f"ALTER TABLE {GOLD}.{t} ADD CONSTRAINT {n} PRIMARY KEY ({c})")
for t, n, c, ref_ in fks:
    spark.sql(f"ALTER TABLE {GOLD}.{t} ADD CONSTRAINT {n} FOREIGN KEY ({c}) REFERENCES {GOLD}.{ref_}")

# COMMAND ----------

r = spark.sql(f"""SELECT count(*) n, count_if(incluir_aderencia) incl,
   round(sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) / sum(CASE WHEN incluir_aderencia THEN energia_esperada_wh END), 4) razao_perez_noct,
   round(sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) / sum(CASE WHEN incluir_aderencia THEN esperada_haydavies_noct_wh END), 4) razao_haydavies_noct,
   round(sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) / sum(CASE WHEN incluir_aderencia THEN esperada_isotropico_noct_wh END), 4) razao_isotropico_noct,
   round(sum(CASE WHEN incluir_aderencia THEN energia_medida_wh END) / sum(CASE WHEN incluir_aderencia THEN esperada_perez_proxy_wh END), 4) razao_perez_proxy
   FROM {GOLD}.fato_energia_canal_hora""").collect()[0].asDict()
r["horas_usina_pareadas"] = spark.sql(f"SELECT count_if(flag_hora_pareada_usina) FROM {GOLD}.gold_incidencia_usina_hora").collect()[0][0]
print(r)
dbutils.notebook.exit(json.dumps(r))
