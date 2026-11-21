# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - fontes complementares de irradiancia (NASA com componentes, Atlas LABREN, INMET A612)
# MAGIC
# MAGIC Tipagem, unidades e convencao de tempo das tres fontes gravadas na bronze
# MAGIC pelo notebook 04b. As convencoes de carimbo horario foram conferidas
# MAGIC empiricamente contra o modelo de ceu claro de Ineichen (pvlib), integrado
# MAGIC minuto a minuto e deslocado em passos de 5 minutos:
# MAGIC
# MAGIC | Fonte | Carimbo `h` representa | Evidencia |
# MAGIC |---|---|---|
# MAGIC | NASA POWER horario | o intervalo `[h, h+1)` UTC | menor RMSE (24,5 Wh/m²) e maior correlacao (0,997) do ceu claro da NASA com deslocamento zero |
# MAGIC | INMET A612 | o intervalo `[h-1, h)` UTC (acumulado da hora anterior) | menor RMSE (77 Wh/m²) com deslocamento de -60 min em 29 dias claros |
# MAGIC
# MAGIC Para alinhar as duas fontes com a usina, todas as series horarias passam a
# MAGIC ser rotuladas pelo **inicio** do intervalo, em UTC e no horario local
# MAGIC (UTC-3), o mesmo rotulo que a agregacao horaria da usina usa
# MAGIC (`date_trunc('hour')` sobre o instante de coleta).

# COMMAND ----------

BRONZE, SILVER = "fotovoltaica_ifes.bronze", "fotovoltaica_ifes.silver"
TZ = "America/Sao_Paulo"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) NASA POWER horario com componentes
# MAGIC Valores `-999` (ausente) viram nulo. Unidades: irradiancias em Wh/m² na
# MAGIC hora (equivalente numerico a W/m² medio na hora), T2M em °C, WS2M em m/s.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {SILVER}.incidencia_nasa_horario_componentes
COMMENT 'NASA POWER horario (UTC) com GHI, DNI e DHI com nuvens e em ceu claro, temperatura e vento. Carimbo = inicio da hora.'
AS
WITH t AS (
  SELECT make_timestamp(CAST(YEAR AS INT), CAST(MO AS INT), CAST(DY AS INT), CAST(HR AS INT), 0, 0) AS hora_inicio_utc,
         nullif(CAST(ALLSKY_SFC_SW_DWN AS DOUBLE), -999) AS ghi_wh_m2,
         nullif(CAST(ALLSKY_SFC_SW_DNI AS DOUBLE), -999) AS dni_wh_m2,
         nullif(CAST(ALLSKY_SFC_SW_DIFF AS DOUBLE), -999) AS dhi_wh_m2,
         nullif(CAST(CLRSKY_SFC_SW_DWN AS DOUBLE), -999) AS ghi_ceu_claro_wh_m2,
         nullif(CAST(CLRSKY_SFC_SW_DNI AS DOUBLE), -999) AS dni_ceu_claro_wh_m2,
         nullif(CAST(CLRSKY_SFC_SW_DIFF AS DOUBLE), -999) AS dhi_ceu_claro_wh_m2,
         nullif(CAST(T2M AS DOUBLE), -999) AS temperatura_ar_c,
         nullif(CAST(WS2M AS DOUBLE), -999) AS vento_2m_m_s,
         arquivo_origem
  FROM {BRONZE}.incidencia_nasa_horario_componentes)
SELECT hora_inicio_utc, from_utc_timestamp(hora_inicio_utc, '{TZ}') AS hora_inicio_local, * EXCEPT (hora_inicio_utc)
FROM t
""")
display(spark.sql(f"""SELECT count(*) n, min(hora_inicio_local) inicio, max(hora_inicio_local) fim,
    count_if(ghi_wh_m2 IS NULL) ghi_nulos, round(sum(ghi_wh_m2)/1000, 1) ghi_total_kwh_m2
    FROM {SILVER}.incidencia_nasa_horario_componentes"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Atlas Brasileiro de Energia Solar 2017 (LABREN/INPE)
# MAGIC Formato longo: uma linha por camada, celula e mes, em Wh/m².dia (unidade
# MAGIC do arquivo original). Distancia da celula a usina calculada pela formula
# MAGIC de haversine.

# COMMAND ----------

LAT, LON = -20.31, -40.32
meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
stack = ", ".join(f"{i + 1}, CAST({m} AS DOUBLE)" for i, m in enumerate(meses))
spark.sql(f"""
CREATE OR REPLACE TABLE {SILVER}.incidencia_atlas_labren
COMMENT 'Atlas Brasileiro de Energia Solar 2017 (LABREN/INPE): medias mensais por celula de 0,1 grau, em Wh/m2.dia. Camadas: global_horizontal, difusa, direta_normal, inclinada_latitude.'
AS
WITH b AS (
  SELECT camada, CAST(id AS BIGINT) AS id_celula, CAST(lat AS DOUBLE) AS latitude, CAST(lon AS DOUBLE) AS longitude,
         CAST(annual AS DOUBLE) AS media_anual_wh_m2_dia, stack(12, {stack}) AS (mes, valor_wh_m2_dia)
  FROM {BRONZE}.incidencia_atlas_labren)
SELECT *, round(2 * 6371 * asin(sqrt(pow(sin(radians(latitude - ({LAT})) / 2), 2)
          + cos(radians({LAT})) * cos(radians(latitude)) * pow(sin(radians(longitude - ({LON})) / 2), 2))), 2) AS distancia_km
FROM b
""")
display(spark.sql(f"""SELECT camada, id_celula, latitude, longitude, distancia_km, media_anual_wh_m2_dia
    FROM {SILVER}.incidencia_atlas_labren WHERE mes = 1 ORDER BY distancia_km, camada LIMIT 8"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) INMET A612 (Vitoria)
# MAGIC Radiacao global em kJ/m² na hora anterior ao carimbo, convertida para
# MAGIC Wh/m² (divide por 3,6). A estacao nao grava radiacao a noite (campo vazio),
# MAGIC por isso a ausencia so e tratada como falha quando ocorre com sol acima do
# MAGIC horizonte (feito na gold). Valores negativos ou acima de 1.400 Wh/m² numa
# MAGIC hora sao anulados (fora de faixa fisica).

# COMMAND ----------

nomes = {r["coluna"]: r["nome_original"] for r in spark.table(f"{BRONZE}.incidencia_inmet_a612_colunas").collect()}
def col_de(trecho):
    return [k for k, v in nomes.items() if v.upper().startswith(trecho)][0]
c_data, c_hora = col_de("DATA"), col_de("HORA UTC")
c_rad, c_prec = col_de("RADIACAO GLOBAL"), col_de("PRECIPITA")
c_temp, c_ur, c_vento = col_de("TEMPERATURA DO AR - BULBO SECO"), col_de("UMIDADE RELATIVA DO AR, HORARIA"), col_de("VENTO, VELOCIDADE")
num = lambda c: f"try_cast(replace(trim({c}), ',', '.') AS DOUBLE)"
spark.sql(f"""
CREATE OR REPLACE TABLE {SILVER}.incidencia_inmet_a612
COMMENT 'INMET estacao automatica A612 (Vitoria, -20,271/-40,306, 9 m). Carimbo realinhado para o inicio da hora (o arquivo original marca o fim). Radiacao em Wh/m2 na hora.'
AS
WITH t AS (
  SELECT to_timestamp(concat(replace({c_data}, '/', '-'), ' ', substr({c_hora}, 1, 2)), 'yyyy-MM-dd HH') AS hora_fim_utc,
         {num(c_rad)} / 3.6 AS ghi_bruta_wh_m2, {num(c_prec)} AS precipitacao_mm,
         {num(c_temp)} AS temperatura_ar_c, {num(c_ur)} AS umidade_relativa_pct, {num(c_vento)} AS vento_m_s,
         arquivo_origem
  FROM {BRONZE}.incidencia_inmet_a612)
SELECT hora_fim_utc - INTERVAL 1 HOUR AS hora_inicio_utc,
       from_utc_timestamp(hora_fim_utc - INTERVAL 1 HOUR, '{TZ}') AS hora_inicio_local,
       CASE WHEN ghi_bruta_wh_m2 BETWEEN 0 AND 1400 THEN ghi_bruta_wh_m2 END AS ghi_wh_m2,
       CASE WHEN precipitacao_mm >= 0 THEN precipitacao_mm END AS precipitacao_mm,
       temperatura_ar_c, umidade_relativa_pct, vento_m_s, arquivo_origem
FROM t
""")
display(spark.sql(f"""SELECT year(hora_inicio_local) ano, count(*) horas, count(ghi_wh_m2) horas_com_radiacao,
    round(sum(ghi_wh_m2)/1000, 1) ghi_kwh_m2, round(sum(precipitacao_mm), 1) chuva_mm
    FROM {SILVER}.incidencia_inmet_a612 GROUP BY 1 ORDER BY 1"""))

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({t: spark.table(f"{SILVER}.{t}").count() for t in
    ["incidencia_nasa_horario_componentes", "incidencia_atlas_labren", "incidencia_inmet_a612"]}))
