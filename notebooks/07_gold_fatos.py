# Databricks notebook source
# MAGIC %md
# MAGIC # Camada gold - fatos (fato_potencia_5min, fato_irradiancia_hora, ref_climatologia_mes)
# MAGIC
# MAGIC - `fato_potencia_5min`: potencia a cada 5 minutos dos canais A e B, com
# MAGIC   indicadores de qualidade.
# MAGIC - `fato_irradiancia_hora`: componentes da irradiancia
# MAGIC   (GHI, DNI e DHI) do NASA POWER, o vento e, como validacao, a radiacao
# MAGIC   global e a precipitacao medidas na estacao A612 do INMET. Todas as
# MAGIC   series sao rotuladas pelo inicio da hora (convencao conferida no 05b).
# MAGIC - `ref_climatologia_mes`: as quatro camadas do Atlas Brasileiro de Energia
# MAGIC   Solar 2017 (global horizontal, difusa, direta normal e plano inclinado).

# COMMAND ----------

SILVER, GOLD = "fotovoltaica_ifes.silver", "fotovoltaica_ifes.gold"
ID_USINA = "ifes_campus_vitoria"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) fato_potencia_5min
# MAGIC Grao: canal (`id_equipamento`) e instante de coleta (nome do arquivo).
# MAGIC Indicadores:
# MAGIC - `flag_horario_reporte_congelado`: o horario de reporte do microinversor
# MAGIC   nao avancou desde a leitura anterior do mesmo microinversor;
# MAGIC - `flag_reporting_time_divergente`: horario de reporte a mais de 60 min do
# MAGIC   instante de coleta (inclui o adiantamento de ~3h45 a partir de out/2021);
# MAGIC - `flag_potencia_no_teto`: potencia >= 265 W, proxima do limite de saida do
# MAGIC   canal observado nos dados (~272 W), usada para medir *clipping*;
# MAGIC - `flag_potencia_acima_nominal`: potencia acima da nominal do modulo;
# MAGIC - `flag_canal_sem_modulo`: canal sem modulo conectado (B dos inversores 3 e 4);
# MAGIC - `flag_temperatura_ausente`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.fato_potencia_5min
COMMENT 'Potencia por canal a cada 5 minutos (canais A e B), com indicadores de qualidade.'
AS
WITH s AS (
  SELECT g.*, lag(g.reporting_time_local) OVER (PARTITION BY g.id_equipamento ORDER BY g.instante_coleta_local) AS reporte_anterior
  FROM {SILVER}.usina_geracao g)
SELECT s.id_equipamento, '{ID_USINA}' AS id_usina, s.id_microinversor, s.canal,
       s.instante_coleta_local, s.instante_coleta_utc,
       CAST(date_format(date_trunc('hour', s.instante_coleta_local), 'yyyyMMddHH') AS BIGINT) AS id_tempo,
       s.potencia_w, s.tensao_v, s.frequencia_hz, s.temperatura_c AS temperatura_microinversor_c,
       s.reporting_time_local AS reporting_time_equipamento_local,
       coalesce(s.reporting_time_local = s.reporte_anterior, false) AS flag_horario_reporte_congelado,
       coalesce(s.flag_reporting_time_divergente, false) AS flag_reporting_time_divergente,
       s.potencia_w >= 265 AS flag_potencia_no_teto,
       coalesce(s.potencia_w > e.potencia_nominal_modulo_wp, false) AS flag_potencia_acima_nominal,
       e.estrutura = 'sem_modulo' AS flag_canal_sem_modulo,
       s.temperatura_c IS NULL AS flag_temperatura_ausente,
       s.arquivo_origem, current_timestamp() AS atualizado_em
FROM s JOIN {GOLD}.dim_equipamento e USING (id_equipamento)
""")
for c in ["id_equipamento", "instante_coleta_local"]:
    spark.sql(f"ALTER TABLE {GOLD}.fato_potencia_5min ALTER COLUMN {c} SET NOT NULL")
display(spark.sql(f"""SELECT canal, count(*) linhas, count_if(flag_horario_reporte_congelado) congelado,
   count_if(flag_reporting_time_divergente) divergente, count_if(flag_potencia_no_teto) no_teto,
   count_if(flag_potencia_acima_nominal) acima_nominal, count_if(flag_canal_sem_modulo) sem_modulo
   FROM {GOLD}.fato_potencia_5min GROUP BY 1 ORDER BY 1"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) fato_irradiancia_hora
# MAGIC Grao: usina e hora (rotulo = inicio da hora). NASA POWER como fonte
# MAGIC principal; INMET A612 como medicao de solo para validacao.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.fato_irradiancia_hora
COMMENT 'Irradiancia horaria (NASA POWER, com componentes) e medicao de solo do INMET A612, rotuladas pelo inicio da hora.'
AS
SELECT '{ID_USINA}' AS id_usina, n.hora_inicio_local AS hora_local, n.hora_inicio_utc AS hora_utc,
       CAST(date_format(n.hora_inicio_local, 'yyyyMMddHH') AS BIGINT) AS id_tempo,
       n.ghi_wh_m2, n.dni_wh_m2, n.dhi_wh_m2, n.ghi_ceu_claro_wh_m2, n.dni_ceu_claro_wh_m2, n.dhi_ceu_claro_wh_m2,
       n.temperatura_ar_c, n.vento_2m_m_s,
       i.ghi_wh_m2 AS ghi_inmet_wh_m2, i.precipitacao_mm AS precipitacao_inmet_mm,
       i.temperatura_ar_c AS temperatura_ar_inmet_c,
       current_timestamp() AS atualizado_em
FROM {SILVER}.incidencia_nasa_horario_componentes n
LEFT JOIN {SILVER}.incidencia_inmet_a612 i USING (hora_inicio_utc)
""")
for c in ["id_usina", "hora_local"]:
    spark.sql(f"ALTER TABLE {GOLD}.fato_irradiancia_hora ALTER COLUMN {c} SET NOT NULL")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) ref_climatologia_mes
# MAGIC Medias mensais de longo prazo em kWh/m².dia: as quatro camadas do Atlas
# MAGIC para a celula mais proxima da usina e a climatologia 2001-2020 do NASA
# MAGIC POWER (global horizontal com nuvens e em ceu claro).

# COMMAND ----------

meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
stack = ", ".join(f"{i + 1}, {m}" for i, m in enumerate(meses))
spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.ref_climatologia_mes
COMMENT 'Referencias climatologicas mensais (kWh/m2.dia): Atlas Brasileiro de Energia Solar 2017, celula mais proxima, e NASA POWER 2001-2020.'
AS
WITH cel AS (SELECT id_celula FROM {SILVER}.incidencia_atlas_labren
             WHERE camada = 'global_horizontal' AND mes = 1 ORDER BY distancia_km LIMIT 1)
SELECT 'atlas_labren_2017' AS fonte, a.camada, a.mes, a.valor_wh_m2_dia / 1000 AS valor_kwh_m2_dia,
       CAST(a.id_celula AS STRING) AS id_referencia, a.distancia_km
FROM {SILVER}.incidencia_atlas_labren a JOIN cel USING (id_celula)
UNION ALL
SELECT 'nasa_power_2001_2020', CASE parametro WHEN 'ALLSKY_SFC_SW_DWN' THEN 'global_horizontal'
                                              ELSE 'global_horizontal_ceu_claro' END,
       mes, valor, 'ponto_usina', 0.0
FROM (SELECT parametro, stack(12, {stack}) AS (mes, valor) FROM {SILVER}.incidencia_nasa_climatologia
      WHERE parametro IN ('ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN'))
""")
for c in ["fonte", "camada", "mes"]:
    spark.sql(f"ALTER TABLE {GOLD}.ref_climatologia_mes ALTER COLUMN {c} SET NOT NULL")
display(spark.sql(f"""SELECT fonte, camada, round(avg(valor_kwh_m2_dia), 3) media_anual, max(distancia_km) dist_km
                      FROM {GOLD}.ref_climatologia_mes GROUP BY 1, 2 ORDER BY 1, 2"""))

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({t: spark.table(f"{GOLD}.{t}").count()
                                  for t in ["fato_potencia_5min", "fato_irradiancia_hora", "ref_climatologia_mes"]}))
