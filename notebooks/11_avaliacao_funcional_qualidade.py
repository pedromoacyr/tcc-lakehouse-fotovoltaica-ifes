# Databricks notebook source
# MAGIC %md
# MAGIC # Avaliacao funcional e de qualidade dos dados
# MAGIC
# MAGIC Reporta, por regra e por camada, a
# MAGIC proporcao de registros aprovados, sinalizados e em quarentena (Quadro 6 e
# MAGIC Secao 3.6), quantifica os arquivos que chegam so com cabecalho e a
# MAGIC completude restrita ao periodo diurno, e traz os resultados do teste de
# MAGIC idempotencia e carga incremental da ingestao das medicoes (notebook 13).

# COMMAND ----------

import json, pandas as pd
B, S, G, T = "fotovoltaica_ifes.bronze", "fotovoltaica_ifes.silver", "fotovoltaica_ifes.gold", "fotovoltaica_ifes.teste_parecer"
res = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Integridade entre camadas

# COMMAND ----------

res["integridade"] = spark.sql(f"""
SELECT (SELECT count(DISTINCT arquivo_dia) FROM {B}.usina_geracao_arquivos) AS arquivos_diarios_recebidos,
       (SELECT count(DISTINCT arquivo_dia) FROM {B}.usina_geracao) AS dias_com_linhas,
       (SELECT count(*) FROM {B}.usina_geracao_arquivos) AS arquivos_csv,
       (SELECT count_if(n_linhas_dados = 0) FROM {B}.usina_geracao_arquivos) AS arquivos_so_cabecalho,
       (SELECT count(*) FROM {B}.usina_geracao) AS linhas_bronze,
       (SELECT count(*) FROM {S}.usina_geracao) AS linhas_silver,
       (SELECT count(*) FROM {S}.usina_geracao_quarentena) AS linhas_quarentena,
       (SELECT count(*) FROM {G}.fato_potencia_5min) AS linhas_gold_fato_potencia
""").collect()[0].asDict()
i = res["integridade"]
i["fecha_bronze_silver_quarentena"] = i["linhas_bronze"] == i["linhas_silver"] + i["linhas_quarentena"]
i["fecha_silver_gold"] = i["linhas_silver"] == i["linhas_gold_fato_potencia"]
print(json.dumps(i, indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Regras de qualidade por camada (aprovado, sinalizado, quarentena)

# COMMAND ----------

regras = spark.sql(f"""
SELECT 'bronze -> silver' AS camada, 'quarentena' AS tratamento, motivo_quarentena AS regra, count(*) AS n,
       count(*) / (SELECT count(*) FROM {B}.usina_geracao) AS fracao
FROM {S}.usina_geracao_quarentena GROUP BY 3
UNION ALL SELECT 'bronze -> silver', 'aprovado', 'linha com canal e potencia validos', count(*), count(*) / (SELECT count(*) FROM {B}.usina_geracao) FROM {S}.usina_geracao
UNION ALL
SELECT 'silver -> gold (fato_potencia_5min)', 'sinalizado', regra, n, n / total FROM (
  SELECT count(*) total, count_if(flag_horario_reporte_congelado) c1, count_if(flag_reporting_time_divergente) c2,
         count_if(flag_potencia_no_teto) c3, count_if(flag_potencia_acima_nominal) c4, count_if(flag_canal_sem_modulo) c5,
         count_if(flag_temperatura_ausente) c6,
         count_if(NOT (flag_horario_reporte_congelado OR flag_reporting_time_divergente OR flag_potencia_no_teto OR
                       flag_potencia_acima_nominal OR flag_canal_sem_modulo OR flag_temperatura_ausente)) c0
  FROM {G}.fato_potencia_5min)
  LATERAL VIEW stack(7, 'horario_reporte_congelado', c1, 'reporting_time_divergente', c2, 'potencia_no_teto', c3,
                     'potencia_acima_nominal', c4, 'canal_sem_modulo', c5, 'temperatura_ausente', c6, 'sem nenhuma sinalizacao', c0) x AS regra, n
UNION ALL
SELECT 'gold (fato_energia_canal_hora, horas com sol)', 'sinalizado', regra, n, n / total FROM (
  SELECT count(*) total, count_if(NOT hora_completa) c1, count_if(canal_sem_geracao_no_dia) c2,
         count_if(rastreador_fora_de_operacao_normal) c3, count_if(incluir_aderencia) c0
  FROM {G}.fato_energia_canal_hora WHERE poa_wh_m2 >= 50)
  LATERAL VIEW stack(4, 'hora_incompleta (menos de 12 leituras)', c1, 'canal_sem_geracao_no_dia', c2,
                     'rastreador_fora_de_operacao_normal', c3, 'incluida nas metricas de aderencia', c0) x AS regra, n
""").toPandas()
display(regras)
res["regras"] = regras.round(5).to_dict("records")
spark.createDataFrame(regras).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.avaliacao_qualidade_regras")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Completude, conformidade de esquema e consistencia

# COMMAND ----------

res["completude"] = spark.sql(f"""
WITH sol AS (SELECT hora_local FROM {G}.fato_irradiancia_plano_hora WHERE zenite_solar_graus < 90),
lim AS (SELECT min(date(instante_coleta_local)) i, max(date(instante_coleta_local)) f FROM {S}.usina_geracao),
dias AS (SELECT count(*) d FROM (SELECT explode(sequence((SELECT i FROM lim), (SELECT f FROM lim)))))
SELECT (SELECT d FROM dias) AS dias_periodo,
       (SELECT count(DISTINCT date(instante_coleta_local)) FROM {S}.usina_geracao) AS dias_com_leitura,
       (SELECT count(*) FROM sol WHERE date(hora_local) BETWEEN (SELECT i FROM lim) AND (SELECT f FROM lim)) * 12 * 6 AS leituras_diurnas_possiveis_6_canais,
       (SELECT count(*) FROM {G}.fato_potencia_5min f JOIN sol ON date_trunc('hour', f.instante_coleta_local) = sol.hora_local
        WHERE NOT f.flag_canal_sem_modulo) AS leituras_diurnas_registradas_6_canais,
       (SELECT count(*) * 288 * 4 FROM (SELECT explode(sequence((SELECT i FROM lim), (SELECT f FROM lim))))) AS leituras_24h_possiveis_4_microinversores,
       (SELECT count(*) FROM {S}.usina_geracao WHERE canal = 'A') AS leituras_canal_a
""").collect()[0].asDict()
c = res["completude"]
c["completude_dias"] = c["dias_com_leitura"] / c["dias_periodo"]
c["completude_diurna"] = c["leituras_diurnas_registradas_6_canais"] / c["leituras_diurnas_possiveis_6_canais"]
c["completude_24h_antiga"] = c["leituras_canal_a"] / c["leituras_24h_possiveis_4_microinversores"]
res["conformidade_esquema"] = spark.sql(f"""
SELECT count_if(trim(current_power) <> '-') AS linhas_com_valor,
       count_if(trim(current_power) <> '-' AND try_cast(substring_index(trim(current_power), ' ', 1) AS DOUBLE) IS NOT NULL) AS parseaveis
FROM {B}.usina_geracao""").collect()[0].asDict()
res["consistencia"] = spark.sql(f"""
WITH b AS (SELECT arquivo_dia, sum(try_cast(substring_index(trim(current_power), ' ', 1) AS DOUBLE)) s FROM {B}.usina_geracao
           WHERE inverter_id RLIKE '-[AB]$' GROUP BY 1),
s AS (SELECT arquivo_dia, sum(potencia_w) s FROM {S}.usina_geracao GROUP BY 1)
SELECT count(*) dias, count_if(abs(b.s - s.s) < 1e-6) dias_identicos FROM b JOIN s USING (arquivo_dia)""").collect()[0].asDict()
print(json.dumps({k: res[k] for k in ["completude", "conformidade_esquema", "consistencia"]}, indent=1, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Idempotencia e carga incremental da ingestao (resultado do notebook 13)

# COMMAND ----------

try:
    res["teste_ingestao"] = spark.table(f"{T}.resultado_teste_ingestao").toPandas().to_dict("records")
except Exception as e:
    res["teste_ingestao"] = f"tabela nao encontrada: {e}"
print(res["teste_ingestao"])
dbutils.notebook.exit(json.dumps(res, default=str))
