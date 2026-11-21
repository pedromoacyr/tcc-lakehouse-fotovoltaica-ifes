# Databricks notebook source
# MAGIC %md
# MAGIC # Camada silver - geracao da usina (tipagem, conformidade e quarentena)
# MAGIC
# MAGIC Le `bronze.usina_geracao` (texto bruto) e produz:
# MAGIC - `silver.usina_geracao`: uma linha por **canal** de microinversor e por
# MAGIC   instante de coleta, com os campos tipados;
# MAGIC - `silver.usina_geracao_quarentena`: as linhas da bronze que nao entram na
# MAGIC   silver, cada uma com o motivo registrado (nada e descartado em silencio).
# MAGIC
# MAGIC **Canais A e B.** O
# MAGIC microinversor APsystems YC500A tem dois canais (um modulo por canal) e o
# MAGIC exportador grava uma linha por canal. A linha A traz a potencia do canal A
# MAGIC e as leituras do microinversor (frequencia, tensao, temperatura e horario
# MAGIC de reporte); a linha B traz so a potencia do canal B e a tensao (gravada na
# MAGIC posicao da coluna de frequencia). Exemplo real (15/07/2021, 12:00):
# MAGIC
# MAGIC ```
# MAGIC 0,403000261021-A,192 W,60.1 Hz,228 V,39 °C,2021-07-15 11:58:07
# MAGIC 1,403000261021-B,188 W,228 V,,,
# MAGIC 4,403000262049-A,216 W,60.0 Hz,228 V,39 °C,2021-07-15 11:58:07
# MAGIC 5,403000262049-B,0 W,228 V,,,
# MAGIC ```
# MAGIC
# MAGIC O canal B carrega o segundo modulo dos microinversores 1 e 2 (27% da
# MAGIC energia medida) e por isso e mantido. O instante de cada leitura passa a vir do nome do arquivo (chave
# MAGIC natural da Metodologia, Secao 3.5.3), e nao do `Reporting Time`, que fica
# MAGIC congelado quando o equipamento esta inativo e, a partir de out/2021, adianta
# MAGIC cerca de 3h45 em relacao ao horario real (conferido contra o meio-dia
# MAGIC solar). As leituras de frequencia, temperatura e horario de reporte da
# MAGIC linha B sao herdadas da linha A do mesmo microinversor e do mesmo arquivo.
# MAGIC
# MAGIC Reprocessamento idempotente: as duas tabelas sao reescritas por completo
# MAGIC a cada execucao, sempre a partir da bronze.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
SILVER = "fotovoltaica_ifes.silver"
BRONZE = "fotovoltaica_ifes.bronze.usina_geracao"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER}.config_parametros (
    parametro STRING, valor STRING, status STRING, observacao STRING, atualizado_em TIMESTAMP)
""")
config = spark.createDataFrame([(
    "timezone_instante_coleta", "America/Sao_Paulo", "confirmado",
    "Fuso do carimbo de coleta extraido do nome do arquivo (dd_m_aaaa_h_m.csv). Confirmado pelos "
    "dados: o centroide diario da geracao dos modulos fixos fica entre 5 e 30 minutos apos o "
    "meio-dia solar calculado para o local (UTC-3) em todos os meses. O Reporting Time do "
    "equipamento nao e usado como instante: fica congelado quando o equipamento esta inativo e, "
    "a partir de out/2021, adianta cerca de 3h45 em relacao ao nome do arquivo.")],
    ["parametro", "valor", "status", "observacao"]).withColumn("atualizado_em", current_timestamp())
config.createOrReplaceTempView("novo_config")
spark.sql(f"""MERGE INTO {SILVER}.config_parametros a USING novo_config n ON a.parametro = n.parametro
              WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")
TZ = "America/Sao_Paulo"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parse e classificacao de cada linha da bronze

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW bronze_classificada AS
WITH b AS (
  SELECT *,
         element_at(split(arquivo_origem, '/'), -1) AS nome_arquivo,
         CASE WHEN inverter_id RLIKE '^[0-9]+-[AB]$' THEN substring_index(inverter_id, '-', 1) END AS id_microinversor,
         CASE WHEN inverter_id RLIKE '^[0-9]+-[AB]$' THEN right(inverter_id, 1) END AS canal,
         try_cast(substring_index(trim(current_power), ' ', 1) AS DOUBLE) AS potencia_w
  FROM {BRONZE}
)
SELECT *,
  make_timestamp(
    CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[2] AS INT),
    CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[1] AS INT),
    CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[0] AS INT),
    CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[3] AS INT),
    CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[4] AS INT), 0) AS instante_coleta_local,
  CASE
    WHEN canal IS NULL AND trim(current_power) = '-' THEN 'id_sem_canal_e_sem_leitura'
    WHEN canal IS NULL THEN 'id_sem_canal'
    WHEN trim(current_power) = '-' THEN 'equipamento_sem_leitura'
    WHEN potencia_w IS NULL THEN 'potencia_nao_numerica'
  END AS motivo_quarentena
FROM b
""")
display(spark.sql("""SELECT coalesce(canal, '(sem canal)') AS canal, coalesce(motivo_quarentena, 'aprovada') AS situacao,
                     count(*) AS linhas FROM bronze_classificada GROUP BY ALL ORDER BY ALL"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. silver.usina_geracao (canais A e B)

# COMMAND ----------

silver_df = spark.sql(f"""
WITH ok AS (SELECT * FROM bronze_classificada WHERE motivo_quarentena IS NULL),
a AS (
  SELECT nome_arquivo, id_microinversor,
         try_cast(substring_index(trim(grid_frequency), ' ', 1) AS DOUBLE) AS frequencia_hz,
         try_cast(substring_index(trim(temperature), ' ', 1) AS DOUBLE) AS temperatura_c,
         try_to_timestamp(trim(reporting_time), 'yyyy-MM-dd HH:mm:ss') AS reporting_time_local
  FROM ok WHERE canal = 'A'
)
SELECT ok.inverter_id AS id_equipamento, ok.id_microinversor, ok.canal,
       ok.instante_coleta_local,
       to_utc_timestamp(ok.instante_coleta_local, '{TZ}') AS instante_coleta_utc,
       ok.potencia_w,
       CASE WHEN ok.canal = 'A' THEN try_cast(substring_index(trim(ok.grid_voltage), ' ', 1) AS DOUBLE)
            ELSE try_cast(substring_index(trim(ok.grid_frequency), ' ', 1) AS DOUBLE) END AS tensao_v,
       a.frequencia_hz, a.temperatura_c, a.reporting_time_local,
       CASE WHEN ok.canal = 'A' THEN 'linha_propria' ELSE 'herdado_da_linha_A' END AS origem_leituras_inversor,
       abs(timestampdiff(MINUTE, a.reporting_time_local, ok.instante_coleta_local)) > 60 AS flag_reporting_time_divergente,
       ok.arquivo_origem, ok.arquivo_dia,
       current_timestamp() AS atualizado_em
FROM ok LEFT JOIN a ON ok.nome_arquivo = a.nome_arquivo AND ok.id_microinversor = a.id_microinversor
""")
silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{SILVER}.usina_geracao")

quarentena_df = spark.sql("""
SELECT inverter_id, id_microinversor, canal, instante_coleta_local, motivo_quarentena,
       indice, current_power, grid_frequency, grid_voltage, temperature, reporting_time,
       arquivo_origem, arquivo_dia, current_timestamp() AS registrado_em
FROM bronze_classificada WHERE motivo_quarentena IS NOT NULL
""")
quarentena_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{SILVER}.usina_geracao_quarentena")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conferencia: bronze = silver + quarentena

# COMMAND ----------

import json
c = spark.sql(f"""
SELECT (SELECT count(*) FROM {BRONZE}) AS bronze,
       (SELECT count(*) FROM {SILVER}.usina_geracao) AS silver,
       (SELECT count(*) FROM {SILVER}.usina_geracao_quarentena) AS quarentena,
       (SELECT count_if(canal = 'A') FROM {SILVER}.usina_geracao) AS silver_canal_a,
       (SELECT count_if(canal = 'B') FROM {SILVER}.usina_geracao) AS silver_canal_b,
       (SELECT round(sum(potencia_w) * 5 / 60 / 1000, 1) FROM {SILVER}.usina_geracao) AS energia_total_kwh,
       (SELECT round(sum(CASE WHEN canal = 'B' THEN potencia_w END) * 5 / 60 / 1000, 1) FROM {SILVER}.usina_geracao) AS energia_canal_b_kwh,
       (SELECT count(*) - count(DISTINCT id_equipamento, instante_coleta_local) FROM {SILVER}.usina_geracao) AS chaves_duplicadas,
       (SELECT count_if(flag_reporting_time_divergente) FROM {SILVER}.usina_geracao WHERE canal = 'A') AS linhas_a_reporting_divergente
""").collect()[0].asDict()
c["fecha"] = c["bronze"] == c["silver"] + c["quarentena"]
print(json.dumps(c, indent=1))
motivos = {r["motivo_quarentena"]: r["n"] for r in spark.sql(
    f"SELECT motivo_quarentena, count(*) n FROM {SILVER}.usina_geracao_quarentena GROUP BY 1").collect()}
print(motivos)
assert c["fecha"] and c["chaves_duplicadas"] == 0
dbutils.notebook.exit(json.dumps({"conferencia": c, "quarentena_por_motivo": motivos}))
