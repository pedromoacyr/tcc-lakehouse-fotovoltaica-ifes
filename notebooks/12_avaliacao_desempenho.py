# Databricks notebook source
# MAGIC %md
# MAGIC # Avaliacao de desempenho (armazenamento, pequenos arquivos, consultas e escalabilidade)
# MAGIC
# MAGIC Protocolo:
# MAGIC 1. **Armazenamento com o mesmo conteudo**: os 63.501 CSVs avulsos do
# MAGIC    artefato `raw_csv/` (residuo de uma extracao interrompida, ~72% dos
# MAGIC    88.010 arquivos) sao comparados com uma tabela Delta que contem
# MAGIC    exatamente as mesmas linhas, e nao com a bronze inteira.
# MAGIC 2. **Leitura e consulta com o mesmo conteudo**: a fatia Delta comparada aos
# MAGIC    CSVs de um dia e filtrada pelos mesmos arquivos .
# MAGIC 3. **Partida fria aproximada**: como o compute serverless nao permite
# MAGIC    limpar o cache, a primeira execucao de cada medicao e registrada a parte
# MAGIC    (aproximacao de partida fria) e as nove seguintes dao media, mediana e
# MAGIC    desvio padrao (partida quente).
# MAGIC 4. **Escalabilidade materializada**: a bronze e replicada em Delta como N
# MAGIC    usinas sinteticas (identificadores distintos) e o processamento
# MAGIC    bronze -> silver -> gold (energia por canal e hora) e executado por
# MAGIC    completo em cada escala.
# MAGIC 5. **Linha de base sem Spark**: a mesma leitura e consulta com pandas e
# MAGIC    DuckDB no proprio no do driver, para dimensionar o custo fixo do Spark
# MAGIC    nesta escala de dados (Redondo; Junior, 2025).
# MAGIC
# MAGIC Tudo o que e materializado vai para o esquema descartavel
# MAGIC `fotovoltaica_ifes.teste_parecer`; a producao so e lida.

# COMMAND ----------

# MAGIC %pip install duckdb==1.1.3 --quiet

# COMMAND ----------

import time, statistics, json, os, glob, io, zipfile
import pandas as pd
BRONZE, SILVER, GOLD, TESTE = ("fotovoltaica_ifes.bronze", "fotovoltaica_ifes.silver", "fotovoltaica_ifes.gold",
                               "fotovoltaica_ifes.teste_parecer")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TESTE}")
RAW = "/Volumes/fotovoltaica_ifes/bronze/landing_usina/raw_csv"
res = {}


def medir(nome, fn, n=10):
    tempos = []
    for _ in range(n):
        t0 = time.time(); fn(); tempos.append(time.time() - t0)
    quente = tempos[1:]
    r = {"medicao": nome, "primeira_execucao_s": tempos[0], "media_quente_s": statistics.mean(quente),
         "mediana_quente_s": statistics.median(quente), "desvio_quente_s": statistics.stdev(quente), "n_quentes": len(quente)}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    return r


def tamanho(tabela):
    d = spark.sql(f"DESCRIBE DETAIL {tabela}").collect()[0]
    return int(d["sizeInBytes"]), int(d["numFiles"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Armazenamento

# COMMAND ----------

arquivos = dbutils.fs.ls(RAW + "/")
bytes_csv = sum(f.size for f in arquivos)
nomes = spark.createDataFrame([(f.name,) for f in arquivos], ["nome"])
nomes.createOrReplaceTempView("nomes_raw")
spark.sql(f"""CREATE OR REPLACE TABLE {TESTE}.bronze_mesmo_conteudo_raw_csv AS
              SELECT b.* FROM {BRONZE}.usina_geracao b JOIN nomes_raw n ON element_at(split(b.arquivo_origem, '/'), -1) = n.nome""")
b_mesmo, f_mesmo = tamanho(f"{TESTE}.bronze_mesmo_conteudo_raw_csv")
linhas_mesmo = spark.table(f"{TESTE}.bronze_mesmo_conteudo_raw_csv").count()
with zipfile.ZipFile("/Volumes/fotovoltaica_ifes/bronze/landing_usina/medicoes_usina.zip") as z:
    zip_bytes = os.path.getsize("/Volumes/fotovoltaica_ifes/bronze/landing_usina/medicoes_usina.zip")
camadas = {}
for sch in ["bronze", "silver", "gold"]:
    tot = 0
    for t in spark.sql(f"SHOW TABLES IN fotovoltaica_ifes.{sch}").collect():
        if not t["isTemporary"]:
            tot += tamanho(f"fotovoltaica_ifes.{sch}.{t['tableName']}")[0]
    camadas[sch] = tot
res["armazenamento"] = {"csv_avulsos_arquivos": len(arquivos), "csv_avulsos_bytes": bytes_csv,
                        "csv_avulsos_fracao_dos_88010": round(len(arquivos) / 88010, 4),
                        "delta_mesmo_conteudo_bytes": b_mesmo, "delta_mesmo_conteudo_arquivos": f_mesmo,
                        "delta_mesmo_conteudo_linhas": linhas_mesmo, "razao_csv_delta": round(bytes_csv / b_mesmo, 1),
                        "zip_original_bytes": zip_bytes, "bytes_por_camada": camadas}
print(json.dumps(res["armazenamento"], indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Leitura de pequenos arquivos e consulta representativa (mesmo conteudo)
# MAGIC Subamostra: os CSVs avulsos de um dia (`10_12_2020_*.csv`) e, na Delta, as
# MAGIC linhas desses mesmos arquivos.

# COMMAND ----------

PADRAO = "10_12_2020_"
dia = [f for f in arquivos if f.name.startswith(PADRAO)]
esquema = "indice STRING, inverter_id STRING, current_power STRING, grid_frequency STRING, grid_voltage STRING, temperature STRING, reporting_time STRING"
csv_dia = lambda: spark.read.format("csv").option("header", "true").schema(esquema).load(f"{RAW}/{PADRAO}*.csv")
spark.sql(f"""CREATE OR REPLACE TABLE {TESTE}.bronze_dia_subamostra AS SELECT * FROM {TESTE}.bronze_mesmo_conteudo_raw_csv
              WHERE element_at(split(arquivo_origem, '/'), -1) LIKE '{PADRAO}%'""")
n_csv, n_delta = csv_dia().count(), spark.table(f"{TESTE}.bronze_dia_subamostra").count()
res["subamostra"] = {"arquivos": len(dia), "linhas_csv": n_csv, "linhas_delta": n_delta, "fracao_corpus": round(len(dia) / len(arquivos), 4)}
print(res["subamostra"])

leitura = [medir("leitura CSV avulsos (1 dia)", lambda: csv_dia().count()),
           medir("leitura Delta mesmo conteudo", lambda: spark.table(f"{TESTE}.bronze_dia_subamostra").count())]
pot = "avg(try_cast(substring_index(trim(current_power), ' ', 1) AS DOUBLE))"
nomes_dia = [f.name for f in dia]
spark.createDataFrame([(n,) for n in nomes_dia], ["nome"]).createOrReplaceTempView("nomes_dia")
consulta = [medir("consulta CSV bruto", lambda: csv_dia().selectExpr(f"{pot} p").collect()),
            medir("consulta silver", lambda: spark.sql(f"""SELECT avg(potencia_w) FROM {SILVER}.usina_geracao s JOIN nomes_dia n
                                                         ON element_at(split(s.arquivo_origem, '/'), -1) = n.nome""").collect()),
            medir("consulta gold", lambda: spark.sql(f"""SELECT avg(potencia_w) FROM {GOLD}.fato_potencia_5min s JOIN nomes_dia n
                                                       ON element_at(split(s.arquivo_origem, '/'), -1) = n.nome""").collect())]

# Linha de base sem Spark, no no do driver (os arquivos sao lidos do Volume)
import duckdb
def pandas_dia():
    dfs = [pd.read_csv(f"{RAW}/{n}", names=["indice", "inverter_id", "current_power", "grid_frequency", "grid_voltage", "temperature", "reporting_time"], header=0, dtype=str) for n in nomes_dia]
    d = pd.concat(dfs)
    return pd.to_numeric(d.current_power.str.split(" ").str[0], errors="coerce").mean()
LISTA_DIA = [f"{RAW}/{n}" for n in nomes_dia]   # lista explicita: evita varrer por glob os 63 mil arquivos do Volume
def duck_dia():
    lista = "[" + ", ".join(f"'{x}'" for x in LISTA_DIA) + "]"
    return duckdb.sql(f"""SELECT avg(try_cast(split_part(trim(current_power), ' ', 1) AS DOUBLE))
                          FROM read_csv({lista}, header=false, skip=1, auto_detect=false, null_padding=true,
                                        columns={{'indice':'VARCHAR','inverter_id':'VARCHAR','current_power':'VARCHAR',
                                                  'grid_frequency':'VARCHAR','grid_voltage':'VARCHAR','temperature':'VARCHAR',
                                                  'reporting_time':'VARCHAR'}})""").fetchall()
base = [medir("consulta pandas (driver, CSV)", pandas_dia), medir("consulta DuckDB (driver, CSV)", duck_dia)]
res["leitura_consulta"] = leitura + consulta + base
spark.createDataFrame(pd.DataFrame(res["leitura_consulta"])).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{GOLD}.avaliacao_desempenho_leitura")
spark.createDataFrame(pd.DataFrame([{"chave": k, "valor": json.dumps(v, default=str)} for k, v in res.items()])) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.avaliacao_desempenho_resumo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Escalabilidade materializada: N usinas sinteticas, bronze -> silver -> gold
# MAGIC Cada replica recebe um prefixo distinto no identificador do
# MAGIC microinversor (`u<k>_`), de modo que as chaves continuam unicas. Para cada
# MAGIC escala mede-se: a gravacao da bronze replicada, a transformacao silver
# MAGIC (mesma regra do notebook 03: classificacao, quarentena, canal A e B,
# MAGIC heranca das leituras do microinversor) e a agregacao gold de energia por
# MAGIC canal e hora (mesma regra do notebook 08).

# COMMAND ----------

SQL_SILVER = """
CREATE OR REPLACE TABLE {dst} AS
WITH b AS (
  SELECT *, element_at(split(arquivo_origem, '/'), -1) AS nome_arquivo,
         CASE WHEN inverter_id RLIKE '-[AB]$' THEN substring_index(inverter_id, '-', 1) END AS id_microinversor,
         CASE WHEN inverter_id RLIKE '-[AB]$' THEN right(inverter_id, 1) END AS canal,
         try_cast(substring_index(trim(current_power), ' ', 1) AS DOUBLE) AS potencia_w
  FROM {src}),
c AS (SELECT *, make_timestamp(CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[2] AS INT),
        CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[1] AS INT), CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[0] AS INT),
        CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[3] AS INT), CAST(split(regexp_replace(nome_arquivo, '\\\\.csv$', ''), '_')[4] AS INT), 0) AS instante
      FROM b WHERE canal IS NOT NULL AND potencia_w IS NOT NULL),
a AS (SELECT nome_arquivo, id_microinversor, try_cast(substring_index(trim(temperature), ' ', 1) AS DOUBLE) t FROM c WHERE canal = 'A')
SELECT c.inverter_id, c.id_microinversor, c.canal, c.instante, c.potencia_w, a.t AS temperatura_c
FROM c LEFT JOIN a USING (nome_arquivo, id_microinversor)
"""
SQL_GOLD = """
CREATE OR REPLACE TABLE {dst} AS
SELECT inverter_id, date_trunc('hour', instante) hora, count(*) n, avg(potencia_w) energia_wh, avg(temperatura_c) t
FROM {src} GROUP BY 1, 2
"""
escalas, linhas_esc = [1, 10, 100], []
for k in escalas:
    t0 = time.time()
    spark.sql(f"""CREATE OR REPLACE TABLE {TESTE}.esc_bronze AS
                  SELECT b.indice, concat('u', r.id, '_', b.inverter_id) AS inverter_id, b.current_power, b.grid_frequency,
                         b.grid_voltage, b.temperature, b.reporting_time, b.arquivo_origem, b.arquivo_dia
                  FROM {BRONZE}.usina_geracao b CROSS JOIN (SELECT explode(sequence(1, {k})) AS id) r""")
    t_b = time.time() - t0
    t0 = time.time(); spark.sql(SQL_SILVER.format(src=f"{TESTE}.esc_bronze", dst=f"{TESTE}.esc_silver")); t_s = time.time() - t0
    t0 = time.time(); spark.sql(SQL_GOLD.format(src=f"{TESTE}.esc_silver", dst=f"{TESTE}.esc_gold")); t_g = time.time() - t0
    nb = spark.table(f"{TESTE}.esc_bronze").count(); ns = spark.table(f"{TESTE}.esc_silver").count(); ng = spark.table(f"{TESTE}.esc_gold").count()
    tb = tamanho(f"{TESTE}.esc_bronze")[0]
    linhas_esc.append({"escala": k, "linhas_bronze": nb, "linhas_silver": ns, "linhas_gold": ng, "bytes_bronze": tb,
                       "t_bronze_s": t_b, "t_silver_s": t_s, "t_gold_s": t_g, "t_silver_gold_s": t_s + t_g,
                       "segundos_por_milhao_linhas_bronze": (t_s + t_g) / (nb / 1e6)})
    print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in linhas_esc[-1].items()}))
    # grava a cada escala, para nao perder o resultado se uma escala maior for interrompida
    spark.createDataFrame(pd.DataFrame(linhas_esc)).write.mode("overwrite").option("overwriteSchema", "true") \
         .saveAsTable(f"{GOLD}.avaliacao_desempenho_escalabilidade")
res["escalabilidade"] = linhas_esc
for t in ["esc_bronze", "esc_silver", "esc_gold"]:
    spark.sql(f"DROP TABLE IF EXISTS {TESTE}.{t}")

# COMMAND ----------

pd_l = pd.DataFrame(res["leitura_consulta"]); pd_e = pd.DataFrame(res["escalabilidade"])
display(pd_l); display(pd_e)
dbutils.notebook.exit(json.dumps(res, default=str))
