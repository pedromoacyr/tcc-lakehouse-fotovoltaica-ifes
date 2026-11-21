# Databricks notebook source
# MAGIC %md
# MAGIC # Auto Loader - ingestao incremental da usina (landing -> bronze), com MERGE
# MAGIC
# MAGIC Processa apenas os CSVs novos depositados na pasta de pouso incremental
# MAGIC (o checkpoint do Auto Loader registra o que ja foi lido) e grava na mesma
# MAGIC tabela bronze da carga historica.
# MAGIC
# MAGIC **Carga idempotente.** A gravacao usa `MERGE` na chave natural da medicao (microinversor e
# MAGIC canal + carimbo da coleta, extraido do nome do arquivo), via
# MAGIC `foreachBatch`. Assim, mesmo que um arquivo ja carregado volte a ser
# MAGIC depositado (por exemplo, uma reexportacao do sistema de monitoramento),
# MAGIC nenhuma linha e duplicada.
# MAGIC
# MAGIC **Parametros** (widgets): `pasta_pouso`, `checkpoint` e `tabela_destino`,
# MAGIC com os valores de producao como padrao, para permitir o teste num
# MAGIC esquema separado, com checkpoint proprio.
# MAGIC
# MAGIC **Nao usar para o historico:** a pasta `landing_usina/raw_csv/` guarda
# MAGIC arquivos de uma primeira tentativa de extracao, abandonada por desempenho.

# COMMAND ----------

dbutils.widgets.text("pasta_pouso", "/Volumes/fotovoltaica_ifes/bronze/landing_usina/incremental")
dbutils.widgets.text("checkpoint", "/Volumes/fotovoltaica_ifes/bronze/_checkpoints/usina_autoloader")
dbutils.widgets.text("tabela_destino", "fotovoltaica_ifes.bronze.usina_geracao")
PASTA = dbutils.widgets.get("pasta_pouso")
CHECKPOINT = dbutils.widgets.get("checkpoint")
TABELA = dbutils.widgets.get("tabela_destino")

import os, time, json
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, regexp_extract, concat, lit

os.makedirs(PASTA, exist_ok=True)
esquema_csv = StructType([StructField(c, StringType()) for c in [
    "indice", "inverter_id", "current_power", "grid_frequency", "grid_voltage", "temperature", "reporting_time"]])
print("Pasta de pouso:", PASTA, "| checkpoint:", CHECKPOINT, "| destino:", TABELA)

# COMMAND ----------

fluxo = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", "true")
    .option("cloudFiles.schemaLocation", CHECKPOINT + "/schema")
    .schema(esquema_csv)
    .load(PASTA)
    .withColumn("arquivo_origem", col("_metadata.file_name"))
    # mesmo formato de arquivo_dia da carga historica: "Medições/<d>_<m>_<aaaa>.tar.gz"
    .withColumn("arquivo_dia", concat(lit("Medições/"),
                regexp_extract(col("_metadata.file_name"), r"^(\d+_\d+_\d+)_", 1), lit(".tar.gz")))
)


def gravar_lote(df_lote, id_lote):
    df_lote.createOrReplaceTempView("lote_incremental")
    df_lote.sparkSession.sql(f"""
        MERGE INTO {TABELA} AS alvo
        USING lote_incremental AS novo
        ON element_at(split(alvo.arquivo_origem, '/'), -1) = element_at(split(novo.arquivo_origem, '/'), -1)
           AND alvo.inverter_id = novo.inverter_id
        WHEN NOT MATCHED THEN INSERT *
    """)


antes = spark.table(TABELA).count()
inicio = time.time()
(fluxo.writeStream.foreachBatch(gravar_lote)
     .option("checkpointLocation", CHECKPOINT)
     .trigger(availableNow=True)
     .start()
     .awaitTermination())
duracao = time.time() - inicio
depois = spark.table(TABELA).count()

resumo = {"linhas_antes": antes, "linhas_depois": depois, "linhas_inseridas": depois - antes,
          "duracao_s": round(duracao, 1)}
print(json.dumps(resumo, indent=1))
dbutils.notebook.exit(json.dumps(resumo))
