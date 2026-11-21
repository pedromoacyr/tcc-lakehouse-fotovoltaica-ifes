# Databricks notebook source
# MAGIC %md
# MAGIC # Extracao da usina -> bronze (Delta), com carga idempotente por MERGE
# MAGIC
# MAGIC Le os 330 arquivos diarios (`.tar.gz`) de dentro de `medicoes_usina.zip`,
# MAGIC faz o parse dos ~288 CSVs de cada dia em memoria e grava as linhas brutas
# MAGIC (todas as colunas como texto) em `bronze.usina_geracao`.
# MAGIC
# MAGIC **Carga idempotente.** Cada lote e gravado com `MERGE` na chave natural da
# MAGIC medicao (TCC, Secao 3.5.3), de modo que reexecutar o notebook nao duplica
# MAGIC linhas. A chave e formada por:
# MAGIC **microinversor e canal** (`inverter_id`, ex. `403000261021-A`) e **carimbo
# MAGIC de data e hora da coleta**, obtido do nome do arquivo (ex.
# MAGIC `15_7_2021_14_30.csv`). A unicidade dessa chave foi conferida na bronze
# MAGIC existente: 528.572 linhas, 528.572 combinacoes distintas.
# MAGIC
# MAGIC Alem das linhas, grava uma tabela de auditoria por arquivo CSV
# MAGIC (`<destino>_arquivos`), com o numero de linhas de dados de cada arquivo:
# MAGIC permite quantificar os arquivos que chegam so com cabecalho (achado (i)
# MAGIC da Secao 3.4.1).
# MAGIC
# MAGIC **Parametros** (widgets), para permitir o teste de idempotencia e de carga
# MAGIC incremental num esquema de teste sem tocar a producao:
# MAGIC - `tabela_destino`: padrao `fotovoltaica_ifes.bronze.usina_geracao`;
# MAGIC - `excluir_ultimos_dias`: carrega o historico sem os ultimos N dias (padrao 0);
# MAGIC - `somente_ultimos_dias`: carrega apenas os ultimos N dias (padrao 0).

# COMMAND ----------

dbutils.widgets.text("tabela_destino", "fotovoltaica_ifes.bronze.usina_geracao")
dbutils.widgets.text("excluir_ultimos_dias", "0")
dbutils.widgets.text("somente_ultimos_dias", "0")
TABELA = dbutils.widgets.get("tabela_destino")
EXCLUIR = int(dbutils.widgets.get("excluir_ultimos_dias"))
SOMENTE = int(dbutils.widgets.get("somente_ultimos_dias"))
TABELA_ARQ = TABELA + "_arquivos"

import zipfile, tarfile, io, csv, time, datetime as dt, json

zip_path = "/Volumes/fotovoltaica_ifes/bronze/landing_usina/medicoes_usina.zip"


def corrigir_nome_arquivo(s):
    """O zip nao marca os nomes como UTF-8 e o Python os decodifica como cp437
    ("Medições" vira "Medic╠ºo╠âes"); desfaz-se a decodificacao errada e
    normaliza-se em NFC, mesma regra de 01b_bronze_normaliza_unicode."""
    import unicodedata
    try:
        s = s.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFC", s)


with zipfile.ZipFile(zip_path) as z:
    membros = [n for n in z.namelist() if n.endswith(".tar.gz") and "__MACOSX" not in n]


def data_do_membro(n):
    d, m, a = n.split("/")[-1].replace(".tar.gz", "").split("_")
    return dt.date(int(a), int(m), int(d))


membros = sorted(membros, key=data_do_membro)
if EXCLUIR:
    membros = membros[:-EXCLUIR]
if SOMENTE:
    membros = membros[-SOMENTE:]
print(f"Destino: {TABELA} | arquivos diarios a carregar: {len(membros)} "
      f"({data_do_membro(membros[0])} a {data_do_membro(membros[-1])})")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from pyspark.sql.functions import udf, explode, col

esquema_linha = StructType([StructField(c, StringType()) for c in [
    "indice", "inverter_id", "current_power", "grid_frequency", "grid_voltage",
    "temperature", "reporting_time", "arquivo_origem", "arquivo_dia"]])
esquema_arquivo = StructType([
    StructField("arquivo_dia", StringType()), StructField("arquivo_origem", StringType()),
    StructField("n_linhas_dados", IntegerType())])
esquema_saida = StructType([
    StructField("linhas", ArrayType(esquema_linha)),
    StructField("arquivos", ArrayType(esquema_arquivo))])


def processar_dia(membro):
    """Le um .tar.gz (um dia) em memoria; devolve as linhas brutas e a
    contagem de linhas de dados de cada CSV. Nada e gravado no Volume."""
    with zipfile.ZipFile(zip_path) as z:
        dados = z.read(membro)
    dia = corrigir_nome_arquivo(membro)
    linhas, arquivos = [], []
    with tarfile.open(fileobj=io.BytesIO(dados), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".csv")):
                continue
            leitor = csv.reader(io.StringIO(tf.extractfile(m).read().decode("utf-8", errors="replace")))
            next(leitor, None)  # cabecalho
            n = 0
            for r in leitor:
                r = (r + [None] * 7)[:7]
                linhas.append((*r, m.name, dia))
                n += 1
            arquivos.append((dia, m.name, n))
    return (linhas, arquivos)


processar_dia_udf = udf(processar_dia, esquema_saida)

# COMMAND ----------

spark.sql(f"CREATE TABLE IF NOT EXISTS {TABELA} ({', '.join(f'{c} STRING' for c in esquema_linha.names)}) USING DELTA")
spark.sql(f"""CREATE TABLE IF NOT EXISTS {TABELA_ARQ} (arquivo_dia STRING, arquivo_origem STRING,
              n_linhas_dados INT) USING DELTA""")

CHAVE_LINHA = """element_at(split(alvo.arquivo_origem, '/'), -1) = element_at(split(novo.arquivo_origem, '/'), -1)
                 AND alvo.inverter_id = novo.inverter_id"""
CHAVE_ARQ = """element_at(split(alvo.arquivo_origem, '/'), -1) = element_at(split(novo.arquivo_origem, '/'), -1)"""

inicio = time.time()
lote, inseridas = 30, 0
for i in range(0, len(membros), lote):
    df = spark.createDataFrame([(m,) for m in membros[i:i + lote]], ["membro"]) \
        .withColumn("s", processar_dia_udf(col("membro")))
    df.select(explode("s.linhas").alias("r")).select("r.*").createOrReplaceTempView("novo_lote")
    df.select(explode("s.arquivos").alias("a")).select("a.*").createOrReplaceTempView("novo_lote_arq")
    r = spark.sql(f"MERGE INTO {TABELA} AS alvo USING novo_lote AS novo ON {CHAVE_LINHA} "
                  f"WHEN NOT MATCHED THEN INSERT *").collect()[0]
    spark.sql(f"MERGE INTO {TABELA_ARQ} AS alvo USING novo_lote_arq AS novo ON {CHAVE_ARQ} "
              f"WHEN NOT MATCHED THEN INSERT *")
    inseridas += r["num_inserted_rows"]
    print(f"Lote {i // lote + 1}: {len(membros[i:i + lote])} dias, {r['num_inserted_rows']} linhas novas "
          f"(acumulado {inseridas})")
duracao = time.time() - inicio

# COMMAND ----------

resumo = spark.sql(f"""
  SELECT (SELECT count(*) FROM {TABELA}) AS linhas_tabela,
         (SELECT count(DISTINCT arquivo_dia) FROM {TABELA}) AS dias_tabela,
         (SELECT count(*) FROM {TABELA_ARQ}) AS arquivos_csv,
         (SELECT count_if(n_linhas_dados = 0) FROM {TABELA_ARQ}) AS arquivos_so_cabecalho
""").collect()[0].asDict()
resumo.update({"linhas_inseridas_nesta_execucao": inseridas, "duracao_s": round(duracao, 1),
               "dias_carregados_nesta_execucao": len(membros)})
print(json.dumps(resumo, indent=1))
dbutils.notebook.exit(json.dumps(resumo))
