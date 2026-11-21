# Databricks notebook source
# MAGIC %md
# MAGIC # Teste de idempotencia e de carga incremental (esquema de teste)
# MAGIC
# MAGIC Demonstra, sem tocar a producao, que a ingestao das medicoes e idempotente e que a carga
# MAGIC incremental funciona. Tudo roda no esquema `fotovoltaica_ifes.teste_parecer`,
# MAGIC com pasta de pouso e checkpoint do Auto Loader proprios.
# MAGIC
# MAGIC Roteiro:
# MAGIC 1. carga historica (notebook 01) **sem os ultimos 7 dias**, numa tabela vazia;
# MAGIC 2. deposito dos CSVs desses 7 dias na pasta de pouso de teste e carga
# MAGIC    incremental (notebook 02, Auto Loader);
# MAGIC 3. reexecucoes que nao devem inserir nada: notebook 02 sem arquivos novos,
# MAGIC    notebook 02 com um dia **redepositado** em outra subpasta (o checkpoint ve
# MAGIC    arquivos novos, mas o MERGE nao duplica) e notebook 01 de novo;
# MAGIC 4. comparacao linha a linha com a bronze de producao;
# MAGIC 5. `SHALLOW CLONE` da bronze de producao e notebook 01 completo sobre o
# MAGIC    clone (330 dias): nenhuma linha deve ser inserida.
# MAGIC
# MAGIC Os tempos de cada etapa sao registrados, inclusive o do notebook 01
# MAGIC completo sobre o clone.

# COMMAND ----------

import json, os, zipfile, tarfile, io, shutil, time, datetime as dt

ESQ = "fotovoltaica_ifes.teste_parecer"
PROD = "fotovoltaica_ifes.bronze.usina_geracao"
N = 7
spark.sql(f"DROP SCHEMA IF EXISTS {ESQ} CASCADE")
spark.sql(f"CREATE SCHEMA {ESQ} COMMENT 'Esquema descartavel para os testes de idempotencia e de carga incremental'")
spark.sql(f"CREATE VOLUME {ESQ}.landing")
POUSO = f"/Volumes/fotovoltaica_ifes/teste_parecer/landing/incremental"
CKPT = f"/Volumes/fotovoltaica_ifes/teste_parecer/landing/_checkpoint_autoloader"
DESTINO = f"{ESQ}.usina_geracao"
os.makedirs(POUSO, exist_ok=True)
resultado = []


def rodar(nb, args, etapa):
    t = time.time()
    saida = json.loads(dbutils.notebook.run(nb, 3600, args))
    saida["tempo_total_notebook_s"] = round(time.time() - t, 1)
    saida["etapa"] = etapa
    resultado.append(saida)
    print(etapa, "->", saida)
    return saida

# COMMAND ----------

# 1) carga historica sem os ultimos N dias (tabela vazia)
r1 = rodar("./01_bronze_extracao_usina", {"tabela_destino": DESTINO, "excluir_ultimos_dias": str(N)},
           "1. notebook 01: historico sem os ultimos 7 dias")

# COMMAND ----------

# 2) deposita os CSVs dos ultimos N dias na pasta de pouso de teste e roda o Auto Loader
zip_path = "/Volumes/fotovoltaica_ifes/bronze/landing_usina/medicoes_usina.zip"
with zipfile.ZipFile(zip_path) as z:
    membros = [n for n in z.namelist() if n.endswith(".tar.gz") and "__MACOSX" not in n]
    def data_m(n):
        d, m, a = n.split("/")[-1].replace(".tar.gz", "").split("_")
        return dt.date(int(a), int(m), int(d))
    ultimos = sorted(membros, key=data_m)[-N:]
    n_csv = 0
    for m in ultimos:
        with tarfile.open(fileobj=io.BytesIO(z.read(m)), mode="r:gz") as tf:
            for f in tf.getmembers():
                if f.isfile() and f.name.endswith(".csv"):
                    with open(f"{POUSO}/{os.path.basename(f.name)}", "wb") as out:
                        out.write(tf.extractfile(f).read())
                    n_csv += 1
print("dias depositados:", [str(data_m(m)) for m in ultimos], "| CSVs:", n_csv)
r2 = rodar("./02_bronze_autoloader_usina", {"pasta_pouso": POUSO, "checkpoint": CKPT, "tabela_destino": DESTINO},
           "2. notebook 02: carga incremental dos 7 dias")

# COMMAND ----------

# 3) reexecucoes que nao devem inserir nada
r3a = rodar("./02_bronze_autoloader_usina", {"pasta_pouso": POUSO, "checkpoint": CKPT, "tabela_destino": DESTINO},
            "3a. notebook 02 sem arquivos novos")
os.makedirs(f"{POUSO}/reenvio", exist_ok=True)
dia_reenvio = os.path.basename(ultimos[-1]).replace(".tar.gz", "")
copiados = 0
for f in os.listdir(POUSO):
    if f.startswith(dia_reenvio + "_") and f.endswith(".csv"):
        shutil.copy(f"{POUSO}/{f}", f"{POUSO}/reenvio/{f}")
        copiados += 1
print("dia redepositado:", dia_reenvio, "| CSVs:", copiados)
r3b = rodar("./02_bronze_autoloader_usina", {"pasta_pouso": POUSO, "checkpoint": CKPT, "tabela_destino": DESTINO},
            "3b. notebook 02 com um dia redepositado em outra subpasta")
r3c = rodar("./01_bronze_extracao_usina", {"tabela_destino": DESTINO, "excluir_ultimos_dias": str(N)},
            "3c. notebook 01 reexecutado")

# COMMAND ----------

# 4) comparacao linha a linha com a producao (colunas de dado + chave natural)
cols = "element_at(split(arquivo_origem, '/'), -1) AS arquivo, inverter_id, indice, current_power, grid_frequency, grid_voltage, temperature, reporting_time"
so_teste = spark.sql(f"SELECT {cols} FROM {DESTINO} EXCEPT ALL SELECT {cols} FROM {PROD}").count()
so_prod = spark.sql(f"SELECT {cols} FROM {PROD} EXCEPT ALL SELECT {cols} FROM {DESTINO}").count()
dup = spark.sql(f"SELECT count(*) - count(DISTINCT element_at(split(arquivo_origem,'/'),-1), inverter_id) AS d FROM {DESTINO}").collect()[0]["d"]
comparacao = {"linhas_teste": spark.table(DESTINO).count(), "linhas_producao": spark.table(PROD).count(),
              "linhas_so_no_teste": so_teste, "linhas_so_na_producao": so_prod, "chaves_duplicadas_teste": dup}
print(comparacao)

# COMMAND ----------

# 5) SHALLOW CLONE da producao + notebook 01 completo sobre o clone
spark.sql(f"CREATE TABLE {ESQ}.usina_geracao_clone SHALLOW CLONE {PROD}")
r5 = rodar("./01_bronze_extracao_usina", {"tabela_destino": f"{ESQ}.usina_geracao_clone"},
           "5. notebook 01 completo (330 dias) sobre SHALLOW CLONE da producao")

# COMMAND ----------

linhas = [(r["etapa"], int(r.get("linhas_inseridas_nesta_execucao", r.get("linhas_inseridas", -1))),
           float(r.get("duracao_s", -1)), float(r["tempo_total_notebook_s"]),
           int(r.get("linhas_tabela", r.get("linhas_depois", -1)))) for r in resultado]
df = spark.createDataFrame(linhas, "etapa STRING, linhas_inseridas INT, duracao_carga_s DOUBLE, tempo_total_notebook_s DOUBLE, linhas_na_tabela_apos INT")
df.write.mode("overwrite").saveAsTable(f"{ESQ}.resultado_teste_ingestao")
display(df)
dbutils.notebook.exit(json.dumps({"etapas": resultado, "comparacao": comparacao, "csv_depositados": n_csv,
                                  "csv_redepositados": copiados}, ensure_ascii=False))
