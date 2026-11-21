# Databricks notebook source
# MAGIC %md
# MAGIC # Fontes complementares de irradiancia -> landing + bronze
# MAGIC
# MAGIC Tres fontes publicas de irradiacao, cada uma gravada
# MAGIC primeiro na landing (`/Volumes/fotovoltaica_ifes/bronze/landing_incidencia_solar/`,
# MAGIC arquivo bruto preservado) e depois em uma tabela bronze propria, com todas
# MAGIC as colunas como texto:
# MAGIC
# MAGIC | Fonte | O que traz | Uso |
# MAGIC |---|---|---|
# MAGIC | NASA POWER horario (UTC) | GHI, DNI e DHI com nuvens e em ceu claro, T2M, vento a 2 m | Transposicao para o plano dos modulos e temperatura de celula |
# MAGIC | Atlas Brasileiro de Energia Solar 2017 (LABREN/INPE) | Medias mensais por celula de 0,1 grau: global horizontal, difusa, direta normal e plano inclinado na latitude | Referencia de longo prazo e energia esperada com o Atlas |
# MAGIC | INMET, estacao automatica A612 (Vitoria) | Radiacao global, precipitacao e temperatura horarias medidas em solo | Validacao da irradiancia estimada e teste da hipotese de sujidade |
# MAGIC
# MAGIC Reprocessamento idempotente: os arquivos da landing sao sobrescritos e as
# MAGIC tabelas bronze sao reescritas por completo a cada execucao.

# COMMAND ----------

import urllib.request, urllib.parse, io, zipfile, csv, json, time, datetime as dt
from pyspark.sql.functions import lit

LANDING = "/Volumes/fotovoltaica_ifes/bronze/landing_incidencia_solar"
BRONZE = "fotovoltaica_ifes.bronze"
LAT, LON = -20.31, -40.32          # mesmo ponto ja usado na serie NASA original
INICIO, FIM = dt.date(2020, 11, 27), dt.date(2022, 2, 15)

def baixar(url, tentativas=4, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "tcc-lakehouse-solar/1.0"})
    for n in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  tentativa {n} falhou: {e}")
            if n == tentativas:
                raise
            time.sleep(5 * n)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) NASA POWER horario com componentes (UTC)
# MAGIC Parametros: `ALLSKY_SFC_SW_DWN` (GHI), `ALLSKY_SFC_SW_DNI` (DNI),
# MAGIC `ALLSKY_SFC_SW_DIFF` (DHI), as tres equivalentes em ceu claro
# MAGIC (`CLRSKY_*`), `T2M` (temperatura do ar a 2 m) e `WS2M` (vento a 2 m).
# MAGIC Segundo a documentacao do POWER, o carimbo horario marca o inicio da hora
# MAGIC e o valor vale para a hora inteira. Baixado em blocos de 90 dias.

# COMMAND ----------

PARAMS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
          "CLRSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DNI", "CLRSKY_SFC_SW_DIFF", "T2M", "WS2M"]
linhas, cabecalho, meta = [], None, None
ini = INICIO
while ini <= FIM:
    fim = min(ini + dt.timedelta(days=89), FIM)
    q = urllib.parse.urlencode({
        "parameters": ",".join(PARAMS), "community": "RE", "longitude": LON, "latitude": LAT,
        "start": ini.strftime("%Y%m%d"), "end": fim.strftime("%Y%m%d"),
        "format": "CSV", "time-standard": "UTC"})
    txt = baixar("https://power.larc.nasa.gov/api/temporal/hourly/point?" + q).decode("utf-8", "replace")
    pre, corpo = txt.split("-END HEADER-", 1)
    meta = meta or pre
    rows = list(csv.reader(io.StringIO(corpo.strip())))
    cabecalho = rows[0]
    linhas += rows[1:]
    print(ini, fim, len(rows) - 1)
    ini = fim + dt.timedelta(days=1)

nome = "nasa_power_horario_componentes_vitoria.csv"
with open(f"{LANDING}/{nome}", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(cabecalho); w.writerows(linhas)
with open(f"{LANDING}/nasa_power_horario_componentes_vitoria_meta.txt", "w", encoding="utf-8") as f:
    f.write(meta.replace("-BEGIN HEADER-", "").strip())
print("Cabecalho:", cabecalho, "| linhas:", len(linhas))
print(meta)

df = spark.createDataFrame(linhas, cabecalho).withColumn("arquivo_origem", lit(nome))
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
  .saveAsTable(f"{BRONZE}.incidencia_nasa_horario_componentes")
print("bronze.incidencia_nasa_horario_componentes:", spark.table(f"{BRONZE}.incidencia_nasa_horario_componentes").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Atlas Brasileiro de Energia Solar 2017 (LABREN/INPE)
# MAGIC Quatro camadas em CSV, todas com a mesma grade de 0,1 grau: global
# MAGIC horizontal, difusa, direta normal e plano inclinado na latitude. A camada
# MAGIC usada antes (Geobases-ES) era apenas a direta normal. Grava os zips
# MAGIC originais na landing e, na bronze, as celulas num raio de 0,5 grau da
# MAGIC usina (todas as colunas como texto).

# COMMAND ----------

CAMADAS = {
    "global_horizontal": "GLOBAL_HORIZONTAL_(csv).zip",
    "difusa": "DIFFUSE_(csv).zip",
    "direta_normal": "DIRECT_NORMAL_(csv).zip",
    "inclinada_latitude": "TILTED_LATITUDE_(csv).zip",
}
registros, cab_atlas, resumo = [], None, {}
for camada, arq in CAMADAS.items():
    b = baixar("http://labren.ccst.inpe.br/projetos/atlas_2017/" + urllib.parse.quote(arq))
    with open(f"{LANDING}/atlas_labren_2017_{camada}.zip", "wb") as f:
        f.write(b)
    z = zipfile.ZipFile(io.BytesIO(b))
    nome_csv = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    texto = z.read(nome_csv).decode("latin-1")
    dialeto = csv.Sniffer().sniff(texto[:2000], delimiters=";,")
    rows = list(csv.reader(io.StringIO(texto), dialeto))
    cab = [c.strip().upper() for c in rows[0]]
    ilon, ilat = cab.index("LON"), cab.index("LAT")
    perto = [r for r in rows[1:] if r and abs(float(r[ilat]) - LAT) <= 0.5 and abs(float(r[ilon]) - LON) <= 0.5]
    cab_atlas = cab_atlas or cab
    assert cab == cab_atlas, (cab, cab_atlas)
    registros += [[camada, nome_csv] + r for r in perto]
    resumo[camada] = {"arquivo": nome_csv, "celulas_total": len(rows) - 1, "celulas_raio_0_5": len(perto)}
print(json.dumps(resumo, indent=1))
print("Cabecalho Atlas:", cab_atlas)

df_atlas = spark.createDataFrame(registros, ["camada", "arquivo_csv"] + [c.lower() for c in cab_atlas]) \
    .withColumn("arquivo_origem", lit("atlas_labren_2017_<camada>.zip"))
df_atlas.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{BRONZE}.incidencia_atlas_labren")
print("bronze.incidencia_atlas_labren:", spark.table(f"{BRONZE}.incidencia_atlas_labren").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) INMET - estacao automatica A612 (Vitoria)
# MAGIC O INMET publica um zip anual com todas as estacoes; extrai-se apenas o
# MAGIC arquivo da A612 de 2020, 2021 e 2022. O arquivo tem 8 linhas de
# MAGIC metadados (regiao, UF, estacao, codigo, latitude, longitude, altitude,
# MAGIC data de fundacao), separador `;`, decimal com virgula e codificacao
# MAGIC latin-1; os metadados sao preservados na landing.

# COMMAND ----------

reg_inmet, cab_inmet, metas = [], None, {}
for ano in (2020, 2021, 2022):
    z = zipfile.ZipFile(io.BytesIO(baixar(f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip")))
    nome_csv = [n for n in z.namelist() if "_A612_" in n][0]
    raw = z.read(nome_csv)
    nome_local = f"inmet_A612_vitoria_{ano}.csv"
    with open(f"{LANDING}/{nome_local}", "wb") as f:
        f.write(raw)
    linhas_txt = raw.decode("latin-1").splitlines()
    metas[ano] = linhas_txt[:8]
    rows = list(csv.reader(linhas_txt[8:], delimiter=";"))
    cab = [c.strip() for c in rows[0] if c.strip() != ""]
    cab_inmet = cab_inmet or cab
    reg_inmet += [r[:len(cab)] + [nome_local] for r in rows[1:] if len(r) >= len(cab)]
    print(ano, nome_csv, len(rows) - 1)
print(json.dumps({str(k): v for k, v in metas.items()}, ensure_ascii=False, indent=1))

colunas = [f"c{i:02d}" for i in range(len(cab_inmet))] + ["arquivo_origem"]
df_inmet = spark.createDataFrame(reg_inmet, colunas)
dic = spark.createDataFrame([(f"c{i:02d}", c) for i, c in enumerate(cab_inmet)], ["coluna", "nome_original"])
df_inmet.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{BRONZE}.incidencia_inmet_a612")
dic.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{BRONZE}.incidencia_inmet_a612_colunas")
print("bronze.incidencia_inmet_a612:", spark.table(f"{BRONZE}.incidencia_inmet_a612").count())
display(dic)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "nasa_linhas": len(linhas), "nasa_cabecalho": cabecalho,
    "atlas": resumo, "atlas_cabecalho": cab_atlas,
    "inmet_linhas": len(reg_inmet), "inmet_cabecalho": cab_inmet, "inmet_meta": {str(k): v for k, v in metas.items()},
}, ensure_ascii=False))
