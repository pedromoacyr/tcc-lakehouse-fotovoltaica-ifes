# Databricks notebook source
# MAGIC %md
# MAGIC # Camada gold - dimensoes (dim_usina, dim_equipamento, dim_tempo)
# MAGIC
# MAGIC Modelagem:
# MAGIC 1. `dim_equipamento` passa a ter o grao de **canal** de microinversor
# MAGIC    (8 linhas: 4 microinversores x canais A e B), com os parametros de
# MAGIC    ficha tecnica do modulo ligado a cada canal; o modelo de energia
# MAGIC    esperada e calculado canal a canal, com o modulo e a estrutura (fixa ou
# MAGIC    rastreador) de cada um.
# MAGIC 2. Os parametros de geometria (coordenadas, inclinacao, azimute, eixo do
# MAGIC    rastreador) e as escolhas de modelo (transposicao, temperatura de
# MAGIC    celula) ficam registrados em `silver.config_parametros`, com status e
# MAGIC    fonte, como os demais parametros do pipeline.
# MAGIC
# MAGIC Fontes dos parametros dos equipamentos: Relatorio de comissionamento da
# MAGIC usina (Tabela 1, conexoes entre microinversores e modulos) e fichas
# MAGIC tecnicas TALESUN TP660P-275, YINGLI YL245P-29b e APsystems YC500A.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
SILVER, GOLD = "fotovoltaica_ifes.silver", "fotovoltaica_ifes.gold"
ID_USINA = "ifes_campus_vitoria"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Parametros de geometria e de modelo (`silver.config_parametros`)

# COMMAND ----------

novos = [
    ("latitude_usina", "-20.31", "provisorio", "Coordenada aproximada do Campus Vitoria, a mesma usada na extracao do NASA POWER. Pendente: coordenada exata da laje da usina."),
    ("longitude_usina", "-40.32", "provisorio", "Coordenada aproximada do Campus Vitoria, a mesma usada na extracao do NASA POWER. Pendente: coordenada exata da laje da usina."),
    ("altitude_m", "20", "provisorio", "Altitude aproximada do campus (usada apenas no modelo de ceu claro). O NASA POWER informa 412,5 m, que e a altitude media da celula de grade, nao do local."),
    ("inclinacao_modulos_graus", "15", "confirmado_ficha", "Relatorio de comissionamento: modulos voltados para o norte com 15 graus em relacao a laje. Equivale a inclinacao absoluta apenas se a laje for horizontal (pendente de confirmacao)."),
    ("azimute_modulos_graus", "0", "confirmado_ficha", "Relatorio de comissionamento: modulos voltados para o norte (azimute 0, convencao pvlib: 0 = norte, 90 = leste). A forma do perfil diario em dias claros e compativel com um desvio de 10 a 20 graus para oeste ou com sombreamento a leste no inicio da manha (pendente de confirmacao)."),
    ("rastreador_inclinacao_eixo_graus", "15", "provisorio", "Rastreador de 1 eixo norte-sul, com o eixo inclinado 15 graus para o norte (posicao de referencia do relatorio). Pendente: confirmar a geometria do eixo."),
    ("rastreador_azimute_eixo_graus", "0", "provisorio", "Eixo do rastreador alinhado ao norte geografico (pendente de confirmacao)."),
    ("rastreador_angulo_maximo_graus", "45", "provisorio", "Angulo maximo de rotacao do rastreador, sem backtracking. Pendente: valor do controlador. Testados 15, 30, 45, 60 e 90 graus; a tarde, o ganho medido do rastreador sobre os fixos fica entre os modelos de 30 e 45 graus."),
    ("modelo_transposicao", "perez", "decisao_metodologica", "Transposicao da irradiancia horizontal (GHI, DNI e DHI do NASA POWER) para o plano dos modulos pelo modelo anisotropico de Perez (pvlib). Hay-Davies e isotropico sao calculados como analise de sensibilidade."),
    ("modelo_temperatura_celula", "noct", "decisao_metodologica", "Temperatura de celula pelo modelo NOCT: Tc = Tar + (NOCT - 20) / 800 x G_plano, com a temperatura do ar a 2 m do NASA POWER. A temperatura reportada pelos microinversores e mantida como analise de sensibilidade (proxy), pois seu significado fisico ainda nao foi confirmado."),
    ("irradiancia_minima_aderencia_wh_m2", "50", "decisao_metodologica", "Horas com irradiancia no plano abaixo deste limiar (madrugada e crepusculo) ficam fora das metricas de aderencia."),
]
df = spark.createDataFrame(novos, ["parametro", "valor", "status", "observacao"]).withColumn("atualizado_em", current_timestamp())
df.createOrReplaceTempView("novos_parametros")
spark.sql(f"""MERGE INTO {SILVER}.config_parametros a USING novos_parametros n ON a.parametro = n.parametro
              WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")
P = {r["parametro"]: r["valor"] for r in spark.table(f"{SILVER}.config_parametros").collect()}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Parametros por canal (`silver.config_equipamentos`)

# COMMAND ----------

TALESUN = ("TALESUN TP660P-275", 275.0, 1.6236, 0.1694, -0.40, 45.0)
YINGLI = ("YINGLI YL245P-29b", 245.0, 1.6335, 0.1500, -0.45, 46.0)
SEM = (None, None, None, None, None, None)
canais = [
    # id_microinversor, numero, canal, estrutura, modulo
    ("403000261021", 1, "A", "rastreador", YINGLI), ("403000261021", 1, "B", "rastreador", YINGLI),
    ("403000261629", 2, "A", "fixa", TALESUN), ("403000261629", 2, "B", "fixa", TALESUN),
    ("403000262049", 3, "A", "fixa", TALESUN), ("403000262049", 3, "B", "sem_modulo", SEM),
    ("403000261008", 4, "A", "fixa", TALESUN), ("403000261008", 4, "B", "sem_modulo", SEM),
]
linhas = [(f"{mi}-{c}", ID_USINA, mi, n, c, est, *mod, 0.95, 250.0,
           "Relatorio de comissionamento (Tabela 1); fichas tecnicas do modulo e do microinversor APsystems YC500A "
           "(eficiencia ponderada CEC de 95%, 250 W de saida continua por canal).")
          for mi, n, c, est, mod in canais]
esquema = """id_equipamento STRING, id_usina STRING, id_microinversor STRING, numero_microinversor INT, canal STRING,
             estrutura STRING, modulo_modelo STRING, potencia_nominal_modulo_wp DOUBLE, area_modulo_m2 DOUBLE,
             eficiencia_modulo DOUBLE, coef_temp_potencia_pct_c DOUBLE, noct_c DOUBLE,
             eficiencia_pcu DOUBLE, potencia_saida_canal_w DOUBLE, fonte STRING"""
spark.createDataFrame(linhas, esquema).withColumn("atualizado_em", current_timestamp()) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SILVER}.config_equipamentos")
display(spark.table(f"{SILVER}.config_equipamentos"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Remocao das chaves antigas antes de recriar as dimensoes
# MAGIC As chaves estrangeiras sao removidas antes das primarias (o Unity Catalog
# MAGIC nao remove uma chave primaria com chaves estrangeiras apontando para ela).
# MAGIC O notebook 08 recria todas ao final.

# COMMAND ----------

cons = spark.sql("""SELECT table_name, constraint_name, constraint_type
                    FROM fotovoltaica_ifes.information_schema.table_constraints
                    WHERE table_schema = 'gold'""").collect()
for tipo in ("FOREIGN KEY", "PRIMARY KEY"):
    for c in cons:
        if c["constraint_type"] == tipo:
            spark.sql(f"ALTER TABLE {GOLD}.{c['table_name']} DROP CONSTRAINT IF EXISTS {c['constraint_name']}")
print(f"{len(cons)} chaves removidas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) dim_usina

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.dim_usina (
  id_usina STRING NOT NULL, nome_usina STRING, instituicao STRING, campus STRING,
  latitude DOUBLE, longitude DOUBLE, altitude_m DOUBLE, timezone STRING,
  capacidade_declarada_kwp DOUBLE, potencia_nominal_modulos_wp DOUBLE, n_modulos INT,
  inclinacao_modulos_graus DOUBLE, azimute_modulos_graus DOUBLE,
  rastreador_inclinacao_eixo_graus DOUBLE, rastreador_azimute_eixo_graus DOUBLE, rastreador_angulo_maximo_graus DOUBLE,
  performance_ratio_referencia DOUBLE, temperatura_padrao_c DOUBLE,
  modelo_transposicao STRING, modelo_temperatura_celula STRING,
  celula_atlas_id BIGINT, celula_atlas_distancia_km DOUBLE,
  estacao_inmet STRING, estacao_inmet_distancia_km DOUBLE,
  observacao STRING, atualizado_em TIMESTAMP
) USING DELTA COMMENT 'Uma linha por usina: localizacao, geometria, capacidade e escolhas de modelo.'
""")
cel = spark.sql(f"""SELECT id_celula, distancia_km FROM {SILVER}.incidencia_atlas_labren
                    WHERE camada = 'global_horizontal' AND mes = 1 ORDER BY distancia_km LIMIT 1""").collect()[0]
eq = spark.sql(f"SELECT sum(potencia_nominal_modulo_wp) wp, count(modulo_modelo) n FROM {SILVER}.config_equipamentos").collect()[0]
lat, lon = float(P["latitude_usina"]), float(P["longitude_usina"])
spark.sql(f"""
INSERT INTO {GOLD}.dim_usina SELECT
  '{ID_USINA}', 'Usina fotovoltaica do Campus Vitoria', 'Instituto Federal do Espirito Santo (Ifes)', 'Campus Vitoria',
  {lat}, {lon}, {float(P['altitude_m'])}, 'America/Sao_Paulo',
  {float(P['capacidade_instalada_kwp'])}, {eq['wp']}, {eq['n']},
  {float(P['inclinacao_modulos_graus'])}, {float(P['azimute_modulos_graus'])},
  {float(P['rastreador_inclinacao_eixo_graus'])}, {float(P['rastreador_azimute_eixo_graus'])}, {float(P['rastreador_angulo_maximo_graus'])},
  {float(P['performance_ratio'])}, {float(P['temperatura_padrao_c'])},
  '{P['modelo_transposicao']}', '{P['modelo_temperatura_celula']}',
  {cel['id_celula']}, {cel['distancia_km']},
  'INMET A612 (Vitoria)',
  round(2 * 6371 * asin(sqrt(pow(sin(radians(-20.27083332 - ({lat})) / 2), 2)
        + cos(radians({lat})) * cos(radians(-20.27083332)) * pow(sin(radians(-40.30583333 - ({lon})) / 2), 2))), 2),
  'Capacidade declarada no relatorio de comissionamento (1,51 kWp); a soma das potencias nominais dos 6 modulos e 1.590 Wp. O modelo de energia esperada usa os parametros de cada modulo (dim_equipamento), nao a capacidade declarada.',
  current_timestamp()
""")
display(spark.table(f"{GOLD}.dim_usina"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) dim_equipamento (grao: canal)
# MAGIC `inicio_geracao` e o primeiro dia em que o canal registrou potencia
# MAGIC positiva; para os canais B dos microinversores 1 e 2 ele marca o momento
# MAGIC em que o segundo modulo passou a gerar (antes disso o canal reportava 0 W
# MAGIC com sol, registrado no log de eventos pelo notebook 08b).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.dim_equipamento
COMMENT 'Um registro por canal de microinversor, com o modulo conectado e seus parametros de ficha tecnica.'
AS
SELECT c.id_equipamento, c.id_usina, c.id_microinversor, c.numero_microinversor, c.canal,
       'microinversor APsystems YC500A' AS tipo_equipamento, c.estrutura, c.modulo_modelo,
       c.potencia_nominal_modulo_wp, c.area_modulo_m2, c.eficiencia_modulo, c.coef_temp_potencia_pct_c, c.noct_c,
       c.eficiencia_pcu, c.potencia_saida_canal_w,
       s.n_leituras, s.primeira_leitura, s.ultima_leitura, s.inicio_geracao, s.potencia_maxima_observada_w,
       c.fonte, current_timestamp() AS atualizado_em
FROM {SILVER}.config_equipamentos c
LEFT JOIN (
  SELECT id_equipamento, count(*) n_leituras, min(instante_coleta_local) primeira_leitura,
         max(instante_coleta_local) ultima_leitura,
         min(CASE WHEN potencia_w > 0 THEN date(instante_coleta_local) END) inicio_geracao,
         max(potencia_w) potencia_maxima_observada_w
  FROM {SILVER}.usina_geracao GROUP BY 1) s USING (id_equipamento)
""")
spark.sql(f"ALTER TABLE {GOLD}.dim_equipamento ALTER COLUMN id_equipamento SET NOT NULL")
display(spark.table(f"{GOLD}.dim_equipamento").orderBy("id_equipamento"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) dim_tempo (grao: hora local, sequencia gerada sem lacunas)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.dim_tempo COMMENT 'Calendario horario continuo (horario local, UTC-3).' AS
WITH lim AS (
  SELECT least((SELECT min(date(instante_coleta_local)) FROM {SILVER}.usina_geracao),
               (SELECT min(date(hora_inicio_local)) FROM {SILVER}.incidencia_nasa_horario_componentes)) AS ini,
         greatest((SELECT max(date(instante_coleta_local)) FROM {SILVER}.usina_geracao),
                  (SELECT max(date(hora_inicio_local)) FROM {SILVER}.incidencia_nasa_horario_componentes)) AS fim)
SELECT CAST(date_format(h, 'yyyyMMddHH') AS BIGINT) AS id_tempo, h AS hora_local,
       to_utc_timestamp(h, 'America/Sao_Paulo') AS hora_utc, date(h) AS data,
       year(h) AS ano, month(h) AS mes, day(h) AS dia, hour(h) AS hora,
       CASE WHEN month(h) IN (12, 1, 2) THEN 'verao' WHEN month(h) IN (3, 4, 5) THEN 'outono'
            WHEN month(h) IN (6, 7, 8) THEN 'inverno' ELSE 'primavera' END AS estacao_meteorologica,
       date_format(h, 'yyyy-MM') AS mes_referencia, 'America/Sao_Paulo' AS timezone
FROM lim LATERAL VIEW explode(sequence(to_timestamp(ini), to_timestamp(date_add(fim, 1)) - INTERVAL 1 HOUR, INTERVAL 1 HOUR)) t AS h
""")
spark.sql(f"ALTER TABLE {GOLD}.dim_tempo ALTER COLUMN id_tempo SET NOT NULL")

import json
dbutils.notebook.exit(json.dumps({t: spark.table(f"{GOLD}.{t}").count() for t in ["dim_usina", "dim_equipamento", "dim_tempo"]}))
