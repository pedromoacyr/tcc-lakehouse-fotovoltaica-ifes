# Databricks notebook source
# MAGIC %md
# MAGIC # Camada gold - log de eventos operacionais (`log_eventos`)
# MAGIC
# MAGIC Implementa o log de eventos previsto no escopo minimo (Secao 2.7 e
# MAGIC Quadros 4 e 5 da Metodologia). Como a usina nao dispoe de um registro de
# MAGIC manutencao, os eventos sao **derivados dos proprios dados**, por regras
# MAGIC explicitas, e agrupados em intervalos continuos (tecnica de "ilhas": uma
# MAGIC sequencia de registros consecutivos com a mesma condicao vira um evento,
# MAGIC com inicio, fim e duracao). Cada evento guarda a regra que o gerou.
# MAGIC
# MAGIC | Tipo | Regra | Grao |
# MAGIC |---|---|---|
# MAGIC | `dia_sem_arquivo` | dia do periodo sem nenhum arquivo diario recebido | usina, dia |
# MAGIC | `equipamento_sem_leitura` | microinversor com leitura `-` durante o dia (sol acima do horizonte) | microinversor, 5 min |
# MAGIC | `horario_reporte_congelado` | horario de reporte do microinversor parado durante o dia | microinversor, 5 min |
# MAGIC | `relogio_equipamento_divergente` | horario de reporte a mais de 60 min do instante de coleta, sem estar congelado | microinversor, dia |
# MAGIC | `canal_sem_geracao` | canal com modulo gerando menos de 5% do outro canal do mesmo microinversor | canal, dia |
# MAGIC | `rastreador_parado_referencia` / `rastreador_travado_oeste` / `rastreador_travado_leste` | estado diario do rastreador (notebook 08) | rastreador, dia |
# MAGIC | `dia_baixo_desempenho` | dia de ceu predominantemente claro (indice de claridade >= 0,6) com razao medida/esperada < 0,8 em >= 6 horas incluidas | usina, dia |
# MAGIC
# MAGIC O resultado e uma tabela que a camada gold oferece para diagnostico
# MAGIC operacional e para a mineracao de processos (trabalhos futuros), sem
# MAGIC precisar voltar aos arquivos brutos.

# COMMAND ----------

GOLD, SILVER = "fotovoltaica_ifes.gold", "fotovoltaica_ifes.silver"

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW diurno AS
SELECT hora_local FROM {GOLD}.fato_irradiancia_plano_hora WHERE zenite_solar_graus < 90
""")

# COMMAND ----------

eventos = spark.sql(f"""
WITH
-- 1) dias do periodo sem arquivo
dias AS (SELECT explode(sequence((SELECT min(date(instante_coleta_local)) FROM {SILVER}.usina_geracao),
                                 (SELECT max(date(instante_coleta_local)) FROM {SILVER}.usina_geracao))) AS data),
com_arquivo AS (SELECT DISTINCT to_date(regexp_extract(arquivo_dia, '([0-9]+_[0-9]+_[0-9]+)', 1), 'd_M_yyyy') AS data
                FROM fotovoltaica_ifes.bronze.usina_geracao_arquivos),
sem_arq AS (SELECT d.data, datediff(d.data, (SELECT min(data) FROM dias)) - row_number() OVER (ORDER BY d.data) AS ilha
            FROM dias d LEFT ANTI JOIN com_arquivo c USING (data)),
e1 AS (SELECT NULL AS id_equipamento, 'dia_sem_arquivo' AS tipo_evento, CAST(min(data) AS TIMESTAMP) AS inicio,
              CAST(date_add(max(data), 1) AS TIMESTAMP) AS fim, count(*) AS n_registros,
              concat(count(*), ' dia(s) sem arquivo diario recebido') AS descricao FROM sem_arq GROUP BY ilha),
-- 2) microinversor sem leitura durante o dia (quarentena da silver)
q AS (SELECT q.id_microinversor, q.instante_coleta_local AS t FROM {SILVER}.usina_geracao_quarentena q
      JOIN diurno d ON date_trunc('hour', q.instante_coleta_local) = d.hora_local
      WHERE q.motivo_quarentena = 'equipamento_sem_leitura' AND q.canal = 'A'),
q2 AS (SELECT *, CAST(unix_timestamp(t) / 300 AS BIGINT) - row_number() OVER (PARTITION BY id_microinversor ORDER BY t) AS ilha FROM q),
e2 AS (SELECT id_microinversor, 'equipamento_sem_leitura', min(t), max(t) + INTERVAL 5 MINUTES, count(*),
              concat('microinversor sem leitura durante o dia (', count(*), ' leituras de 5 min)') FROM q2 GROUP BY id_microinversor, ilha),
-- 3) horario de reporte congelado durante o dia
c AS (SELECT f.id_microinversor, f.instante_coleta_local AS t FROM {GOLD}.fato_potencia_5min f
      JOIN diurno d ON date_trunc('hour', f.instante_coleta_local) = d.hora_local
      WHERE f.canal = 'A' AND f.flag_horario_reporte_congelado),
c2 AS (SELECT *, CAST(unix_timestamp(t) / 300 AS BIGINT) - row_number() OVER (PARTITION BY id_microinversor ORDER BY t) AS ilha FROM c),
e3 AS (SELECT id_microinversor, 'horario_reporte_congelado', min(t), max(t) + INTERVAL 5 MINUTES, count(*),
              concat('horario de reporte parado por ', count(*), ' leituras consecutivas') FROM c2 GROUP BY id_microinversor, ilha),
-- 4) relogio do equipamento divergente (por dia, sem congelamento)
r AS (SELECT id_microinversor, date(instante_coleta_local) AS data,
             percentile(timestampdiff(MINUTE, instante_coleta_local, reporting_time_equipamento_local), 0.5) AS desvio_mediano_min
      FROM {GOLD}.fato_potencia_5min WHERE canal = 'A' AND flag_reporting_time_divergente AND NOT flag_horario_reporte_congelado
      GROUP BY 1, 2 HAVING count(*) >= 12),
r2 AS (SELECT *, datediff(data, DATE'2020-01-01') - row_number() OVER (PARTITION BY id_microinversor ORDER BY data) AS ilha FROM r),
e4 AS (SELECT id_microinversor, 'relogio_equipamento_divergente', CAST(min(data) AS TIMESTAMP), CAST(date_add(max(data), 1) AS TIMESTAMP), count(*),
              concat('horario de reporte adiantado/atrasado em mediana ', CAST(round(percentile(desvio_mediano_min, 0.5)) AS INT), ' min em relacao ao instante de coleta')
       FROM r2 GROUP BY id_microinversor, ilha),
-- 5) canal com modulo sem geracao
s AS (SELECT DISTINCT id_equipamento, date(hora_local) AS data FROM {GOLD}.fato_energia_canal_hora WHERE canal_sem_geracao_no_dia),
s2 AS (SELECT *, datediff(data, DATE'2020-01-01') - row_number() OVER (PARTITION BY id_equipamento ORDER BY data) AS ilha FROM s),
e5 AS (SELECT id_equipamento, 'canal_sem_geracao', CAST(min(data) AS TIMESTAMP), CAST(date_add(max(data), 1) AS TIMESTAMP), count(*),
              concat('canal com modulo gerando menos de 5% do outro canal do microinversor em ', count(*), ' dia(s) com dado') FROM s2 GROUP BY id_equipamento, ilha),
-- 6) estado do rastreador (apenas estados fora da operacao prevista)
t AS (SELECT data, estado FROM {GOLD}.estado_rastreador_dia WHERE estado IN ('parado_referencia', 'travado_oeste', 'travado_leste')),
t2 AS (SELECT *, datediff(data, DATE'2020-01-01') - row_number() OVER (PARTITION BY estado ORDER BY data) AS ilha FROM t),
e6 AS (SELECT '403000261021', concat('rastreador_', estado), CAST(min(data) AS TIMESTAMP), CAST(date_add(max(data), 1) AS TIMESTAMP), count(*),
              concat('rastreador ', replace(estado, '_', ' '), ' (estado inferido da razao rastreador/fixo de manha e a tarde)')
       FROM t2 GROUP BY estado, ilha HAVING count(*) >= 2),
-- 7) dia de baixo desempenho com ceu claro
dd AS (SELECT date(hora_local) AS data, sum(energia_medida_wh) AS m, sum(energia_esperada_wh) AS e, count(*) AS n,
              sum(ghi_wh_m2) / sum(ghi_ceu_claro_wh_m2) AS claridade
       FROM {GOLD}.gold_incidencia_usina_hora WHERE n_canais_incluidos > 0 GROUP BY 1),
e7 AS (SELECT NULL, 'dia_baixo_desempenho', CAST(data AS TIMESTAMP), CAST(date_add(data, 1) AS TIMESTAMP), n,
              concat('razao medida/esperada ', format_number(m / e, 2), ' em dia de indice de claridade ', format_number(claridade, 2))
       FROM dd WHERE n >= 6 AND claridade >= 0.6 AND m / e < 0.8)
SELECT * FROM e1 UNION ALL SELECT * FROM e2 UNION ALL SELECT * FROM e3 UNION ALL SELECT * FROM e4
UNION ALL SELECT * FROM e5 UNION ALL SELECT * FROM e6 UNION ALL SELECT * FROM e7
""")

from pyspark.sql.functions import expr, lit, current_timestamp, sha2, concat_ws, col
log = (eventos.toDF("id_equipamento", "tipo_evento", "inicio", "fim", "n_registros", "descricao")
       .withColumn("id_usina", lit("ifes_campus_vitoria"))
       .withColumn("duracao_min", expr("timestampdiff(MINUTE, inicio, fim)"))
       .withColumn("id_evento", sha2(concat_ws("|", col("tipo_evento"), col("id_equipamento"), col("inicio").cast("string")), 256).substr(1, 16))
       .withColumn("atualizado_em", current_timestamp())
       .select("id_evento", "id_usina", "id_equipamento", "tipo_evento", "inicio", "fim", "duracao_min", "n_registros", "descricao", "atualizado_em"))
log.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.log_eventos")
spark.sql(f"COMMENT ON TABLE {GOLD}.log_eventos IS 'Eventos operacionais derivados dos dados por regras explicitas (sem registro de manutencao disponivel): falhas de coleta e comunicacao, relogio do equipamento, canal sem geracao, estados do rastreador e dias de baixo desempenho.'")
spark.sql(f"ALTER TABLE {GOLD}.log_eventos ALTER COLUMN id_evento SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD}.log_eventos DROP CONSTRAINT IF EXISTS pk_log_eventos")
spark.sql(f"ALTER TABLE {GOLD}.log_eventos ADD CONSTRAINT pk_log_eventos PRIMARY KEY (id_evento)")

# COMMAND ----------

import json
resumo = spark.sql(f"""SELECT tipo_evento, count(*) eventos, sum(n_registros) registros, round(sum(duracao_min) / 60, 1) horas,
                       min(inicio) primeiro, max(fim) ultimo FROM {GOLD}.log_eventos GROUP BY 1 ORDER BY 1""")
display(resumo)
dbutils.notebook.exit(json.dumps([r.asDict() for r in resumo.collect()], default=str))
