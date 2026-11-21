# Databricks notebook source
# MAGIC %md
# MAGIC # Governanca - dicionario de dados no Unity Catalog
# MAGIC
# MAGIC Registra a descricao de cada tabela e coluna das camadas silver e gold
# MAGIC (`COMMENT ON TABLE` / `ALTER TABLE ... ALTER COLUMN ... COMMENT`).
# MAGIC
# MAGIC **Aplicacao incremental.** O notebook le os comentarios ja registrados no `information_schema` e so
# MAGIC aplica os que mudaram; numa reexecucao sem mudancas, nenhuma instrucao de
# MAGIC escrita e enviada ao metastore. O tempo de governanca fica separado do
# MAGIC tempo de processamento de dados na avaliacao de desempenho.

# COMMAND ----------

import time, json
t_inicio = time.time()
D = {
 "silver.usina_geracao": ("Leituras de 5 minutos por canal de microinversor (A e B), tipadas, com o instante da coleta extraido do nome do arquivo.", {
   "id_equipamento": "Canal do microinversor (numero de serie + canal, ex.: 403000261021-A).",
   "id_microinversor": "Numero de serie do microinversor APsystems YC500A.",
   "canal": "Canal do microinversor (A ou B); cada canal atende um modulo.",
   "instante_coleta_local": "Data e hora da coleta (UTC-3), extraidas do nome do arquivo de origem.",
   "instante_coleta_utc": "Instante da coleta em UTC.",
   "potencia_w": "Potencia instantanea do canal (W).",
   "tensao_v": "Tensao da rede (V); na linha B vem na posicao da coluna de frequencia do CSV.",
   "frequencia_hz": "Frequencia da rede (Hz), leitura do microinversor (linha A, herdada pela linha B).",
   "temperatura_c": "Temperatura reportada pelo microinversor (graus C), herdada pela linha B.",
   "reporting_time_local": "Horario de reporte do equipamento (nao usado como instante: congela e diverge a partir de out/2021).",
   "origem_leituras_inversor": "linha_propria (canal A) ou herdado_da_linha_A (canal B).",
   "flag_reporting_time_divergente": "Verdadeiro quando o horario de reporte difere mais de 60 min do instante de coleta.",
   "arquivo_origem": "Caminho do CSV de origem dentro do arquivo diario.",
   "arquivo_dia": "Arquivo diario (.tar.gz) de origem."}),
 "silver.usina_geracao_quarentena": ("Linhas da bronze que nao entram na silver, com o motivo.", {
   "motivo_quarentena": "equipamento_sem_leitura (valor '-') ou id_sem_canal_e_sem_leitura (identificador sem sufixo A/B).",
   "registrado_em": "Momento do processamento."}),
 "silver.config_equipamentos": ("Parametros de ficha tecnica por canal (modulo conectado, estrutura e microinversor).", {
   "estrutura": "fixa, rastreador ou sem_modulo.", "modulo_modelo": "Modelo do modulo conectado ao canal.",
   "potencia_nominal_modulo_wp": "Potencia nominal do modulo (Wp, STC).", "area_modulo_m2": "Area do modulo (m2).",
   "eficiencia_modulo": "Eficiencia do modulo em STC (fracao).", "coef_temp_potencia_pct_c": "Coeficiente de temperatura da potencia (%/C).",
   "noct_c": "Temperatura nominal de operacao da celula (C).", "eficiencia_pcu": "Eficiencia ponderada CEC do microinversor (fracao).",
   "potencia_saida_canal_w": "Potencia de saida continua por canal (W).", "fonte": "Documentos de origem dos parametros."}),
 "gold.dim_equipamento": ("Dimensao de canais de microinversor com o modulo e os parametros usados no modelo.", {
   "inicio_geracao": "Primeiro dia com potencia positiva no canal.", "potencia_maxima_observada_w": "Maior potencia registrada no canal (W)."}),
 "gold.fato_potencia_5min": ("Fato de potencia por canal a cada 5 minutos, com indicadores de qualidade.", {
   "id_tempo": "Chave da hora em dim_tempo (yyyyMMddHH).", "temperatura_microinversor_c": "Temperatura reportada pelo microinversor (C).",
   "flag_horario_reporte_congelado": "Horario de reporte igual ao da leitura anterior do mesmo microinversor.",
   "flag_reporting_time_divergente": "Horario de reporte a mais de 60 min do instante de coleta.",
   "flag_potencia_no_teto": "Potencia >= 265 W (limite de saida do canal); mede clipping.",
   "flag_potencia_acima_nominal": "Potencia acima da nominal do modulo.", "flag_canal_sem_modulo": "Canal sem modulo conectado.",
   "flag_temperatura_ausente": "Temperatura nao informada."}),
 "gold.fato_irradiancia_hora": ("Irradiancia horaria do NASA POWER (com componentes) e medicao de solo do INMET A612, rotuladas pelo inicio da hora.", {
   "hora_local": "Inicio do intervalo de 1 h (UTC-3).", "hora_utc": "Inicio do intervalo de 1 h (UTC).",
   "ghi_wh_m2": "Irradiacao global horizontal na hora (Wh/m2), NASA POWER.", "dni_wh_m2": "Irradiacao direta normal (Wh/m2), NASA POWER.",
   "dhi_wh_m2": "Irradiacao difusa horizontal (Wh/m2), NASA POWER.", "ghi_ceu_claro_wh_m2": "Global horizontal em ceu claro (Wh/m2).",
   "temperatura_ar_c": "Temperatura do ar a 2 m (C), NASA POWER.", "vento_2m_m_s": "Velocidade do vento a 2 m (m/s), NASA POWER.",
   "ghi_inmet_wh_m2": "Radiacao global medida na estacao INMET A612 (Wh/m2).", "precipitacao_inmet_mm": "Precipitacao na hora (mm), INMET A612."}),
 "gold.fato_irradiancia_plano_hora": ("Irradiancia no plano dos modulos (fixa e rastreador) por tres modelos de transposicao e pela GHI do INMET.", {
   "zenite_solar_graus": "Angulo zenital aparente no meio da hora.", "azimute_solar_graus": "Azimute solar no meio da hora (0 = norte).",
   "angulo_rastreador_graus": "Rotacao do rastreador de 1 eixo no modelo (graus)."}),
 "gold.fato_energia_canal_hora": ("Energia medida e esperada (Equacao 3.1 no plano dos modulos) por canal e hora, com as variantes de sensibilidade.", {
   "geometria": "fixa, rastreamento, fixa_referencia (rastreador parado) ou travado_ou_indeterminado.",
   "estado_rastreador": "Estado diario do rastreador (estado_rastreador_dia).", "n_leituras_5min": "Leituras de 5 min registradas na hora (0 a 12).",
   "hora_completa": "Hora com as 12 leituras.", "canal_sem_geracao_no_dia": "Canal gerou menos de 5% do outro canal do microinversor no dia.",
   "rastreador_fora_de_operacao_normal": "Dia em que o rastreador estava travado ou em estado indeterminado.",
   "incluir_aderencia": "Hora completa, canal gerando, rastreador normal e irradiancia no plano >= 50 Wh/m2.",
   "energia_medida_wh": "Energia medida na hora (Wh), media das 12 leituras de potencia.",
   "energia_esperada_wh": "Energia esperada (Wh): Perez + temperatura de celula NOCT (modelo principal).",
   "poa_wh_m2": "Irradiacao no plano do modulo (Wh/m2), Perez.", "temperatura_celula_noct_c": "Temperatura de celula pelo modelo NOCT (C).",
   "esperada_perez_noct_wh": "Energia esperada: Perez + NOCT.", "esperada_haydavies_noct_wh": "Energia esperada: Hay-Davies + NOCT.",
   "esperada_isotropico_noct_wh": "Energia esperada: isotropico + NOCT.", "esperada_perez_proxy_wh": "Energia esperada: Perez + temperatura do microinversor.",
   "esperada_inmet_noct_wh": "Energia esperada com a GHI medida no INMET (Erbs + Perez + NOCT)."}),
 "gold.gold_incidencia_usina_hora": ("Tabela final usina x hora: irradiancia, energia medida e esperada dos canais incluidos e razao.", {
   "n_canais_em_operacao": "Canais com modulo em operacao normal na hora.", "n_canais_incluidos": "Canais incluidos nas metricas na hora.",
   "flag_hora_pareada_usina": "Todos os canais em operacao incluidos (comparacao da usina inteira).",
   "razao_real_esperado": "Energia medida / energia esperada na hora."}),
 "gold.gold_incidencia_usina_mes": ("Tabela final usina x mes: razao de somas, cobertura e disponibilidade diurnas e referencias de irradiacao.", {
   "disponibilidade_diurna": "Leituras registradas / possiveis nas horas com sol, canais com modulo em operacao.",
   "razao_real_esperado_mes": "Soma medida / soma esperada no mes (canal-horas incluidos)."}),
 "gold.energia_esperada_atlas_mes": ("Energia diaria esperada de um mes tipico pela climatologia do Atlas 2017 e energia medida nos dias completos por canal.", {}),
 "gold.estado_rastreador_dia": ("Estado diario do rastreador inferido pela razao rastreador/fixo de manha e a tarde.", {
   "ganho_manha": "Razao rastreador/fixo 8h-10h normalizada pelo meio-dia.", "ganho_tarde": "Razao rastreador/fixo 14h-16h normalizada pelo meio-dia.",
   "estado": "rastreando, parado_referencia, travado_oeste, travado_leste ou indeterminado.", "estado_inferido": "Estado herdado de dia vizinho (ate 7 dias)."}),
 "gold.log_eventos": ("Eventos operacionais derivados dos dados por regras explicitas.", {
   "tipo_evento": "Regra que gerou o evento.", "inicio": "Inicio do evento.", "fim": "Fim do evento.",
   "duracao_min": "Duracao (min).", "n_registros": "Registros (leituras ou dias) agrupados no evento."}),
 "gold.ref_climatologia_mes": ("Referencias climatologicas mensais (kWh/m2.dia): Atlas 2017 (quatro camadas) e NASA POWER 2001-2020.", {
   "camada": "global_horizontal, difusa, direta_normal, inclinada_latitude ou global_horizontal_ceu_claro."}),
}
existentes = {}
for r in spark.sql("""SELECT table_schema, table_name, column_name, comment FROM fotovoltaica_ifes.information_schema.columns
                      WHERE table_schema IN ('silver', 'gold')""").collect():
    existentes[(f"{r['table_schema']}.{r['table_name']}", r["column_name"])] = r["comment"]
tab_exist = {f"{r['table_schema']}.{r['table_name']}": r["comment"] for r in spark.sql(
    "SELECT table_schema, table_name, comment FROM fotovoltaica_ifes.information_schema.tables WHERE table_schema IN ('silver', 'gold')").collect()}
aplicados, ignorados = 0, 0
esc = lambda t: t.replace("\\", "\\\\").replace("'", "\\'")   # escapa aspas simples no literal SQL
for tab, (desc, cols) in D.items():
    if tab not in tab_exist:
        continue
    if tab_exist[tab] != desc:
        spark.sql(f"COMMENT ON TABLE fotovoltaica_ifes.{tab} IS '{esc(desc)}'"); aplicados += 1
    else:
        ignorados += 1
    for c, d in cols.items():
        if (tab, c) not in existentes:
            continue
        if existentes[(tab, c)] != d:
            spark.sql(f"ALTER TABLE fotovoltaica_ifes.{tab} ALTER COLUMN {c} COMMENT '{esc(d)}'"); aplicados += 1
        else:
            ignorados += 1
dur = time.time() - t_inicio
r = {"comentarios_aplicados": aplicados, "comentarios_ja_atualizados": ignorados, "duracao_s": round(dur, 1)}
print(r)
dbutils.notebook.exit(json.dumps(r))
