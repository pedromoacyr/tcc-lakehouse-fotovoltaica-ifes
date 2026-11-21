# Databricks notebook source
# MAGIC %md
# MAGIC # Figuras do capitulo de Resultados (geradas da gold, com o mesmo codigo dos numeros)
# MAGIC
# MAGIC Cada figura le exatamente as tabelas
# MAGIC `avaliacao_*` e `gold_*` que fornecem os numeros citados no texto, e a
# MAGIC propria execucao imprime os valores plotados, para conferencia.
# MAGIC
# MAGIC Convencoes: sem titulo interno (o titulo fica na legenda do LaTeX), eixos
# MAGIC acentuados, irradiancia em W/m² e irradiacao em kWh·m⁻²·dia⁻¹, barras de
# MAGIC erro = intervalo de confianca de 95% (bootstrap por dia) e n de cada
# MAGIC barra anotado. Arquivos gravados em `/Volumes/fotovoltaica_ifes/gold/graficos_tcc/`.

# COMMAND ----------

import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD = "fotovoltaica_ifes.gold"
VOL = "/Volumes/fotovoltaica_ifes/gold/graficos_tcc"
os.makedirs(VOL, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
AZUL, LARANJA, CINZA, VERDE = "#1f5aa6", "#e08a1e", "#6b6b6b", "#2e8b57"
conferencia = {}


def salvar(fig, nome):
    # separador decimal em virgula (norma brasileira) nos eixos numericos
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    for a in fig.axes:
        for eixo in (a.xaxis, a.yaxis):
            if isinstance(eixo.get_major_formatter(), ScalarFormatter):
                eixo.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
    fig.savefig(f"{VOL}/{nome}", bbox_inches="tight")
    plt.close(fig)


def barras_ic(ax, df, x, cor, rotulo=None, deslocar=0.0, largura=0.38):
    pos = np.arange(len(df)) + deslocar
    err = np.vstack([df.razao_somas - df.ic95_inf, df.ic95_sup - df.razao_somas])
    ax.bar(pos, df.razao_somas, largura, color=cor, label=rotulo, yerr=err, capsize=2.5, error_kw={"lw": 0.8})
    for p, r, n in zip(pos, df.ic95_sup, df.n_canal_horas):
        ax.text(p, r + 0.015, f"{int(n)}", ha="center", va="bottom", fontsize=6.5, color=CINZA, rotation=90)
    return pos

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figura: dispersao energia medida x esperada (horas pareadas)

# COMMAND ----------

uh = spark.table(f"{GOLD}.gold_incidencia_usina_hora").where("flag_hora_pareada_usina").toPandas()
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.scatter(uh.energia_esperada_wh, uh.energia_medida_wh, s=4, alpha=0.25, color=AZUL, edgecolors="none")
lim = max(uh.energia_esperada_wh.max(), uh.energia_medida_wh.max()) * 1.03
ax.plot([0, lim], [0, lim], color=CINZA, lw=1, ls="--", label="medida = esperada")
r = np.corrcoef(uh.energia_esperada_wh, uh.energia_medida_wh)[0, 1]
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Energia esperada na hora (Wh)"); ax.set_ylabel("Energia medida na hora (Wh)")
ax.text(0.03, 0.95, f"r = {r:.3f}".replace(".", ",") + f"\nn = {len(uh):,} horas".replace(",", "."), transform=ax.transAxes, va="top", fontsize=9)
ax.legend(loc="lower right", frameon=False)
salvar(fig, "01_dispersao_medida_esperada.png")
conferencia["fig1"] = {"r": round(r, 4), "n": len(uh)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figuras: razao medida/esperada por hora, estacao e faixa de irradiancia

# COMMAND ----------

rec = spark.table(f"{GOLD}.avaliacao_recortes").toPandas()
h = rec[(rec.recorte == "hora_do_dia")].copy(); h["ordem"] = h.categoria.astype(int)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ht = h[h.estrutura == "todas"].sort_values("ordem"); hf = h[h.estrutura == "fixa"].sort_values("ordem")
pos = barras_ic(ax, ht, "categoria", AZUL, "todos os canais", -0.2)
barras_ic(ax, hf, "categoria", LARANJA, "só módulos fixos", 0.2)
ax.axhline(1, color=CINZA, lw=0.8, ls="--")
ax.set_xticks(np.arange(len(ht))); ax.set_xticklabels([f"{int(c)}h" for c in ht.categoria])
ax.set_xlabel("Hora local (início do intervalo de 1 h)"); ax.set_ylabel("Razão medida/esperada")
ax.set_ylim(0, 1.55); ax.legend(frameon=False, ncol=2, loc="upper left")
salvar(fig, "02_razao_por_hora.png")
conferencia["fig2"] = ht[["categoria", "razao_somas", "n_canal_horas"]].round(3).to_dict("records")

ordem = ["verao", "outono", "inverno", "primavera"]; nomes = ["Verão\n(dez–fev)", "Outono\n(mar–mai)", "Inverno\n(jun–ago)", "Primavera\n(set–nov)"]
e = rec[rec.recorte == "estacao"].copy(); e["o"] = e.categoria.map({k: i for i, k in enumerate(ordem)})
fig, ax = plt.subplots(figsize=(5.6, 3.6))
et = e[e.estrutura == "todas"].sort_values("o"); ef = e[e.estrutura == "fixa"].sort_values("o")
barras_ic(ax, et, "categoria", AZUL, "todos os canais", -0.2); barras_ic(ax, ef, "categoria", LARANJA, "só módulos fixos", 0.2)
ax.axhline(1, color=CINZA, lw=0.8, ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels(nomes)
ax.set_ylabel("Razão medida/esperada"); ax.set_ylim(0, 1.45); ax.legend(frameon=False, ncol=2, loc="upper left")
salvar(fig, "03_razao_por_estacao.png")
conferencia["fig3"] = et[["categoria", "razao_somas", "ic95_inf", "ic95_sup", "n_canal_horas"]].round(3).to_dict("records")

fx = rec[rec.recorte == "faixa_irradiancia_plano"].copy().sort_values("categoria")
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ft = fx[fx.estrutura == "todas"]; ff = fx[fx.estrutura == "fixa"]
barras_ic(ax, ft, "categoria", AZUL, "todos os canais", -0.2); barras_ic(ax, ff, "categoria", LARANJA, "só módulos fixos", 0.2)
ax.axhline(1, color=CINZA, lw=0.8, ls="--"); ax.set_xticks(range(len(ft)))
ax.set_xticklabels([c.replace("050", "50").replace("-", "–") for c in ft.categoria])
ax.set_xlabel("Irradiância no plano dos módulos na hora (W/m²)"); ax.set_ylabel("Razão medida/esperada")
ax.set_ylim(0, 1.45); ax.legend(frameon=False, ncol=2, loc="upper left")
salvar(fig, "04_razao_por_faixa_irradiancia.png")
conferencia["fig4"] = ft[["categoria", "razao_somas", "n_canal_horas"]].round(3).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figura: serie mensal de energia medida e esperada

# COMMAND ----------

m = spark.table(f"{GOLD}.gold_incidencia_usina_mes").toPandas().sort_values(["ano", "mes"])
m["rot"] = [f"{['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'][a-1]}/{str(b)[2:]}" for a, b in zip(m.mes, m.ano)]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
x = np.arange(len(m))
ax.bar(x - 0.2, m.energia_medida_kwh.fillna(0), 0.4, color=AZUL, label="medida")
ax.bar(x + 0.2, m.energia_esperada_kwh.fillna(0), 0.4, color=LARANJA, label="esperada")
ax.set_xticks(x); ax.set_xticklabels(m.rot, rotation=45, ha="right")
ax.set_ylabel("Energia nas horas incluídas (kWh)")
ax2 = ax.twinx(); ax2.plot(x, m.razao_real_esperado_mes, "o-", color=CINZA, ms=3.5, lw=1, label="razão (eixo direito)")
ax2.set_ylim(0, 1.4); ax2.set_ylabel("Razão medida/esperada"); ax2.spines["right"].set_visible(True)
for xi, n in zip(x, m.canal_horas_incluidas.fillna(0)):
    ax.text(xi, 2, f"{int(n)}", ha="center", va="bottom", fontsize=6, color="white", rotation=90)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
salvar(fig, "05_serie_mensal_medida_esperada.png")
conferencia["fig5"] = m[["ano", "mes", "canal_horas_incluidas", "energia_medida_kwh", "energia_esperada_kwh", "razao_real_esperado_mes"]].round(3).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figura: irradiacao global horizontal mensal - NASA POWER, INMET e Atlas

# COMMAND ----------

v = spark.table(f"{GOLD}.avaliacao_nasa_inmet_mes").toPandas().sort_values("mes_referencia")
mes_nome = ['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez']
v["rot"] = [f"{mes_nome[int(s[5:7]) - 1]}/{s[2:4]}" for s in v.mes_referencia]
fig, ax = plt.subplots(figsize=(7.4, 3.6))
x = np.arange(len(v))
ax.bar(x - 0.27, v.nasa, 0.27, color=AZUL, label="NASA POWER (estimada por satélite)")
ax.bar(x, v.inmet, 0.27, color=LARANJA, label="INMET A612 (medida em solo)")
ax.bar(x + 0.27, v.atlas_kwh_m2_dia, 0.27, color=CINZA, label="Atlas 2017 (média de longo prazo)")
for xi, d in zip(x, v.dias):
    ax.text(xi, 0.1, f"{int(d)}", ha="center", va="bottom", fontsize=6, color="white")
ax.set_xticks(x); ax.set_xticklabels(v.rot, rotation=45, ha="right")
ax.set_ylabel("Irradiação global horizontal\n(kWh·m⁻²·dia⁻¹)"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
salvar(fig, "06_irradiacao_nasa_inmet_atlas.png")
conferencia["fig6"] = v[["mes_referencia", "nasa", "inmet", "atlas_kwh_m2_dia", "dias"]].round(3).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figura: comportamento do rastreador por periodo (razao rastreador/fixo por hora)

# COMMAND ----------

rt = spark.sql(f"""
WITH p AS (SELECT date(instante_coleta_local) data, hour(instante_coleta_local) h, id_equipamento, potencia_w
           FROM {GOLD}.fato_potencia_5min WHERE id_equipamento IN ('403000261021-A', '403000261629-A'))
SELECT e.estado, p.h, sum(CASE WHEN id_equipamento = '403000261021-A' THEN potencia_w END) / 245.0
                      / sum(CASE WHEN id_equipamento = '403000261629-A' THEN potencia_w END) * 275.0 AS razao,
       count(DISTINCT p.data) dias
FROM p JOIN {GOLD}.estado_rastreador_dia e USING (data)
WHERE e.estado IN ('parado_referencia', 'rastreando', 'travado_oeste') AND NOT e.estado_inferido AND p.h BETWEEN 7 AND 16
GROUP BY 1, 2 ORDER BY 1, 2""").toPandas()
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for est, cor, nome in [("parado_referencia", CINZA, "parado na posição de referência"), ("rastreando", VERDE, "rastreando"),
                       ("travado_oeste", LARANJA, "travado voltado para oeste")]:
    s = rt[rt.estado == est]
    ax.plot(s.h, s.razao, "o-", color=cor, ms=3.5, label=f"{nome} ({int(s.dias.max())} dias)")
ax.axhline(1, color=CINZA, lw=0.8, ls="--")
ax.set_xticks(range(7, 17)); ax.set_xticklabels([f"{h}h" for h in range(7, 17)])
ax.set_xlabel("Hora local"); ax.set_ylabel("Energia por Wp: rastreador / fixo")
ax.legend(frameon=False, fontsize=8)
salvar(fig, "07_rastreador_por_estado.png")
conferencia["fig7"] = rt.round(3).to_dict("records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figura: teste de alinhamento sub-horario

# COMMAND ----------

al = spark.table(f"{GOLD}.avaliacao_alinhamento").toPandas().sort_values("deslocamento_min")
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(al.deslocamento_min, al.razao_manha_ate_10h, "o-", color=AZUL, ms=3.5, label="razão nas horas até 10h")
ax.plot(al.deslocamento_min, al.razao_tarde_desde_14h, "s-", color=LARANJA, ms=3.5, label="razão nas horas a partir de 14h")
ax.axhline(1, color=CINZA, lw=0.8, ls="--"); ax.axvline(0, color=CINZA, lw=0.6)
ax.set_xlabel("Deslocamento aplicado ao carimbo da usina (min)"); ax.set_ylabel("Razão medida/esperada (fixos)")
ax2 = ax.twinx(); ax2.plot(al.deslocamento_min, al.rmse_wh, ":", color=CINZA, lw=1.2, label="RMSE horário (eixo direito)")
ax2.set_ylabel("RMSE horário (Wh)"); ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center")
salvar(fig, "08_alinhamento_sub_horario.png")
conferencia["fig8"] = al.round(4).to_dict("records")

# COMMAND ----------

print(json.dumps(conferencia, indent=1, default=str)[:20000])
print(sorted(os.listdir(VOL)))
dbutils.notebook.exit(json.dumps({"arquivos": sorted(os.listdir(VOL)), "conferencia": conferencia}, default=str))
