#!/usr/bin/env python3
"""
Baixa dados de irradiância solar e temperatura do NASA POWER para o Campus Vitória (Ifes),
cobrindo o período das medições da usina (27/11/2020 a 15/02/2022).

Uso (no Windows, em um terminal, dentro da pasta onde este arquivo está):
    python baixar_nasa_power.py

Gera, na mesma pasta:
    nasa_power_horario_vitoria.csv        série horária (UTC), uma linha por hora
    nasa_power_horario_vitoria_meta.txt   cabeçalho oficial do NASA POWER (unidades, fontes, coordenadas)
    nasa_power_climatologia_vitoria.csv   médias mensais climatológicas
    resumo_irradiacao_mensal.csv          média do total diário por mês (kWh/m2.dia), calculada da série horária

Não precisa instalar nada: usa apenas a biblioteca padrão do Python 3.
"""
import csv
import datetime as dt
import io
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

# --- Parâmetros (ajuste aqui) -------------------------------------------------
LAT, LON = -20.31, -40.32            # aproximado para o Campus Vitória; trocar pelas coordenadas da usina
INICIO, FIM = dt.date(2020, 11, 27), dt.date(2022, 2, 15)
PARAMETROS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M"
# ALLSKY_SFC_SW_DWN = irradiância global horizontal com nuvens (Wh/m2 por hora)
# CLRSKY_SFC_SW_DWN = irradiância global horizontal em céu claro (Wh/m2 por hora)
# T2M               = temperatura do ar a 2 m (graus C)
PADRAO_TEMPO = "UTC"                 # "UTC" ou "LST" (hora solar local)
DIAS_POR_BLOCO = 90
BASE = "https://power.larc.nasa.gov/api/temporal"
# ------------------------------------------------------------------------------

PASTA = pathlib.Path(__file__).resolve().parent


def baixar(url, tentativas=4):
    req = urllib.request.Request(url, headers={"User-Agent": "tcc-lakehouse-solar/1.0"})
    for n in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  tentativa {n}/{tentativas} falhou: {e}")
            if n == tentativas:
                raise
            time.sleep(5 * n)


def separar_cabecalho(texto):
    """Devolve (linhas_do_cabecalho, linhas_de_dados) de uma resposta CSV do POWER."""
    linhas = texto.splitlines()
    for i, ln in enumerate(linhas):
        if ln.strip() == "-END HEADER-":
            return linhas[: i + 1], linhas[i + 1 :]
    return [], linhas  # resposta sem cabeçalho


def blocos(inicio, fim, dias):
    atual = inicio
    while atual <= fim:
        final = min(atual + dt.timedelta(days=dias - 1), fim)
        yield atual, final
        atual = final + dt.timedelta(days=1)


def serie_horaria():
    meta, cabecalho_csv, linhas = None, None, []
    for ini, fim in blocos(INICIO, FIM, DIAS_POR_BLOCO):
        qs = urllib.parse.urlencode({
            "parameters": PARAMETROS, "community": "RE",
            "longitude": LON, "latitude": LAT,
            "start": ini.strftime("%Y%m%d"), "end": fim.strftime("%Y%m%d"),
            "format": "CSV", "time-standard": PADRAO_TEMPO,
        })
        print(f"Horário {ini} a {fim} ...")
        texto = baixar(f"{BASE}/hourly/point?{qs}")
        cab, dados = separar_cabecalho(texto)
        if meta is None:
            meta = cab
        dados = [d for d in dados if d.strip()]
        if not dados:
            raise RuntimeError("Resposta vazia; confira os parâmetros.")
        if cabecalho_csv is None:
            cabecalho_csv = dados[0]
        linhas.extend(dados[1:])
        time.sleep(1)
    (PASTA / "nasa_power_horario_vitoria_meta.txt").write_text("\n".join(meta or []), encoding="utf-8")
    (PASTA / "nasa_power_horario_vitoria.csv").write_text(
        cabecalho_csv + "\n" + "\n".join(linhas) + "\n", encoding="utf-8")
    print(f"  salvo: nasa_power_horario_vitoria.csv ({len(linhas)} linhas)")
    return cabecalho_csv, linhas


def climatologia():
    qs = urllib.parse.urlencode({
        "parameters": PARAMETROS, "community": "RE",
        "longitude": LON, "latitude": LAT, "format": "CSV",
    })
    print("Climatologia mensal ...")
    texto = baixar(f"{BASE}/climatology/point?{qs}")
    (PASTA / "nasa_power_climatologia_vitoria.csv").write_text(texto, encoding="utf-8")
    print("  salvo: nasa_power_climatologia_vitoria.csv")


def resumo_mensal(cabecalho_csv, linhas):
    """Média do total diário de ALLSKY_SFC_SW_DWN por mês (kWh/m2.dia), ignorando -999."""
    leitor = csv.DictReader(io.StringIO(cabecalho_csv + "\n" + "\n".join(linhas)))
    por_dia = defaultdict(float)
    faltas_dia = defaultdict(int)
    faltas = 0
    total = 0
    for r in leitor:
        total += 1
        valor = float(r["ALLSKY_SFC_SW_DWN"])
        chave = (int(r["YEAR"]), int(r["MO"]), int(r["DY"]))
        if valor <= -900:
            faltas += 1
            faltas_dia[chave] += 1
        else:
            por_dia[chave] += valor
    por_mes = defaultdict(list)
    for (a, m, d), wh in por_dia.items():
        if faltas_dia[(a, m, d)] == 0:
            por_mes[(a, m)].append(wh / 1000.0)
    with open(PASTA / "resumo_irradiacao_mensal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ano", "mes", "dias_completos", "media_total_diario_kWh_m2_dia"])
        for (a, m) in sorted(por_mes):
            v = por_mes[(a, m)]
            w.writerow([a, m, len(v), round(sum(v) / len(v), 3)])
    print(f"  salvo: resumo_irradiacao_mensal.csv (valores ausentes: {faltas} de {total})")


def main():
    try:
        cab, linhas = serie_horaria()
        climatologia()
        resumo_mensal(cab, linhas)
    except Exception as e:  # noqa: BLE001
        print("\nErro:", e)
        print("Se for bloqueio de rede, tente em outra rede ou abra o link no navegador:")
        print("https://power.larc.nasa.gov/data-access-viewer/")
        sys.exit(1)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
