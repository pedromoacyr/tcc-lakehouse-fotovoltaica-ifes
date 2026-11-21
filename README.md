# Arquitetura lakehouse para dados de geração fotovoltaica e irradiação solar

Código do Trabalho de Conclusão de Curso de Engenharia Elétrica de Pedro Moacyr Martins, no Instituto Federal do Espírito Santo (Ifes), Campus Vitória, 2026:

> *Arquitetura lakehouse para integração de medições de geração fotovoltaica e dados públicos de irradiação solar: estudo de caso da usina do Campus Vitória do Ifes.*

Orientador: Prof. Renato Benezath Cabelino Ribeiro.

O repositório reúne os notebooks do Databricks que consolidam as medições dos microinversores da usina fotovoltaica do Campus Vitória numa arquitetura em camadas (landing, bronze, silver e gold), integram essas medições a fontes públicas de irradiação (NASA POWER, Atlas Brasileiro de Energia Solar e estação automática A612 do INMET) e calculam a energia esperada por canal de microinversor e por hora.

## Conteúdo

| Pasta | Conteúdo |
| --- | --- |
| `notebooks/` | Os 15 notebooks do pipeline e da avaliação, no formato de código-fonte do Databricks (`.py`) |
| `scripts/` | Script em Python, só com a biblioteca padrão, que baixa a série horária e a climatologia do NASA POWER |

Os dados de medição da usina não estão incluídos; o acesso depende da coordenação da usina no Ifes.

## Ordem de execução

A numeração é a mesma do Apêndice B do TCC. Os sufixos (04b, 05b e 08b) identificam etapas inseridas na sequência sem renumerar as demais.

| Notebook | Camada | Função |
| --- | --- | --- |
| `01_bronze_extracao_usina` | bronze | Carga histórica das medições, por mesclagem (`MERGE`) |
| `02_bronze_autoloader_usina` | bronze | Carga incremental com Auto Loader, por mesclagem |
| `03_silver_usina_geracao` | silver | Tratamento das medições, canais A e B, e tabela de quarentena |
| `04b_bronze_fontes_complementares` | bronze | Coleta do NASA POWER, do Atlas (LABREN/INPE) e da estação A612 do INMET |
| `05b_silver_fontes_complementares` | silver | Padronização das fontes de irradiação |
| `06_gold_dimensoes` | gold | Dimensões e parâmetros da usina e dos equipamentos |
| `07_gold_fatos` | gold | Fatos de potência e de irradiação |
| `08_gold_tabela_final` | gold | Irradiação no plano dos módulos, estado do rastreador e energia esperada |
| `08b_gold_log_eventos` | gold | Log de eventos operacionais |
| `09_governanca_dicionario_dados` | governança | Comentários de tabelas e colunas no Unity Catalog |
| `10_gold_visualizacoes_tcc` | gold | Figuras do Capítulo 4 |
| `11_avaliacao_funcional_qualidade` | avaliação | Regras de qualidade por camada |
| `12_avaliacao_desempenho` | avaliação | Armazenamento, leitura e escalabilidade |
| `13_teste_idempotencia_incremental` | avaliação | Teste de idempotência e de carga incremental num esquema separado |
| `14_avaliacao_aderencia` | avaliação | Métricas de erro, recortes, sensibilidade e alinhamento temporal |

Os notebooks 01 a 10 formam o pipeline. Os notebooks 11 a 14 executam a avaliação e gravam os resultados em tabelas `gold.avaliacao_*`.

## Como executar

1. Crie no Unity Catalog o catálogo `fotovoltaica_ifes`, com os esquemas `bronze`, `silver` e `gold`, e o volume `bronze.landing_usina`.
2. Deposite o arquivo de medições `medicoes_usina.zip` em `/Volumes/fotovoltaica_ifes/bronze/landing_usina/`.
3. Importe os notebooks no workspace (*Import* > *File*) e execute-os na ordem acima. O trabalho usou o Databricks Free Edition, com compute serverless.

Os parâmetros da usina (coordenadas, inclinação, azimute e geometria do rastreador) ficam na tabela `silver.config_parametros`. Para usar outra usina ou corrigir uma premissa, basta alterar essa tabela e reprocessar a gold, sem mudar o código.

## Dependências

Databricks com Unity Catalog, Delta Lake e Auto Loader, e as bibliotecas Python pvlib, pandas, NumPy e Matplotlib. A avaliação de desempenho também usa DuckDB.

## Licença

Código distribuído sob a licença MIT (arquivo `LICENSE`).
