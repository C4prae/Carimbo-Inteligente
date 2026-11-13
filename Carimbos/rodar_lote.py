#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rodar_lote.py
=============
Roda o fluxo de carimbos (fluxo_carimbos.py) em vários PDFs de uma vez:
"teste (1).pdf", "teste (2).pdf", ... "teste (20).pdf".

Tudo vai para a mesma pasta de saída:
    saida_fluxo/docs/    -> "teste (n)_carimbado.png", "teste (n)_metadados.json"
    saida_fluxo/Stamps/  -> "teste (n)_carimbo_redondo.png", "teste (n)_carimbo_retangular.png"

Exemplos
--------
    python rodar_lote.py                                  # PDFs na pasta atual, n de 1 a 20
    python rodar_lote.py --pasta C:\\meus_pdfs --saida resultado
    python rodar_lote.py --semente 100                    # reprodutível: o doc n usa a semente 100 + n
    python rodar_lote.py --inicio 5 --fim 8               # só "teste (5).pdf" a "teste (8).pdf"
    python rodar_lote.py --modelo qwen3-vl:8b
"""
import argparse
import sys
import time
from pathlib import Path

import fluxo_carimbos as fc


def main():
    ap = argparse.ArgumentParser(description='Gera carimbos em "teste (n).pdf" para n de 1 a 20.')
    ap.add_argument("--pasta", default=".", help="pasta onde estão os PDFs (padrão: pasta atual)")
    ap.add_argument("--prefixo", default="teste", help='parte fixa do nome: "<prefixo> (n).pdf"')
    ap.add_argument("--inicio", type=int, default=1)
    ap.add_argument("--fim", type=int, default=20)
    ap.add_argument("--saida", default="saida_fluxo", help="pasta de saída (cria docs/ e Stamps/ dentro dela)")
    ap.add_argument("--modelo", default=fc.MODELO)
    ap.add_argument("--host", default=fc.OLLAMA_HOST)
    ap.add_argument("--semente", type=int, default=None,
                    help="semente base: o documento n usa semente+n (padrão: aleatória; fica gravada no JSON)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Confere o Ollama e o modelo uma única vez (evita repetir o mesmo erro 20 vezes)
    try:
        fc.verificar_ollama(a.host, a.modelo)
    except (ConnectionError, RuntimeError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        raise SystemExit(1)

    resultados, t0 = [], time.time()
    for n in range(a.inicio, a.fim + 1):
        pdf = Path(a.pasta) / f"{a.prefixo} ({n}).pdf"
        print(f"\n===== [{n - a.inicio + 1}/{a.fim - a.inicio + 1}] {pdf.name} =====", flush=True)
        if not pdf.exists():
            print("  arquivo não encontrado, pulando")
            resultados.append((n, "NÃO ENCONTRADO", ""))
            continue
        try:
            meta = fc.executar_fluxo(pdf, a.saida, modelo=a.modelo, host=a.host,
                                     semente=None if a.semente is None else a.semente + n)
            nao = [c["tipo"] for c in meta["carimbos"] if not c.get("aplicado")]
            resultados.append((n, "ok" if not nao else "ok (sem espaço p/ " + ", ".join(nao) + ")", ""))
        except ConnectionError as e:                       # Ollama caiu no meio: não adianta continuar
            print(f"ERRO: {e}", file=sys.stderr)
            resultados.append((n, "ERRO", str(e)))
            break
        except Exception as e:                             # um PDF com problema não derruba o lote
            print(f"  ERRO em {pdf.name}: {e}", file=sys.stderr)
            resultados.append((n, "ERRO", str(e)))

    print(f"\n===== RESUMO ({time.time() - t0:.0f}s) =====")
    for n, status, erro in resultados:
        print(f"  {a.prefixo} ({n}).pdf: {status}{' - ' + erro if erro else ''}")
    ok = sum(1 for _, s, _ in resultados if s.startswith("ok"))
    print(f"\n{ok}/{len(resultados)} processados. Saída em: {Path(a.saida).resolve()}")
    raise SystemExit(0 if all(s.startswith("ok") for _, s, _ in resultados) else 1)


if __name__ == "__main__":
    main()