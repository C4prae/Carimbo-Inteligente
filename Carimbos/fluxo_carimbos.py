#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fluxo_carimbos.py
=================
Fluxo completo para carimbar certidões de nascimento SINTÉTICAS:

  PDF ──► imagem ──► Ollama + Qwen (visão) lê o documento e extrai os dados do cartório
      ──► validação/normalização do que a IA devolveu
      ──► Faker completa SÓ o que não foi possível extrair (e registra a origem de cada campo)
      ──► carimbos coerentes com esses dados (layout aleatório, ruído realista de 0.1 a 0.6)
      ──► detecta o texto impresso e procura, no documento inteiro, um local com espaço livre
          (o carimbo nunca cobre um texto por inteiro; roçar levemente é tolerado e configurável)
      ──► compõe o carimbo "dentro da digitalização": tinta esmaecida, textura do papel,
          mesma nitidez e grão do scan (sem cores chapadas)
      ──► salva  • o documento carimbado como PNG            -> <saida>/docs/
                 • cada carimbo como PNG com fundo transparente -> <saida>/Stamps/
                 • um JSON com tudo (resposta da IA, dados finais, origem, layout, ruído)

Pré-requisitos
--------------
    pip install pillow numpy opencv-python faker pypdfium2
    ollama pull qwen2.5vl:7b          # modelo de VISÃO da família Qwen (ou qwen3-vl:8b etc.)
    ollama serve                       # se o Ollama ainda não estiver rodando

Arquivos que precisam estar na mesma pasta: gerar_carimbo.py, carimbo_imperfeito.py
e integracao_documento.py

Exemplos
--------
    python fluxo_carimbos.py certidao.pdf
    python fluxo_carimbos.py certidao.pdf --modelo qwen3-vl:8b --saida saida_fluxo --semente 7
    python fluxo_carimbos.py certidao.pdf --debug-mapa                # salva também o mapa de texto/espaço
    python fluxo_carimbos.py certidao.pdf --prob-sobreposicao 0       # nunca encostar no texto
    python fluxo_carimbos.py certidao.pdf --saturacao 0.4 --opacidade-tinta 0.6   # tinta mais discreta
    python fluxo_carimbos.py certidao.pdf --so-extrair                # só mostra o que a IA leu
    python fluxo_carimbos.py certidao.pdf --extraidos saida_fluxo/docs/certidao_metadados.json
                                                                      # reaproveita a extração (sem Ollama)
"""
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

import gerar_carimbo as gc
import integracao_documento as idoc

# =========================================================================== #
#  CONFIGURAÇÃO
# =========================================================================== #
OLLAMA_HOST = "http://localhost:11434"
MODELO = "qwen2.5vl:7b"          # modelo Qwen com visão (necessário: os campos estão na imagem)
TIMEOUT_S = 900                  # modelos de visão locais podem demorar
NUM_CTX = 8192                   # janela de contexto pedida ao Ollama (a imagem usa ~2-3k tokens)
LADO_MAX_MODELO = 1600           # lado maior (px) das imagens enviadas ao modelo
RECORTE_RODAPE = 0.40            # fração inferior da página enviada ampliada (bloco do cartório); 0 = desliga
DPI = 200                        # resolução do PNG final do documento

RUIDO_MIN, RUIDO_MAX = gc.RUIDO_MIN, gc.RUIDO_MAX     # 0.1 a 0.6
CARIMBOS = ("redondo", "retangular")                  # carimbos criados para o documento
PASTA_DOCS = "docs"                                   # documento carimbado (+ metadados e mapa de debug)
PASTA_CARIMBOS = "Stamps"                             # um PNG transparente por carimbo

# Tamanho do carimbo na página (fração da largura): sorteado nesta faixa para cada carimbo.
LARGURA_REL = {"redondo": (0.14, 0.21), "retangular": (0.26, 0.38)}
ESCALA_MIN = 0.55            # se não houver espaço, o carimbo é reduzido até esta fração do tamanho sorteado
MARGEM_PAGINA = 0.01         # distância mínima da borda da página (fração da largura)

# Quanto o carimbo pode "roçar" o texto impresso. Com PROB_SOBREPOSICAO = 0 ele só usa espaço livre;
# com 0.3, em ~30% dos carimbos é aceita uma sobreposição mínima (nunca uma palavra inteira).
PROB_SOBREPOSICAO = 0.30

# Aparência de digitalização (veja integracao_documento.compor_na_digitalizacao)
APARENCIA = {
    "saturacao": 0.62,       # 1 = cor original da tinta; menor = mais acinzentada/esmaecida
    "clareza": 0.18,         # clareia a tinta (carimbo nunca mais escuro que o texto impresso)
    "opacidade": 0.80,       # opacidade máxima da tinta sobre o papel
    "papel": 0.55,           # quanto a textura do papel modula a tinta
    "grao": 0.012,           # grão do scanner dentro da tinta
}

# Campos que pedimos ao modelo
CAMPOS_MODELO = ["cns", "matricula", "titulo_cartorio", "oficial", "escrevente",
                 "cargo", "cidade", "uf", "endereco", "data_emissao"]
SCHEMA = {"type": "object",
          "properties": {k: {"type": "string"} for k in CAMPOS_MODELO},
          "required": CAMPOS_MODELO}

PROMPT_SISTEMA = ("Você extrai dados de imagens de documentos e responde SOMENTE com um objeto "
                  "JSON válido, sem comentários e sem markdown.")
PROMPT_USUARIO = """A imagem é uma CERTIDÃO DE NASCIMENTO SINTÉTICA (fictícia, com marca d'água "DOCUMENTO SINTÉTICO").
Extraia APENAS os dados do cartório e da emissão da certidão (NÃO os dados da pessoa registrada):

- cns: número do campo "CNS" (código do cartório, 6 dígitos)
- matricula: número do campo "Matrícula", exatamente como aparece (com pontos e hífen)
- titulo_cartorio: texto do cabeçalho do cartório (ex.: REGISTRO CIVIL DAS PESSOAS NATURAIS)
- oficial: campo "Nome do Oficial"
- escrevente: campo "Nome do Oficial/Escrevente"
- cargo: campo "Cargo"
- cidade: cidade do cartório (campo "Cidade · UF"), SEM a sigla da UF
- uf: sigla da UF (2 letras) desse mesmo campo
- endereco: campo "Endereço", completo
- data_emissao: data do campo "Local e data de emissão", no formato DD/MM/AAAA (converta datas por extenso)

Regras: copie os textos exatamente como aparecem. Se um campo não estiver visível ou legível,
devolva "" (string vazia). NUNCA invente valores."""
PROMPT_RECORTE = ("\n\nA 1ª imagem é a página inteira. A 2ª é um recorte AMPLIADO do rodapé "
                  "(bloco do cartório): use-a para ler com precisão os campos pequenos.")


# --------------------------------------------------------------------------- #
# 1) Documento -> imagens para o modelo
# --------------------------------------------------------------------------- #
def preparar_imagens(pagina, lado_max=LADO_MAX_MODELO, recorte_rodape=RECORTE_RODAPE):
    """Página inteira (reduzida) + recorte do rodapé (quase em resolução original)."""
    def reduz(im):
        s = min(1.0, lado_max / max(im.size))
        return im if s >= 1.0 else im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)

    imagens = [reduz(pagina)]
    if recorte_rodape and recorte_rodape > 0:
        h = pagina.height
        imagens.append(reduz(pagina.crop((0, int(h * (1 - recorte_rodape)), pagina.width, h))))
    return imagens


def _b64(img, qualidade=95):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=qualidade)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# 2) Ollama
# --------------------------------------------------------------------------- #
def _requisicao(url, payload=None, timeout=30):
    corpo = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=corpo, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        texto = e.read().decode("utf-8", "replace")
        try:
            texto = json.loads(texto).get("error", texto)
        except ValueError:
            pass
        raise RuntimeError(f"O Ollama respondeu HTTP {e.code}: {texto}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Não consegui conectar ao Ollama em {url}. Ele está rodando? "
                              f"(execute: ollama serve)  [{e.reason}]") from e
    except TimeoutError as e:
        raise TimeoutError(f"O Ollama não respondeu em {timeout}s. Aumente --timeout ou use um "
                           f"modelo menor.") from e


def verificar_ollama(host=OLLAMA_HOST, modelo=MODELO):
    """Confere se o servidor responde e se o modelo está instalado."""
    tags = _requisicao(f"{host}/api/tags", timeout=15)
    nomes = [m.get("name") or m.get("model") for m in tags.get("models", [])]
    if modelo not in nomes and f"{modelo}:latest" not in nomes:
        raise RuntimeError(f"O modelo '{modelo}' não está instalado no Ollama.\n"
                           f"Instale com:  ollama pull {modelo}\n"
                           f"Modelos instalados: {', '.join(n for n in nomes if n) or '(nenhum)'}")
    return nomes


def _json_da_resposta(texto):
    """Interpreta a resposta do modelo, tolerando cercas ```json e texto em volta."""
    t = (texto or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for candidato in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0]):
        if candidato:
            try:
                obj = json.loads(candidato)
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    return None


def extrair_com_ollama(imagens, host=OLLAMA_HOST, modelo=MODELO, timeout=TIMEOUT_S, num_ctx=NUM_CTX):
    """Envia as imagens ao modelo e devolve (dict_extraido, texto_bruto_da_resposta)."""
    prompt = PROMPT_USUARIO + (PROMPT_RECORTE if len(imagens) > 1 else "")
    payload = {
        "model": modelo, "stream": False,
        "format": SCHEMA,                                   # saída estruturada (JSON Schema)
        "options": {"temperature": 0, "num_ctx": num_ctx},  # determinístico
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": prompt, "images": [_b64(i) for i in imagens]},
        ],
    }
    texto, dados = "", None
    for tentativa in (SCHEMA, "json"):                      # 2ª tentativa: modo JSON simples
        payload["format"] = tentativa
        resp = _requisicao(f"{host}/api/chat", payload, timeout=timeout)
        texto = (resp.get("message") or {}).get("content", "")
        dados = _json_da_resposta(texto)
        if dados is not None:
            return dados, texto
    raise ValueError(f"A resposta do modelo não é um JSON válido:\n{texto[:500]}")


# --------------------------------------------------------------------------- #
# 3) Validação / normalização do que a IA devolveu
# --------------------------------------------------------------------------- #
_UNIDADES = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12, "treze": 13,
    "catorze": 14, "quatorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18,
    "dezenove": 19, "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
    "setenta": 70, "oitenta": 80, "noventa": 90, "cem": 100, "cento": 100, "duzentos": 200,
    "trezentos": 300, "quatrocentos": 400, "quinhentos": 500, "seiscentos": 600,
    "setecentos": 700, "oitocentos": 800, "novecentos": 900,
}
_MESES = {m: i + 1 for i, m in enumerate(
    ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto",
     "setembro", "outubro", "novembro", "dezembro"])}
_PLACEHOLDERS = {"n/a", "na", "null", "none", "nao", "desconhecido", "ilegivel", "nao visivel",
                 "vazio", "-", "--", "nao informado", "nao legivel"}
_UFS = set(gc._CIDADES_POR_UF)
_MINUSC = {"de", "da", "do", "das", "dos", "e"}


def _limpa(txt):
    return re.sub(r"\s+", " ", str(txt or "")).strip(" \t\n.,;:")


def _titulo_pt(txt):
    return " ".join(p.lower() if (i and p.lower() in _MINUSC) else p.capitalize()
                    for i, p in enumerate(txt.split()))


def _numero_por_extenso(frase):
    """'dois mil e doze' -> 2012 | 'vinte e cinco' -> 25 | None se não entender."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", gc._sem_acento(frase)) if t and t != "e"]
    if not tokens:
        return None
    total = atual = 0
    for t in tokens:
        if t.isdigit():
            atual += int(t)
        elif t == "mil":
            total += max(atual, 1) * 1000
            atual = 0
        elif t in _UNIDADES:
            atual += _UNIDADES[t]
        else:
            return None
    return total + atual


def interpretar_data(txt):
    """Aceita 25/08/2012, 2012-08-25, '25 de agosto de 2012' e datas por extenso."""
    t = _limpa(txt)
    if not t:
        return None
    d = None
    for f in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(t, f).date()
            break
        except ValueError:
            pass
    if d is None:
        m = re.search(r"(.+?)\s+d[eo]\s+([A-Za-zÀ-ÿ]+)\s+d[eo]\s+(.+)$", t)
        if m:
            dia_t, mes_t, ano_t = m.groups()
            dia_t = dia_t.split(",")[-1]                        # tira "Manaus - AM," do começo
            dia = int(dia_t) if dia_t.strip().isdigit() else _numero_por_extenso(dia_t)
            ano = int(ano_t) if ano_t.strip().isdigit() else _numero_por_extenso(ano_t)
            mes = _MESES.get(gc._sem_acento(mes_t))
            try:
                d = date(ano, mes, dia) if (dia and mes and ano) else None
            except ValueError:
                d = None
    if d is None or not (1900 <= d.year <= date.today().year):
        return None
    return d


def _nome_valido(txt):
    t = _limpa(txt)
    if not t or gc._sem_acento(t) in _PLACEHOLDERS:
        return None
    if len(t.split()) < 2 or len(t) < 5 or not re.fullmatch(r"[A-Za-zÀ-ÿ'’.\- ]+", t):
        return None
    return _titulo_pt(t) if t.isupper() else t


def _separa_cidade_uf(txt):
    m = re.match(r"^(.*?)\s*[-–—/·,|]\s*([A-Za-z]{2})\s*$", txt)
    return (m.group(1).strip(), m.group(2).upper()) if m else (txt, "")


def normalizar_extracao(bruto):
    """
    Valida cada campo devolvido pela IA. Retorna (extraidos, descartados, notas):
      extraidos   : só campos válidos, com as chaves de DadosCartorio (cns, cidade, uf, ...)
      descartados : campos que vieram mas foram rejeitados (valor + motivo) -> irão para o Faker
      notas       : correções automáticas feitas (ex.: CNS deduzido da matrícula)
    """
    ext, desc, notas = {}, {}, []
    g = lambda k: _limpa(bruto.get(k, ""))

    def descarta(campo, motivo):
        if g(campo):
            desc[campo] = {"valor": g(campo), "motivo": motivo}

    # CNS (6 dígitos). Se faltar, os 6 primeiros dígitos da matrícula são o CNS.
    cns, mat = re.sub(r"\D", "", g("cns")), re.sub(r"\D", "", g("matricula"))
    if len(cns) == 6:
        ext["cns"] = cns
    elif len(mat) >= 12:
        ext["cns"] = mat[:6]
        notas.append("cns deduzido dos 6 primeiros dígitos da matrícula")
    else:
        descarta("cns", "não tem 6 dígitos")

    # Cidade / UF (a IA às vezes devolve "Manaus - AM" no campo cidade)
    cidade, uf_txt = g("cidade"), g("uf").upper()
    cidade, uf_da_cidade = _separa_cidade_uf(cidade)
    if uf_da_cidade:
        notas.append("UF separada do campo cidade")
    m = re.search(r"\b([A-Z]{2})\b", uf_txt or uf_da_cidade)
    uf = m.group(1) if m and m.group(1) in _UFS else None
    if uf:
        ext["uf"] = uf
    else:
        descarta("uf", "sigla de UF inválida")
    if re.fullmatch(r"[A-Za-zÀ-ÿ'’\- ]{3,60}", cidade):
        ext["cidade"] = _titulo_pt(cidade) if cidade.isupper() else cidade
    else:
        descarta("cidade", "texto de cidade inválido")

    for campo in ("oficial", "escrevente"):
        nome = _nome_valido(g(campo))
        if nome:
            ext[campo] = nome
        else:
            descarta(campo, "não parece um nome completo")

    titulo = g("titulo_cartorio").upper()
    if re.search(r"REGISTRO CIVIL|OF[IÍ]CIO|CART[OÓ]RIO", titulo) and len(titulo) <= 80:
        ext["titulo"] = titulo
    else:
        descarta("titulo_cartorio", "não parece um título de cartório")

    cargo = g("cargo").upper()
    if re.search(r"OFICIAL|REGISTRADOR|ESCREVENTE|TABELI", cargo) and len(cargo) <= 60:
        ext["cargo"] = cargo
    else:
        descarta("cargo", "cargo não reconhecido")

    end = g("endereco")
    if len(end) >= 10 and gc._sem_acento(end) not in _PLACEHOLDERS:
        ext["endereco"] = end
    else:
        descarta("endereco", "endereço curto demais")

    d = interpretar_data(g("data_emissao"))
    if d:
        ext["data_emissao"] = d.strftime("%d/%m/%Y")
    else:
        descarta("data_emissao", "data não reconhecida")
    return ext, desc, notas


# --------------------------------------------------------------------------- #
# 4) Fluxo completo
# --------------------------------------------------------------------------- #
def _log(msg):
    print(msg, flush=True)


def posicionar_carimbo(mapa, img, doc_w, ocupado, rng, larg_rel, prob_sobreposicao=PROB_SOBREPOSICAO,
                       escala_min=ESCALA_MIN):
    """
    Redimensiona o carimbo (largura = larg_rel * largura da página) e procura um lugar com espaço.
    Se não couber, reduz o carimbo aos poucos (até escala_min); em último caso aceita uma
    sobreposição um pouco maior, mas ainda sem cobrir nenhuma palavra por inteiro.
    Retorna (imagem_final, posicao_ou_None, escala).
    """
    margem = int(doc_w * MARGEM_PAGINA)
    escala = 1.0
    while True:
        larg = max(8, int(doc_w * larg_rel * escala))
        im = img.resize((larg, max(8, round(img.height * larg / img.width))), Image.LANCZOS)
        nivel = int(rng.integers(1, 4)) if rng.random() < prob_sobreposicao else 0
        pos = idoc.procurar_posicao(mapa, im, rng, ocupado=ocupado, margem_px=margem, nivel_inicial=nivel)
        if pos or escala <= escala_min + 1e-6:
            break
        escala = max(escala_min, escala * 0.88)
    if pos is None:                                   # último recurso: tolerância maior, ainda < 50% de qualquer palavra
        pos = idoc.procurar_posicao(mapa, im, rng, ocupado=ocupado, margem_px=margem,
                                    tolerancias=(0.015, 0.03), max_palavra=(0.45, 0.5))
        if pos:
            pos["ultimo_recurso"] = True
    return im, pos, escala


def executar_fluxo(pdf, saida_dir="saida_fluxo", pagina=0, modelo=MODELO, host=OLLAMA_HOST,
                   timeout=TIMEOUT_S, num_ctx=NUM_CTX, semente=None, tipos=CARIMBOS,
                   ruido_min=RUIDO_MIN, ruido_max=RUIDO_MAX, layout="aleatorio", cor=None,
                   dpi=DPI, extraidos_json=None, recorte_rodape=RECORTE_RODAPE, so_extrair=False,
                   prob_sobreposicao=PROB_SOBREPOSICAO, aparencia=None, debug_mapa=False):
    aparencia = {**APARENCIA, **(aparencia or {})}
    if not 0.0 <= ruido_min <= ruido_max <= 1.0:
        raise ValueError("Use 0 <= ruido_min <= ruido_max <= 1")
    pdf = Path(pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf}")
    semente = semente if semente is not None else random.randrange(2 ** 31)
    saida = Path(saida_dir)
    pasta_docs, pasta_carimbos = saida / PASTA_DOCS, saida / PASTA_CARIMBOS
    pasta_docs.mkdir(parents=True, exist_ok=True)
    pasta_carimbos.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(saida).as_posix()          # caminhos relativos à pasta de saída no JSON
    stem = pdf.stem

    # 1) PDF -> imagem
    _log(f"[1/5] Renderizando {pdf.name} (página {pagina + 1}, {dpi} dpi)...")
    doc = gc.renderizar_pdf(pdf, pagina, dpi)

    # 2) Extração dos dados
    if extraidos_json:
        _log(f"[2/5] Lendo extração já pronta: {extraidos_json}")
        bruto = json.loads(Path(extraidos_json).read_text(encoding="utf-8"))
        bruto = bruto.get("resposta_bruta_modelo", bruto)
        texto_bruto, modelo_usado = None, None
    else:
        _log(f"[2/5] Enviando o documento ao Ollama ({modelo})... pode levar alguns minutos")
        verificar_ollama(host, modelo)
        bruto, texto_bruto = extrair_com_ollama(preparar_imagens(doc, LADO_MAX_MODELO, recorte_rodape),
                                                host, modelo, timeout, num_ctx)
        modelo_usado = modelo
    extraidos, descartados, notas = normalizar_extracao(bruto)
    _log(f"      campos aproveitados da IA: {', '.join(extraidos) or '(nenhum)'}")
    for campo, info in descartados.items():
        _log(f"      descartado '{campo}': {info['valor']!r} ({info['motivo']})")
    if so_extrair:
        _log(json.dumps({"extraidos": extraidos, "descartados": descartados, "notas": notas},
                        ensure_ascii=False, indent=2))
        return {"extraidos": extraidos, "descartados": descartados, "notas": notas}

    # 3) Faker completa só o que faltou
    dados, origem = gc.completar_dados(extraidos, semente)
    faltantes = [c for c, o in origem.items() if o == "faker"]
    _log(f"[3/5] Dados completados com Faker: {', '.join(faltantes) or '(nenhum, tudo extraído)'}")

    # 4) Carimbos: gera, procura lugar com espaço livre e compõe "dentro" da digitalização
    _log(f"[4/5] Gerando carimbos {list(tipos)} com ruído entre {ruido_min} e {ruido_max}...")
    cor = cor or random.Random(semente).choice(["azul", "azul", "violeta", "preto"])
    rng = np.random.default_rng(semente + 7)
    doc_arr = np.asarray(doc.convert("RGB"))
    mapa = idoc.mascara_texto(doc_arr)                           # texto impresso x fundo decorativo
    _log(f"      texto impresso: {100 * mapa['texto'].mean():.1f}% da página | "
         f"moldura decorativa ignorada: {100 * mapa['deco'].mean():.1f}%")
    if debug_mapa:
        dbg = doc_arr.copy()
        dbg[mapa["deco"]] = (dbg[mapa["deco"]] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        dbg[mapa["texto"]] = (255, 0, 0)
        Image.fromarray(dbg).save(pasta_docs / f"{stem}_mapa_texto.png")

    gerados = []
    for k, tipo in enumerate(tipos):
        img, info = gc.gerar_carimbo_aleatorio(semente=semente + 1000 * (k + 1), tipo=tipo, cor=cor,
                                               ruido=(ruido_min, ruido_max), dados=dados, layout=layout)
        larg_rel = float(rng.uniform(*LARGURA_REL[tipo]))
        gerados.append((tipo, img, info, larg_rel))

    ocupado = np.zeros(doc_arr.shape[:2], bool)
    registros = []
    # os maiores primeiro (são os mais difíceis de encaixar)
    for tipo, img, info, larg_rel in sorted(gerados, key=lambda g: -(g[3] * g[1].height / g[1].width * g[3])):
        im, pos, escala = posicionar_carimbo(mapa, img, doc_arr.shape[1], ocupado, rng, larg_rel,
                                             prob_sobreposicao)
        arq = pasta_carimbos / f"{stem}_carimbo_{tipo}.png"
        im.save(arq)                                              # RGBA, fundo transparente, tamanho usado no documento
        if pos is None:
            _log(f"      AVISO {tipo}: não encontrei espaço no documento; carimbo NÃO aplicado")
            info.update({"arquivo": rel(arq), "posicao": None, "aplicado": False})
        else:
            doc_arr = idoc.compor_na_digitalizacao(doc_arr, im, pos["x"], pos["y"], semente=info["semente"],
                                                   **aparencia)
            idoc.marcar_ocupado(ocupado, im, pos["x"], pos["y"])
            info.update({"arquivo": rel(arq), "aplicado": True,
                         "posicao": {**pos, "largura_px": im.width, "altura_px": im.height,
                                     "escala": round(escala, 3)}})
            _log(f"      {tipo}: ruído {info['ruido']} | {im.width}x{im.height}px em (x={pos['x']}, y={pos['y']}) "
                 f"| sobreposição nível {pos['nivel']}"
                 f"{' (último recurso)' if pos.get('ultimo_recurso') else ''} -> {arq.name}")
        info["aparencia"] = dict(aparencia)
        registros.append(info)
    doc = Image.fromarray(doc_arr)
    registros.sort(key=lambda r: list(tipos).index(r["tipo"]))

    # 5) Documento carimbado + metadados
    arq_doc = pasta_docs / f"{stem}_carimbado.png"
    doc.save(arq_doc, dpi=(dpi, dpi))
    meta = {
        "pdf": str(pdf), "pagina": pagina, "modelo": modelo_usado, "semente": semente,
        "faixa_ruido": [ruido_min, ruido_max], "prob_sobreposicao": prob_sobreposicao,
        "resposta_bruta_modelo": bruto, "texto_bruto_modelo": texto_bruto,
        "campos_extraidos": extraidos, "campos_descartados": descartados, "notas": notas,
        "dados_finais": dados.to_dict(), "origem_dos_campos": origem,
        "carimbos": registros, "documento_carimbado": rel(arq_doc),
    }
    arq_meta = pasta_carimbos / f"{stem}_metadados.json"
    arq_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[5/5] Documento: {arq_doc}\n      Carimbos:  {pasta_carimbos}{os.sep}\n      Metadados: {arq_meta}")
    return meta


def _main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="PDF -> Ollama/Qwen -> carimbos coerentes -> PNG carimbado")
    ap.add_argument("pdf", help="PDF da certidão sintética")
    ap.add_argument("--saida", default="saida_fluxo", help=f"pasta de saída; cria '{PASTA_DOCS}/' e '{PASTA_CARIMBOS}/' dentro dela (padrão: saida_fluxo)")
    ap.add_argument("--pagina", type=int, default=0, help="página do PDF (começa em 0)")
    ap.add_argument("--modelo", default=MODELO, help=f"modelo de visão do Ollama (padrão: {MODELO})")
    ap.add_argument("--host", default=OLLAMA_HOST)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--semente", type=int, default=None, help="torna dados/layout/ruído reprodutíveis")
    ap.add_argument("--carimbos", default=",".join(CARIMBOS), help="ex.: redondo,retangular")
    ap.add_argument("--ruido-min", type=float, default=RUIDO_MIN)
    ap.add_argument("--ruido-max", type=float, default=RUIDO_MAX)
    ap.add_argument("--layout", choices=["aleatorio", "padrao"], default="aleatorio")
    ap.add_argument("--cor", choices=list(gc.CORES_TINTA), default=None, help="padrão: sorteada")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--sem-recorte", action="store_true", help="não envia o recorte ampliado do rodapé")
    ap.add_argument("--extraidos", help="JSON com a extração (pula o Ollama)")
    ap.add_argument("--so-extrair", action="store_true", help="só mostra o que a IA leu e valida")
    ap.add_argument("--prob-sobreposicao", type=float, default=PROB_SOBREPOSICAO,
                    help="chance (0 a 1) de aceitar que o carimbo roce levemente o texto impresso; 0 = só espaço livre")
    ap.add_argument("--saturacao", type=float, default=APARENCIA["saturacao"],
                    help="1 = cor original da tinta; menor = mais acinzentada (padrão %(default)s)")
    ap.add_argument("--opacidade-tinta", type=float, default=APARENCIA["opacidade"],
                    help="opacidade máxima da tinta no papel, 0 a 1 (padrão %(default)s)")
    ap.add_argument("--debug-mapa", action="store_true",
                    help="salva uma imagem mostrando o texto detectado (vermelho) e a moldura ignorada (azul)")
    a = ap.parse_args()

    tipos = tuple(t.strip() for t in a.carimbos.split(",") if t.strip())
    invalidos = [t for t in tipos if t not in LARGURA_REL]
    if invalidos:
        ap.error(f"carimbo(s) inválido(s): {invalidos}. Use: {', '.join(LARGURA_REL)}")
    try:
        executar_fluxo(a.pdf, a.saida, a.pagina, a.modelo, a.host, a.timeout, a.num_ctx, a.semente,
                       tipos, a.ruido_min, a.ruido_max, a.layout, a.cor, a.dpi, a.extraidos,
                       0.0 if a.sem_recorte else RECORTE_RODAPE, a.so_extrair,
                       a.prob_sobreposicao, {"saturacao": a.saturacao, "opacidade": a.opacidade_tinta},
                       a.debug_mapa)
    except (ConnectionError, TimeoutError, RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    _main()