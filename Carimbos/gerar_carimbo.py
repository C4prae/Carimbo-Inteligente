#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar_carimbo.py
================
Gera carimbos de cartório (redondo e/ou retangular) para documentos SINTÉTICOS
e os aplica sobre a imagem/PDF do documento.

Novidades desta versão
----------------------
* DADOS ALEATÓRIOS E ANÔNIMOS: nomes, endereço, CNS, data e cidade/UF são
  gerados com a biblioteca Faker (pt_BR). Nada vem de pessoas ou cartórios reais.
* LAYOUT ALEATÓRIO: os formatos (redondo/retangular) continuam os mesmos, mas
  a FORMA como os dados entram no carimbo é sorteada: ordem e quantidade de
  linhas, alinhamento, bordas, formato da data, rótulos, o que vai em cada arco
  do carimbo redondo, o que vai no centro, separadores etc.
* METADADOS: cada carimbo aleatório devolve um dicionário com os dados e o
  layout usados (útil como "ground truth" para treinar/avaliar OCR).
* DADOS EXTRAÍDOS + FAKER: `completar_dados()` recebe os campos extraídos de um
  documento (ex.: por uma IA via Ollama, veja fluxo_carimbos.py) e completa com
  Faker apenas o que faltar, registrando a origem de cada campo.
* RUÍDO REALISTA: `gerar_carimbo(..., ruido=0.1..0.6)` aplica o desgaste de
  carimbo_imperfeito.py (precisa de opencv-python e do arquivo na mesma pasta).

Dependências:
    pip install pillow numpy faker pypdfium2 opencv-python
    # pypdfium2: só p/ PDF | opencv-python: só p/ o ruído realista (parâmetro `ruido`)

Uso rápido (linha de comando):
    # carimbo fixo (dados do exemplo embutido), como antes
    python gerar_carimbo.py --pdf documento.pdf --saida documento_carimbado.png

    # 20 carimbos totalmente aleatórios + JSON com os metadados
    python gerar_carimbo.py --so-carimbo --aleatorio --n 20 --semente 1 --saida carimbos/c.png

    # carimbo aleatório coerente com o documento (fixa cidade/UF/CNS, sorteia o resto)
    python gerar_carimbo.py --pdf doc.pdf --aleatorio --fixar "cidade=Manaus;uf=AM;cns=127260"

Uso em código:
    from gerar_carimbo import gerar_carimbo_aleatorio, aplicar_carimbo
    carimbo, info = gerar_carimbo_aleatorio(semente=42)          # tipo/cor/layout/dados sorteados
    resultado = aplicar_carimbo(imagem_do_documento, carimbo, centro_rel=(0.72, 0.85))
"""
from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
import unicodedata
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Fator de super-amostragem: desenhamos em alta resolução e reduzimos no final
# (serrilhado menor e bordas mais naturais).
SS = 3

# Cores de tinta típicas de carimbos de cartório (RGB)
CORES_TINTA = {
    "azul": (52, 52, 140),
    "preto": (32, 32, 36),
}

# Faixa de ruído (desgaste) usada nos carimbos gerados automaticamente
RUIDO_MIN, RUIDO_MAX = 0.1, 0.6


# --------------------------------------------------------------------------- #
# Dados do cartório
# --------------------------------------------------------------------------- #
@dataclass
class DadosCartorio:
    cns: str                           # Código Nacional de Serventia (ex.: "127260")
    cidade: str
    uf: str
    oficial: str                       # Nome do Oficial titular
    cargo: str = "OFICIAL DE REGISTRO CIVIL"
    escrevente: str = ""               # opcional
    endereco: str = ""                 # endereço completo em uma string
    data_emissao: date | str | None = None
    titulo: str = "REGISTRO CIVIL DAS PESSOAS NATURAIS"

    @classmethod
    def from_dict(cls, d: dict) -> "DadosCartorio":
        """Cria a partir de um dict (ignora chaves desconhecidas)."""
        validas = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in validas})

    def to_dict(self) -> dict:
        """Dict serializável em JSON (data no formato dd/mm/aaaa)."""
        d = asdict(self)
        d["data_emissao"] = self.data_formatada
        return d

    @property
    def data_formatada(self) -> str:
        return _fmt_data(self.data_emissao, "numerico")


# Exemplo baseado no documento enviado (certidão de Manaus - AM)
EXEMPLO = DadosCartorio(
    cns="127260",
    cidade="Manaus",
    uf="AM",
    oficial="Letícia Nunes Rodrigues",
    cargo="OFICIAL DE REGISTRO CIVIL",
    escrevente="Júlia Ferreira Martins",
    endereco="Avenida Eduardo Ribeiro, nº 2979, Centro, CEP 69005-040, Manaus - AM",
    data_emissao=date(2012, 8, 25),
)


# --------------------------------------------------------------------------- #
# Formatação de textos (usada pelo layout)
# --------------------------------------------------------------------------- #
_MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
          "agosto", "setembro", "outubro", "novembro", "dezembro"]


def _como_data(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for f in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(v, f).date()
            except ValueError:
                pass
    return None


def _fmt_data(v, formato: str = "numerico") -> str:
    """formato: 'numerico' 25/08/2012 | 'extenso' 25 de agosto de 2012 |
    'pontos' 25.08.12 | 'hifen' 25-08-2012"""
    dt = _como_data(v)
    if dt is None:
        return str(v) if v else ""
    if formato == "extenso":
        return f"{dt.day} de {_MESES[dt.month - 1]} de {dt.year}"
    if formato == "pontos":
        return dt.strftime("%d.%m.%y")
    if formato == "hifen":
        return dt.strftime("%d-%m-%Y")
    return dt.strftime("%d/%m/%Y")


def _fmt_local(cidade: str, uf: str, forma: str = "hifen") -> str:
    u = uf.upper()
    return {"hifen": f"{cidade} - {u}", "barra": f"{cidade}/{u}",
            "ponto": f"{cidade} · {u}", "parenteses": f"{cidade} ({u})"}.get(forma, f"{cidade} - {u}")


def _nome(txt: str, caixa: str) -> str:
    return txt.upper() if caixa == "upper" else txt


# --------------------------------------------------------------------------- #
# Layout: COMO os dados entram no carimbo
# --------------------------------------------------------------------------- #
@dataclass
class LayoutCarimbo:
    """Os valores padrão reproduzem o carimbo original (sem aleatoriedade)."""
    # ---- comuns aos dois formatos ----
    caixa_nomes: str = "title"          # 'title' (Maria Silva) | 'upper' (MARIA SILVA)
    formato_data: str = "numerico"      # 'numerico' | 'extenso' | 'pontos' | 'hifen'
    rotulo_cns: str = "CNS"             # 'CNS' | 'CNS nº' | 'Cód. CNS' | 'Serventia'
    forma_local: str = "hifen"          # 'hifen' | 'barra' | 'ponto' | 'parenteses'
    separador_linha: str = " · "        # entre CNS e cidade etc.
    escala_fonte: float = 1.0           # variação geral do tamanho das letras
    # ---- retangular ----
    ordem: tuple = ("titulo", "serventia", "filete", "responsavel", "endereco", "data")
    alinhamento: str = "centro"         # 'centro' | 'esquerda'
    borda: str = "dupla_arredondada"    # 'dupla_arredondada' | 'simples_arredondada' | 'dupla_reta' | 'simples_reta'
    rotulo_oficial: str = ""            # '' | 'Oficial: ' | 'Oficial Titular: '
    mostrar_cargo: bool = True
    mostrar_escrevente: bool = True
    rotulo_escrevente: str = "Escrevente: "
    prefixo_data: str = "{cidade}, "    # '{cidade}, ' | 'Emitido em ' | 'Data: ' | ''
    serventia_em_duas_linhas: bool = False
    data_na_serventia: bool = False     # junta a data na linha do CNS
    # ---- redondo ----
    arco_superior: str = "titulo"       # 'titulo' | 'curto_oficial' (REGISTRO CIVIL - NOME)
    arco_inferior: str = "local"        # 'local' | 'cns_local' | 'oficial' | 'local_data' | 'data' | 'cns'
    centro: str = "cns"                 # 'cns' | 'data' | 'cidade' | 'cns_cidade' | 'cargo'
    aneis: str = "completo"             # 'completo' | 'simples' | 'sem_interno' | 'duplo_fino'
    separador_arco: str = "estrelas"    # 'estrelas' | 'pontos' | 'traco' | 'nenhum'
    estrela_centro: bool = True
    dou_fe: bool = True


# Que informação cada opção do carimbo redondo mostra (para evitar repetir o mesmo dado)
_INFO_SUP = {"titulo": {"titulo"}, "curto_oficial": {"titulo", "oficial"}}
_INFO_CEN = {"cns": {"cns"}, "data": {"data"}, "cidade": {"local", "cns"},
             "cns_cidade": {"cns", "local"}, "cargo": {"cargo", "cns"}}
_INFO_INF = {"local": {"local"}, "cns_local": {"cns", "local"}, "oficial": {"oficial"},
             "local_data": {"local", "data"}, "data": {"data"}, "cns": {"cns"}}


def sortear_layout(tipo: str, rng: random.Random | None = None,
                   semente: int | None = None) -> LayoutCarimbo:
    """Sorteia um layout plausível para o tipo 'redondo' ou 'retangular'."""
    rng = rng or random.Random(semente)
    L = LayoutCarimbo()
    L.caixa_nomes = rng.choice(["title", "title", "upper"])
    L.formato_data = rng.choice(["numerico", "numerico", "extenso", "pontos", "hifen"])
    L.rotulo_cns = rng.choice(["CNS", "CNS", "CNS nº", "Cód. CNS", "Serventia"])
    L.forma_local = rng.choice(["hifen", "hifen", "barra", "ponto", "parenteses"])
    L.separador_linha = rng.choice([" · ", " · ", " - ", " | ", " • "])
    L.escala_fonte = round(rng.uniform(0.92, 1.08), 3)

    if tipo == "retangular":
        blocos = ["titulo", "serventia", "responsavel"]
        if rng.random() < 0.65:
            blocos.append("endereco")
        if rng.random() < 0.90:
            blocos.append("data")
        rng.shuffle(blocos)
        if rng.random() < 0.70:                      # título costuma vir primeiro
            blocos.remove("titulo")
            blocos.insert(0, "titulo")
        if "data" in blocos and rng.random() < 0.25:  # data na mesma linha do CNS
            blocos.remove("data")
            L.data_na_serventia = True
        p_filete = rng.choice([0.0, 0.35, 0.7, 1.0])
        ordem = []
        for i, b in enumerate(blocos):
            if i and rng.random() < p_filete:
                ordem.append("filete")
            ordem.append(b)
        L.ordem = tuple(ordem)
        L.alinhamento = rng.choice(["centro", "centro", "esquerda"])
        L.borda = rng.choice(["dupla_arredondada", "simples_arredondada", "dupla_reta", "simples_reta"])
        L.rotulo_oficial = rng.choice(["", "", "Oficial: ", "Oficial Titular: "])
        L.mostrar_cargo = rng.random() < 0.80
        L.mostrar_escrevente = rng.random() < 0.55
        L.rotulo_escrevente = rng.choice(["Escrevente: ", "Escrevente Autorizado(a): ", "Resp.: "])
        L.prefixo_data = rng.choice(["{cidade}, ", "Emitido em ", "Data: ", ""])
        L.serventia_em_duas_linhas = rng.random() < 0.30
    else:
        for _ in range(20):                           # sorteia sem repetir informação
            sup = rng.choice(["titulo", "titulo", "titulo", "curto_oficial"])
            cen = rng.choice(["cns", "cns", "cns", "data", "cidade", "cns_cidade", "cargo"])
            if not (_INFO_SUP[sup] & _INFO_CEN[cen]):
                break
        usado = _INFO_SUP[sup] | _INFO_CEN[cen]
        peso = ["local", "local", "cns_local", "oficial", "local_data", "data", "cns"]
        cand = [b for b in peso if not (_INFO_INF[b] & usado)] or ["local"]
        L.arco_superior, L.centro, L.arco_inferior = sup, cen, rng.choice(cand)
        L.aneis = rng.choice(["completo", "completo", "sem_interno", "simples", "duplo_fino"])
        L.separador_arco = rng.choice(["estrelas", "estrelas", "pontos", "traco", "nenhum"])
        L.estrela_centro = rng.random() < 0.75
        L.dou_fe = rng.random() < 0.70
    return L


# --------------------------------------------------------------------------- #
# Dados aleatórios e anônimos (Faker)
# --------------------------------------------------------------------------- #
# Cidades reais por UF (só o NOME da cidade é real; nomes de pessoas, CNS,
# endereços e datas são todos fictícios). Garante pares cidade/UF coerentes,
# o que o Faker sozinho não faz.
_CIDADES_POR_UF = {
    "AC": ["Rio Branco", "Cruzeiro do Sul"], "AL": ["Maceió", "Arapiraca"],
    "AP": ["Macapá", "Santana"], "AM": ["Manaus", "Parintins", "Itacoatiara"],
    "BA": ["Salvador", "Feira de Santana", "Vitória da Conquista"],
    "CE": ["Fortaleza", "Sobral", "Juazeiro do Norte"], "DF": ["Brasília"],
    "ES": ["Vitória", "Vila Velha", "Serra"], "GO": ["Goiânia", "Anápolis", "Rio Verde"],
    "MA": ["São Luís", "Imperatriz"], "MT": ["Cuiabá", "Rondonópolis"],
    "MS": ["Campo Grande", "Dourados"], "MG": ["Belo Horizonte", "Uberlândia", "Juiz de Fora"],
    "PA": ["Belém", "Santarém"], "PB": ["João Pessoa", "Campina Grande"],
    "PR": ["Curitiba", "Londrina", "Maringá"], "PE": ["Recife", "Caruaru", "Petrolina"],
    "PI": ["Teresina", "Parnaíba"], "RJ": ["Rio de Janeiro", "Niterói", "Petrópolis"],
    "RN": ["Natal", "Mossoró"], "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas"],
    "RO": ["Porto Velho", "Ji-Paraná"], "RR": ["Boa Vista"],
    "SC": ["Florianópolis", "Joinville", "Blumenau"], "SP": ["São Paulo", "Campinas", "Santos"],
    "SE": ["Aracaju", "Itabaiana"], "TO": ["Palmas", "Araguaína"],
}
_TITULOS = ["REGISTRO CIVIL DAS PESSOAS NATURAIS"] * 3 + [
    "CARTÓRIO DE REGISTRO CIVIL", "OFÍCIO DE REGISTRO CIVIL", "REGISTRO CIVIL DE PESSOAS NATURAIS"]
_CARGOS = ["OFICIAL DE REGISTRO CIVIL"] * 3 + [
    "OFICIAL DO REGISTRO CIVIL", "REGISTRADOR CIVIL", "OFICIAL TITULAR"]
_LOGRADOUROS = ["Rua", "Rua", "Rua", "Avenida", "Avenida", "Travessa", "Alameda", "Praça"]
_BAIRROS_RESERVA = ["Centro", "Centro", "Jardim América", "Vila Nova", "São José", "Boa Vista"]


def _nome_completo(fake) -> str:
    """Nome fictício no padrão brasileiro: 1 nome + (opcional) 2º nome + 1 ou 2 sobrenomes."""
    fem = fake.pybool()
    primeiro = fake.first_name_female() if fem else fake.first_name_male()
    partes = [primeiro]
    if fake.random_int(1, 100) <= 25:
        partes.append(fake.first_name_female() if fem else fake.first_name_male())
    n_sob = 2 if fake.random_int(1, 100) <= 75 else 1
    sobrenomes: list[str] = []
    while len(sobrenomes) < n_sob:
        s = fake.last_name()
        if s not in sobrenomes:
            sobrenomes.append(s)
    return " ".join(partes + sobrenomes)


def gerar_dados_aleatorios(semente: int | None = None, **fixos) -> DadosCartorio:
    """
    Gera dados 100% fictícios com Faker (pt_BR).

    semente : torna o resultado reprodutível (mesma semente -> mesmos dados).
    **fixos : campos de DadosCartorio que você quer manter (ex.: cidade='Manaus',
              uf='AM', cns='127260'); o restante continua sendo sorteado. Útil para
              deixar o carimbo coerente com o documento.
    """
    try:
        from faker import Faker
    except ImportError as e:  # pragma: no cover
        raise ImportError("Para dados aleatórios instale o Faker:  pip install faker") from e

    fake = Faker("pt_BR")
    if semente is not None:
        fake.seed_instance(semente)

    uf = str(fixos.get("uf") or fake.estado_sigla()).upper()
    cidade = fixos.get("cidade") or fake.random_element(_CIDADES_POR_UF.get(uf, ["Brasília"]))

    bairro = fake.bairro() if hasattr(fake, "bairro") else fake.random_element(_BAIRROS_RESERVA)
    logradouro = f"{fake.random_element(_LOGRADOUROS)} {fake.first_name()} {fake.last_name()}"
    numero = fake.random_int(1, 3999)
    cep = fake.numerify("%####-###")
    endereco = fake.random_element([
        f"{logradouro}, nº {numero}, {bairro}, CEP {cep}, {cidade} - {uf}",
        f"{logradouro}, {numero} - {bairro}, {cidade}/{uf}",
        f"{logradouro}, {numero}, {bairro} - CEP {cep}",
    ])

    titulo = fake.random_element(_TITULOS + [f"{fake.random_int(1, 5)}º OFÍCIO DE REGISTRO CIVIL"])
    dados = dict(
        cns=fake.numerify("%#####"),
        cidade=cidade,
        uf=uf,
        oficial=_nome_completo(fake),
        cargo=fake.random_element(_CARGOS),
        escrevente=_nome_completo(fake) if fake.random_int(1, 100) <= 70 else "",
        endereco=endereco,
        data_emissao=fake.date_between(start_date=date(2010, 1, 1), end_date=date(2025, 12, 31)),
        titulo=titulo,
    )
    dados.update({k: v for k, v in fixos.items() if k in dados})
    return DadosCartorio(**dados)


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower().strip()


_UF_POR_CIDADE = {_sem_acento(c): uf for uf, cs in _CIDADES_POR_UF.items() for c in cs}
CAMPOS_DADOS = [f.name for f in fields(DadosCartorio)]


def completar_dados(extraidos: dict | None, semente: int | None = None):
    """
    Combina os campos EXTRAÍDOS de um documento com dados sintéticos do Faker.

    extraidos : dict com qualquer subconjunto dos campos de DadosCartorio
                (cns, cidade, uf, oficial, escrevente, cargo, endereco, data_emissao, titulo).
                Valores vazios/None são tratados como "não extraído".
    Retorna (DadosCartorio, origem). `origem[campo]` é 'extraido', 'faker' ou
    'deduzido_da_cidade' (UF inferida a partir de uma cidade conhecida, para não gerar
    um par cidade/UF incoerente).
    """
    fixos = {k: v for k, v in (extraidos or {}).items()
             if k in CAMPOS_DADOS and v not in (None, "")}
    uf_deduzida = False
    if "cidade" in fixos and "uf" not in fixos:
        uf = _UF_POR_CIDADE.get(_sem_acento(fixos["cidade"]))
        if uf:
            fixos["uf"], uf_deduzida = uf, True
    dados = gerar_dados_aleatorios(semente, **fixos)
    origem = {c: ("extraido" if c in fixos else "faker") for c in CAMPOS_DADOS}
    if uf_deduzida:
        origem["uf"] = "deduzido_da_cidade"
    return dados, origem


# --------------------------------------------------------------------------- #
# Fontes
# --------------------------------------------------------------------------- #
_CAND_NEGRITO = [
    "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_CAND_REGULAR = [
    "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@lru_cache(maxsize=256)
def _fonte(tamanho: int, negrito: bool = True) -> ImageFont.FreeTypeFont:
    tamanho = max(6, int(tamanho))
    for caminho in (_CAND_NEGRITO if negrito else _CAND_REGULAR):
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)  # último recurso


def _ajustar_fonte(texto: str, tamanho: int, negrito: bool, largura_max: float):
    """Reduz a fonte até o texto caber na largura máxima."""
    tamanho = int(tamanho)
    f = _fonte(tamanho, negrito)
    while f.getlength(texto) > largura_max and tamanho > 8:
        tamanho -= 1
        f = _fonte(tamanho, negrito)
    return f


# --------------------------------------------------------------------------- #
# Primitivas de desenho
# --------------------------------------------------------------------------- #
def _bbox(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def _estrela(draw: ImageDraw.ImageDraw, cx, cy, raio, pontas=5):
    pts = []
    for i in range(pontas * 2):
        ang = -math.pi / 2 + i * math.pi / pontas
        r = raio if i % 2 == 0 else raio * 0.42
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=255)


def _texto_em_arco(mask, centro, raio, texto, fonte, theta_centro_deg,
                   inferior=False, tracking=0.0):
    """
    Desenha texto ao longo de um arco.
    theta é medido a partir do topo (12h), em sentido horário.
    Texto superior: letras "em pé" voltadas para fora.
    Texto inferior: lê-se da esquerda p/ direita, letras voltadas para o centro.
    Retorna (theta_inicio, theta_fim) em graus (ocupação angular do texto).
    """
    cx, cy = centro
    larguras = [fonte.getlength(c) for c in texto]
    total = sum(larguras) + tracking * (len(texto) - 1)
    meia_abertura = math.degrees(total / raio) / 2
    tam = int(fonte.size * 2) + 4
    acum = 0.0
    for ch, w in zip(texto, larguras):
        meio = acum + w / 2
        delta = (meio - total / 2) / raio                    # rad (neg. = esquerda)
        theta = math.radians(theta_centro_deg) + (-delta if inferior else delta)
        acum += w + tracking
        if ch == " ":
            continue
        t_deg = math.degrees(theta)
        rot = (180 - t_deg) if inferior else (-t_deg)
        ci = Image.new("L", (tam, tam), 0)
        ImageDraw.Draw(ci).text((tam / 2, tam / 2), ch, font=fonte, fill=255, anchor="mm")
        ci = ci.rotate(rot, resample=Image.BICUBIC)
        x = cx + raio * math.sin(theta)
        y = cy - raio * math.cos(theta)
        mask.paste(255, (int(round(x - tam / 2)), int(round(y - tam / 2))), ci)
    return theta_centro_deg - meia_abertura, theta_centro_deg + meia_abertura


def _fs_para_arco(texto, raio, span_graus, negrito=True, tracking_rel=0.06):
    """Tamanho de fonte para o texto ocupar `span_graus` no raio dado."""
    ref = 100
    f = _fonte(ref, negrito)
    larg = sum(f.getlength(c) for c in texto) + tracking_rel * ref * (len(texto) - 1)
    comprimento_alvo = math.radians(span_graus) * raio
    return comprimento_alvo / larg * ref


# --------------------------------------------------------------------------- #
# Carimbo redondo
# --------------------------------------------------------------------------- #
# Raio (em frações de R) da área livre no miolo, conforme o estilo de anéis
_R_INTERNO = {"completo": 0.62, "simples": 0.62, "sem_interno": 0.66, "duplo_fino": 0.58}


def _itens_centro(d: DadosCartorio, L: LayoutCarimbo, sintetico: bool) -> list:
    """Monta a pilha vertical do miolo: ('estrela', r) | ('texto', txt, fs, negrito) | ('filete', meia_larg)."""
    rot = L.rotulo_cns.upper()
    cns_linha = f"{rot} {d.cns}"
    data = _fmt_data(d.data_emissao, "numerico" if L.formato_data == "extenso" else L.formato_data)
    it: list = []
    if L.estrela_centro:
        it.append(("estrela", 0.09))
    if L.centro == "data":
        it += [("texto", "EMISSÃO", 0.09, True), ("texto", data, 0.17, True), ("filete", 0.45)]
    elif L.centro == "cidade":
        it += [("texto", d.cidade.upper(), 0.17, True), ("texto", d.uf.upper(), 0.12, True),
               ("filete", 0.45), ("texto", cns_linha, 0.09, True)]
    elif L.centro == "cns_cidade":
        it += [("texto", rot, 0.09, True), ("texto", d.cns, 0.17, True), ("filete", 0.45),
               ("texto", d.cidade.upper(), 0.11, True)]
    elif L.centro == "cargo":
        for ln in textwrap.wrap(d.cargo.upper(), width=12):
            it.append(("texto", ln, 0.12, True))
        it += [("filete", 0.45), ("texto", cns_linha, 0.09, True)]
    else:  # 'cns' (padrão)
        it += [("texto", rot, 0.10, True), ("texto", d.cns, 0.19, True), ("filete", 0.45)]
    if L.dou_fe and L.centro in ("cns", "data", "cargo"):
        it.append(("texto", "DOU FÉ", 0.12, True))
    if sintetico:
        it.append(("texto", "SINTÉTICO", 0.075, True))
    return it


def _desenhar_pilha(dr, cx, cy, R, itens, r_in):
    """Empilha os itens centralizados dentro do círculo interno, ajustando a fonte à corda do círculo."""
    esp = 0.04 * R
    alt = []
    for it in itens:
        if it[0] == "estrela":
            alt.append(2 * it[1] * R)
        elif it[0] == "filete":
            alt.append(0.02 * R)
        else:
            alt.append(it[2] * R * 1.15)
    total = sum(alt) + esp * (len(itens) - 1)
    k = min(1.0, (2 * r_in * R * 0.92) / total)               # encolhe se não couber
    y = cy - total * k / 2 - 0.01 * R
    for it, h in zip(itens, alt):
        h *= k
        yc = y + h / 2
        if it[0] == "estrela":
            _estrela(dr, cx, yc, it[1] * R * k)
        elif it[0] == "filete":
            meia = min(it[1] * R, math.sqrt(max((r_in * R) ** 2 - (yc - cy) ** 2, 0)) * 0.95)
            dr.line((cx - meia, yc, cx + meia, yc), fill=255, width=max(1, int(0.012 * R)))
        else:
            _, txt, fs, neg = it
            fs_px = fs * R * k
            borda_y = abs(yc - cy) + fs_px * 0.55
            corda = 2 * math.sqrt(max((r_in * R) ** 2 - borda_y ** 2, (0.12 * R) ** 2))
            f = _ajustar_fonte(txt, int(fs_px), neg, corda * 0.98)
            dr.text((cx, yc), txt, font=f, fill=255, anchor="mm")
        y += h + esp * k


def _desenhar_redondo(d: DadosCartorio, D: int, sintetico: bool, L: LayoutCarimbo) -> Image.Image:
    mask = Image.new("L", (D, D), 0)
    dr = ImageDraw.Draw(mask)
    cx = cy = D / 2
    R = D / 2 - D * 0.02

    def anel(r_rel, larg_rel):
        dr.ellipse(_bbox(cx, cy, R * r_rel), outline=255,
                   width=max(1 if larg_rel < 0.02 else 2, int(larg_rel * R)))

    # Anéis (conforme o estilo sorteado)
    estilo = L.aneis if L.aneis in _R_INTERNO else "completo"
    anel(1.0, 0.035)
    if estilo in ("completo", "sem_interno", "duplo_fino"):
        anel(0.93, 0.012)
    if estilo in ("completo", "simples"):
        anel(0.70, 0.022)
    if estilo == "duplo_fino":
        anel(0.70, 0.012)
        anel(0.665, 0.008)

    # Textos dos arcos
    r_txt = R * 0.815
    titulo = d.titulo.upper()
    oficial = d.oficial.upper()
    local = _fmt_local(d.cidade, d.uf, L.forma_local).upper()
    fdata = "numerico" if L.formato_data == "extenso" else L.formato_data   # extenso não cabe no arco
    data = _fmt_data(d.data_emissao, fdata)
    cns = f"{L.rotulo_cns} {d.cns}".upper()
    sep = L.separador_linha

    sup = f"REGISTRO CIVIL - {oficial}" if L.arco_superior == "curto_oficial" else titulo
    inf_map = {"local": local, "cns_local": f"{cns}{sep}{local}", "oficial": oficial,
               "local_data": f"{local}{sep}{data}", "data": data, "cns": cns}
    inf = inf_map.get(L.arco_inferior, local)

    fs_sup = min(_fs_para_arco(sup, r_txt, 190), R * 0.20)
    if fs_sup < 0.095 * R and sup != titulo:            # texto longo demais -> usa só o título
        sup = titulo
        fs_sup = min(_fs_para_arco(sup, r_txt, 190), R * 0.20)
    fs_inf = min(fs_sup, _fs_para_arco(inf, r_txt, 120))
    if fs_inf < 0.085 * R and inf != local:             # idem para o arco inferior
        inf = local
        fs_inf = min(fs_sup, _fs_para_arco(inf, r_txt, 120))

    f_sup, f_inf = _fonte(fs_sup, True), _fonte(fs_inf, True)
    _, t_fim_top = _texto_em_arco(mask, (cx, cy), r_txt, sup, f_sup, 0, tracking=0.06 * fs_sup)
    t_ini_bot, _ = _texto_em_arco(mask, (cx, cy), r_txt, inf, f_inf, 180,
                                  inferior=True, tracking=0.06 * fs_inf)

    # Separadores nas duas "folgas" laterais entre os arcos
    meio = (t_fim_top + t_ini_bot) / 2
    for th in (meio, -meio):
        a = math.radians(th)
        px, py = cx + r_txt * math.sin(a), cy - r_txt * math.cos(a)
        if L.separador_arco == "estrelas":
            _estrela(dr, px, py, R * 0.055)
        elif L.separador_arco == "pontos":
            dr.ellipse(_bbox(px, py, R * 0.03), fill=255)
        elif L.separador_arco == "traco":
            r1, r2 = R * 0.75, R * 0.88
            dr.line((cx + r1 * math.sin(a), cy - r1 * math.cos(a),
                     cx + r2 * math.sin(a), cy - r2 * math.cos(a)),
                    fill=255, width=max(1, int(0.016 * R)))

    # Miolo
    _desenhar_pilha(dr, cx, cy, R, _itens_centro(d, L, sintetico), _R_INTERNO[estilo])
    return mask


# --------------------------------------------------------------------------- #
# Carimbo retangular
# --------------------------------------------------------------------------- #
def _linhas_retangular(d: DadosCartorio, L: LayoutCarimbo, W: int, max_w: float, sintetico: bool):
    """Lista de ('t', texto, fonte, espaço_depois) e ('f', None, None, espaço) (filete)."""
    esc = L.escala_fonte
    data_txt = _fmt_data(d.data_emissao, L.formato_data)
    local = _fmt_local(d.cidade, d.uf, L.forma_local).upper()
    cns_txt = f"{L.rotulo_cns} {d.cns}"
    itens: list = []

    def add(texto, frac, negrito, esp):
        f = _ajustar_fonte(texto, frac * W * esc, negrito, max_w)
        itens.append(("t", texto, f, int(esp * W)))

    for b in L.ordem:
        if b == "filete":
            if itens and itens[-1][0] != "f":                     # sem filetes seguidos/iniciais
                itens.append(("f", None, None, int(0.016 * W)))
        elif b == "titulo":
            add(d.titulo.upper(), 0.036, True, 0.012)
        elif b == "serventia":
            if L.serventia_em_duas_linhas:
                add(cns_txt.upper(), 0.034, True, 0.004)
                add(local, 0.030, True, 0.014)
            else:
                linha = f"{cns_txt}{L.separador_linha}{local}".upper()
                if L.data_na_serventia and data_txt:
                    linha += f"{L.separador_linha}{data_txt.upper()}"
                add(linha, 0.034, True, 0.014)
        elif b == "responsavel":
            add(f"{L.rotulo_oficial}{_nome(d.oficial, L.caixa_nomes)}", 0.040, True, 0.004)
            if L.mostrar_cargo:
                add(d.cargo.upper(), 0.027, False, 0.010)
            if d.escrevente and L.mostrar_escrevente:
                add(f"{L.rotulo_escrevente}{_nome(d.escrevente, L.caixa_nomes)}", 0.026, False, 0.010)
        elif b == "endereco" and d.endereco:
            f_end = _fonte(int(0.024 * W * esc), False)
            chars = max(10, int(max_w / f_end.getlength("n")))
            linhas = textwrap.wrap(d.endereco, width=chars)
            for i, ln in enumerate(linhas):
                itens.append(("t", ln, f_end, int((0.012 if i == len(linhas) - 1 else 0.002) * W)))
        elif b == "data" and data_txt and not L.data_na_serventia:
            add(f"{L.prefixo_data.format(cidade=d.cidade)}{data_txt}", 0.030, True, 0.008)
    if itens and itens[-1][0] == "f":                              # sem filete no final
        itens.pop()
    if sintetico:
        add("DOCUMENTO SINTÉTICO", 0.022, True, 0)
    return itens


def _borda_retangular(dr, W, H, estilo):
    lw, lw2 = max(2, int(0.013 * W)), max(1, int(0.004 * W))
    ins, raio = int(0.026 * W), int(0.045 * W)
    arred = estilo.endswith("arredondada")

    def caixa(box, r, w):
        if arred:
            dr.rounded_rectangle(box, radius=r, outline=255, width=w)
        else:
            dr.rectangle(box, outline=255, width=w)

    caixa((0, 0, W - 1, H - 1), raio, lw)
    if estilo.startswith("dupla"):
        caixa((ins, ins, W - 1 - ins, H - 1 - ins), max(2, raio - ins // 2), lw2)


def _desenhar_retangular(d: DadosCartorio, W: int, sintetico: bool, L: LayoutCarimbo) -> Image.Image:
    pad = int(0.05 * W)
    max_w = W - 2 * pad
    itens = _linhas_retangular(d, L, W, max_w, sintetico)

    def alt(f):
        return int((f.size if f else 0) * 1.18)

    H = 2 * pad + sum((alt(f) if f else int(0.01 * W)) + esp for _, _, f, esp in itens)
    mask = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(mask)
    _borda_retangular(dr, W, H, L.borda)

    esq = L.alinhamento == "esquerda"
    x_txt, ancora = (pad, "la") if esq else (W / 2, "ma")
    y = pad
    for tipo, texto, f, esp in itens:
        if tipo == "f":                                            # filete
            yy = y + int(0.005 * W)
            dr.line((pad, yy, W - pad, yy), fill=255, width=max(1, int(0.005 * W)))
            y += int(0.01 * W) + esp
        else:
            dr.text((x_txt, y), texto, font=f, fill=255, anchor=ancora)
            y += alt(f) + esp
    return mask


# --------------------------------------------------------------------------- #
# Acabamento "tinta de carimbo"
# --------------------------------------------------------------------------- #
def _mapa_baixa_freq(h, w, rng, celula):
    gh, gw = max(2, h // celula), max(2, w // celula)
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _acabamento(mask: Image.Image, rng, intensidade: float, rotacao: float) -> Image.Image:
    """Pressão irregular, falhas de tinta, bordas levemente borradas e rotação."""
    w, h = mask.size
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(0.8, w * 0.0012)))
    arr = np.clip(np.asarray(mask, dtype=np.float32) / 255.0 * 1.2, 0, 1)

    pressao = 1.0 - intensidade * 0.55 * _mapa_baixa_freq(h, w, rng, celula=max(40, w // 8))
    ruido = rng.random((h, w), dtype=np.float32)
    limiar = 0.10 * intensidade * (0.4 + _mapa_baixa_freq(h, w, rng, celula=max(40, w // 6)))
    falhas = (ruido > limiar).astype(np.float32)

    arr = arr * pressao * falhas
    out = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(radius=max(0.6, w * 0.0007)))
    return out.rotate(rotacao, expand=True, resample=Image.BICUBIC)


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def gerar_carimbo(dados: DadosCartorio, tipo: str = "redondo", cor: str | tuple = "azul",
                  tamanho: int = 420, rotacao: float | None = None,
                  intensidade: float = 0.5, opacidade: float = 0.88,
                  semente: int | None = None, marca_sintetica: bool = True,
                  layout: LayoutCarimbo | str | None = None,
                  ruido: float | None = None) -> Image.Image:
    """
    Retorna o carimbo como imagem RGBA com fundo transparente.

    tipo         : "redondo" ou "retangular"
    cor          : nome em CORES_TINTA ou tupla RGB
    tamanho      : diâmetro (redondo) ou largura (retangular) final, em px
    rotacao      : graus; None = aleatório entre -8° e 8°
    intensidade  : 0 (tinta perfeita) a 1 (muito desgastado)
    opacidade    : 0–1, transparência da tinta
    semente      : torna o resultado reprodutível (tinta e, se layout='aleatorio', o layout)
    marca_sintetica : inclui a indicação "SINTÉTICO" no carimbo (recomendado,
                      para que o carimbo siga identificável como dado artificial)
    layout       : None = layout original | LayoutCarimbo | "aleatorio" (sorteia pela semente)
    ruido        : None = ruído simples (parâmetro `intensidade`). Número de 0 a 1 = desgaste
                   realista de carimbo_imperfeito.py (faixa usada no fluxo: 0.1 a 0.6)
    """
    rng = np.random.default_rng(semente)
    if rotacao is None:
        rotacao = float(rng.uniform(-8, 8))
    rgb = CORES_TINTA[cor] if isinstance(cor, str) else tuple(cor)

    if layout is None:
        layout = LayoutCarimbo()
    elif isinstance(layout, str):
        if layout != "aleatorio":
            raise ValueError("layout deve ser None, 'aleatorio' ou um LayoutCarimbo")
        layout = sortear_layout(tipo, semente=semente)

    if tipo == "redondo":
        mask = _desenhar_redondo(dados, tamanho * SS, marca_sintetica, layout)
    elif tipo == "retangular":
        mask = _desenhar_retangular(dados, tamanho * SS, marca_sintetica, layout)
    else:
        raise ValueError("tipo deve ser 'redondo' ou 'retangular'")

    # com `ruido`, o acabamento antigo só rotaciona/suaviza; o desgaste vem de carimbo_imperfeito
    mask = _acabamento(mask, rng, intensidade if ruido is None else 0.0, rotacao)
    mask = mask.resize((max(1, mask.width // SS), max(1, mask.height // SS)), Image.LANCZOS)

    alpha = np.asarray(mask, dtype=np.float32) / 255.0 * opacidade
    rgba = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgba)
    if ruido is not None:
        try:
            from carimbo_imperfeito import aplicar_efeito_em_imagem
        except ImportError as e:  # pragma: no cover
            raise ImportError("O parâmetro `ruido` requer carimbo_imperfeito.py na mesma pasta "
                              "e opencv-python (pip install opencv-python)") from e
        # a opacidade já foi aplicada acima; aqui não reduzimos de novo
        img = aplicar_efeito_em_imagem(img, intensidade=float(ruido), semente=semente, opacidade=1.0)
    return img


def gerar_carimbo_aleatorio(semente: int | None = None, tipo: str | None = None,
                            cor: str | tuple | None = None, tamanho: int | None = None,
                            ruido: float | tuple | None = None,
                            dados: DadosCartorio | None = None, fixos: dict | None = None,
                            marca_sintetica: bool = True,
                            layout: LayoutCarimbo | str | None = "aleatorio",
                            intensidade: float | None = None):
    """
    Carimbo com layout aleatório e ruído realista. Tudo que não for informado é sorteado.

    ruido  : número (0 a 1), tupla (min, max) para sortear dentro da faixa, ou None
             (= sorteia entre RUIDO_MIN e RUIDO_MAX, ou seja, 0.1 a 0.6).
    dados  : use para reaproveitar os mesmos dados em mais de um carimbo (ex.: redondo +
             retangular do mesmo documento, ou dados vindos de completar_dados()).
             None = gera com Faker a partir da semente.
    fixos  : campos a manter ao gerar os dados, ex.: {"cidade": "Manaus", "uf": "AM"}.
    layout : "aleatorio" (padrão), "padrao" (layout original) ou um LayoutCarimbo.
    intensidade : apenas por compatibilidade; equivale a `ruido`.

    Retorna (imagem_RGBA, info). `info` é um dict serializável em JSON com a semente,
    o tipo, a cor, o ruído, os dados e o layout usados (ground truth do que está no carimbo).
    """
    if semente is None:
        semente = random.randrange(2 ** 31)
    if ruido is None:
        ruido = intensidade if intensidade is not None else (RUIDO_MIN, RUIDO_MAX)
    rng = random.Random(semente)
    tipo = tipo or rng.choice(["redondo", "retangular"])
    cor_nome = cor if cor is not None else rng.choice(["azul", "azul", "violeta", "preto", "vermelho"])
    tamanho = tamanho or (420 if tipo == "redondo" else 760)
    if isinstance(ruido, (tuple, list)):
        ruido = round(rng.uniform(*ruido), 2)
    rotacao = round(rng.uniform(-8, 8), 2)

    if dados is None:
        dados = gerar_dados_aleatorios(semente, **(fixos or {}))
    if isinstance(layout, LayoutCarimbo):
        layout_obj = layout
    elif layout in (None, "padrao"):
        layout_obj = LayoutCarimbo()
    else:
        layout_obj = sortear_layout(tipo, rng=random.Random(semente + 1))

    img = gerar_carimbo(dados, tipo=tipo, cor=cor_nome, tamanho=tamanho, rotacao=rotacao,
                        semente=semente, marca_sintetica=marca_sintetica,
                        layout=layout_obj, ruido=float(ruido))
    info = {"semente": semente, "tipo": tipo,
            "cor": cor_nome if isinstance(cor_nome, str) else list(cor_nome),
            "ruido": float(ruido), "rotacao": rotacao,
            "dados": dados.to_dict(), "layout": asdict(layout_obj)}
    return img, info


def aplicar_carimbo(documento: Image.Image, carimbo: Image.Image,
                    centro_rel: tuple[float, float] = (0.72, 0.87),
                    largura_rel: float = 0.22) -> Image.Image:
    """
    Aplica o carimbo sobre o documento com mistura "multiply" (a tinta escurece
    o papel e deixa o texto por baixo visível, como um carimbo de verdade).

    centro_rel  : (x, y) do centro do carimbo, em fração da largura/altura
    largura_rel : largura do carimbo em fração da largura do documento
    """
    doc = documento.convert("RGB")
    nova_larg = int(doc.width * largura_rel)
    nova_alt = int(carimbo.height * nova_larg / carimbo.width)
    st = carimbo.resize((nova_larg, nova_alt), Image.LANCZOS)

    x0 = int(doc.width * centro_rel[0] - nova_larg / 2)
    y0 = int(doc.height * centro_rel[1] - nova_alt / 2)
    # recorte caso o carimbo ultrapasse a borda
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(doc.width, x0 + nova_larg), min(doc.height, y0 + nova_alt)
    if cx1 <= cx0 or cy1 <= cy0:
        return doc

    st_arr = np.asarray(st, dtype=np.float32)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    base = np.asarray(doc, dtype=np.float32)
    reg = base[cy0:cy1, cx0:cx1]
    alpha = st_arr[..., 3:4] / 255.0
    tinta = st_arr[..., :3] / 255.0
    base[cy0:cy1, cx0:cx1] = reg * (1 - alpha) + (reg * tinta) * alpha   # multiply
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def renderizar_pdf(caminho: str | Path, pagina: int = 0, dpi: int = 200) -> Image.Image:
    import pypdfium2 as pdfium  # import tardio: só exigido para PDFs
    pdf = pdfium.PdfDocument(str(caminho))
    return pdf[pagina].render(scale=dpi / 72).to_pil().convert("RGB")


# --------------------------------------------------------------------------- #
# Linha de comando
# --------------------------------------------------------------------------- #
def _parse_fixos(txt: str | None) -> dict:
    """'cidade=Manaus;uf=AM;cns=127260' -> {'cidade': 'Manaus', 'uf': 'AM', 'cns': '127260'}"""
    if not txt:
        return {}
    out = {}
    for par in txt.split(";"):
        if "=" in par:
            k, v = par.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_ruido(txt: str | None):
    """'0.35' -> 0.35 | '0.1:0.6' -> (0.1, 0.6) | None -> None"""
    if not txt:
        return None
    if ":" in txt:
        a, b = txt.split(":", 1)
        return (float(a), float(b))
    return float(txt)


def _main():
    ap = argparse.ArgumentParser(description="Gera/aplica carimbos de cartório sintéticos.")
    ap.add_argument("--pdf", help="PDF do documento (1ª página por padrão)")
    ap.add_argument("--imagem", help="Imagem do documento (png/jpg), alternativa ao --pdf")
    ap.add_argument("--pagina", type=int, default=0)
    ap.add_argument("--dados", help="JSON com os dados do cartório (padrão: exemplo embutido)")
    ap.add_argument("--saida", default="documento_carimbado.png")
    ap.add_argument("--tipo", choices=["redondo", "retangular", "ambos", "sorteado"], default="ambos")
    ap.add_argument("--cor", choices=list(CORES_TINTA) + ["sorteada"], default="azul")
    ap.add_argument("--semente", type=int, default=None)
    ap.add_argument("--intensidade", type=float, default=None,
                    help="0 a 1 (padrão: 0.5; no modo --aleatorio, sorteada)")
    ap.add_argument("--ruido", default=None,
                    help="Desgaste realista (requer opencv-python): '0.35' ou faixa '0.1:0.6'. "
                         "No modo --aleatorio o padrão é 0.1:0.6")
    ap.add_argument("--so-carimbo", action="store_true", help="Salva só o(s) carimbo(s) em PNG transparente")
    ap.add_argument("--aleatorio", action="store_true",
                    help="Dados (Faker) e layout aleatórios; salva um JSON com os metadados")
    ap.add_argument("--fixar", help="Campos a manter no modo --aleatorio, ex.: 'cidade=Manaus;uf=AM;cns=127260'")
    ap.add_argument("--n", type=int, default=1, help="Quantidade de documentos/carimbos a gerar")
    args = ap.parse_args()

    base_dados = EXEMPLO
    if args.dados:
        base_dados = DadosCartorio.from_dict(json.loads(Path(args.dados).read_text(encoding="utf-8")))
    fixos = _parse_fixos(args.fixar)
    cor_arg = None if args.cor == "sorteada" else args.cor
    inten = args.intensidade if args.intensidade is not None else 0.5
    ruido = _parse_ruido(args.ruido)

    if args.tipo == "ambos":
        tipos = ["redondo", "retangular"]
    elif args.tipo == "sorteado":
        tipos = [None]
    else:
        tipos = [args.tipo]

    saida = Path(args.saida)
    doc0 = None
    if args.pdf:
        doc0 = renderizar_pdf(args.pdf, args.pagina)
    elif args.imagem:
        doc0 = Image.open(args.imagem)
    so_carimbo = args.so_carimbo or doc0 is None

    # posições/larguras padrão pensadas para a área em branco no rodapé do documento
    layout_pos = {"redondo": ((0.74, 0.845), 0.20), "retangular": ((0.30, 0.842), 0.40)}
    registros = []

    for i in range(args.n):
        sem = None if args.semente is None else args.semente + i
        sufixo = f"_{i + 1:03d}" if args.n > 1 else ""
        dados_i = None
        if args.aleatorio:   # mesmos dados para todos os carimbos do mesmo documento
            dados_i = gerar_dados_aleatorios(sem if sem is not None else random.randrange(2 ** 31), **fixos)
        doc = doc0
        for k, t in enumerate(tipos):
            sem_t = None if sem is None else sem + 100000 * k
            if args.aleatorio:
                st, info = gerar_carimbo_aleatorio(semente=sem_t, tipo=t, cor=cor_arg,
                                                   ruido=ruido, dados=dados_i)
            else:
                t = t or random.choice(["redondo", "retangular"])
                r_t = round(random.uniform(*ruido), 2) if isinstance(ruido, tuple) else ruido
                st = gerar_carimbo(base_dados, tipo=t, cor=cor_arg or "azul", semente=sem_t,
                                   intensidade=inten, ruido=r_t,
                                   tamanho=420 if t == "redondo" else 760)
                info = {"tipo": t, "dados": base_dados.to_dict()}
            tipo_real = info["tipo"]
            if so_carimbo:
                destino = saida.with_name(f"{saida.stem}{sufixo}_{tipo_real}.png")
                destino.parent.mkdir(parents=True, exist_ok=True)
                st.save(destino)
                info["arquivo"] = str(destino)
                print("Salvo:", destino)
            else:
                centro, larg = layout_pos[tipo_real]
                if len(tipos) == 1:
                    centro = (0.60, centro[1])
                doc = aplicar_carimbo(doc, st, centro_rel=centro, largura_rel=larg)
            registros.append(info)
        if not so_carimbo:
            destino = saida.with_name(f"{saida.stem}{sufixo}{saida.suffix or '.png'}")
            destino.parent.mkdir(parents=True, exist_ok=True)
            doc.save(destino)
            for r in registros[-len(tipos):]:
                r["arquivo"] = str(destino)
            print("Salvo:", destino)

    if args.aleatorio:
        meta = saida.with_name(f"{saida.stem}_metadados.json")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Metadados:", meta)


if __name__ == "__main__":
    _main()