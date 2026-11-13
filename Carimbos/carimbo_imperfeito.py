#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carimbo_imperfeito.py
=====================
Aplica o efeito de "carimbo de borracha real e imperfeito" em um PNG de carimbo
(de preferência com fundo transparente).

Diferença para um ruído "padronizado": aqui os defeitos NÃO são distribuídos de
forma uniforme. O script usa
  * ruído de Perlin fractal (fBm) em várias escalas, em vez de ruído branco;
  * "mapas de zona" de baixa frequência que fazem algumas regiões do carimbo
    ficarem muito degradadas enquanto outras ficam quase intactas (como num
    carimbo real, em que um lado pega mais tinta que o outro);
  * operações morfológicas com elementos estruturantes de tamanhos aleatórios
    para criar vazios de formatos e tamanhos variados;
  * desgaste concentrado nas extremidades (distância à silhueta do carimbo).

Instalação das dependências
---------------------------
    pip install opencv-python numpy pillow

Exemplos de execução
--------------------
    python carimbo_imperfeito.py carimbo.png
    python carimbo_imperfeito.py carimbo.png -o saida.png --intensidade 0.8 --semente 42
    python carimbo_imperfeito.py carimbo.png --cor "#1c348c" --opacidade 0.85 --previa
    python carimbo_imperfeito.py carimbo.png --variantes 10        # 10 variações p/ dataset

Uso como biblioteca:
    from carimbo_imperfeito import carregar_carimbo, aplicar_efeito_carimbo_realista, salvar_carimbo
    rgb, alpha = carregar_carimbo("carimbo.png")
    rgb2, alpha2 = aplicar_efeito_carimbo_realista(rgb, alpha, intensidade=0.7, semente=1)
    salvar_carimbo("saida.png", rgb2, alpha2)

Uso "em memória" (sem arquivos), como faz o fluxo com IA (fluxo_carimbos.py):
    from carimbo_imperfeito import aplicar_efeito_em_imagem
    carimbo_gasto = aplicar_efeito_em_imagem(carimbo_pil_rgba, intensidade=0.35, semente=1)
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

# =========================================================================== #
#  PARÂMETROS AJUSTÁVEIS  (todos de 0.0 a 1.0, salvo indicação)
# =========================================================================== #
# O efeito de cada defeito = (valor individual) x INTENSIDADE_DESGASTE.
INTENSIDADE_DESGASTE = 0.55   # global: 0 = carimbo novo/perfeito ... 1 = carimbo muito gasto

QUANT_FALHAS_TINTA = 0.90     # quantidade de vazios/falhas de tinta (ruído de contato + morfologia)
DESGASTE_BORDAS = 0.80        # pedaços faltando, cortes e fraturas nas extremidades
FALTA_PRESSAO = 0.70          # manchas grandes onde o carimbo "não pegou" (pressão desigual)
SANGRAMENTO = 0.60            # borrão/dispersão da tinta nas bordas (fibras do papel)
GRANULACAO_PAPEL = 0.80       # variação de densidade da tinta (manchado/granulado)
IRREGULARIDADE = 0.75         # 0 = defeitos espalhados por igual; 1 = concentrados em zonas

COR_TINTA = None              # None = mantém a cor do PNG; ou (R, G, B), ex.: (28, 52, 140)
OPACIDADE_GERAL = 0.92        # opacidade máxima da tinta (0 a 1)
SEMENTE = None                # número inteiro = resultado reprodutível; None = aleatório
# =========================================================================== #


# --------------------------------------------------------------------------- #
# 1) Ruído de Perlin / fractal (fBm)
# --------------------------------------------------------------------------- #
def _fade(t):
    """Curva de suavização de Perlin (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(h, w, escala, rng):
    """Ruído de Perlin 2D vetorizado. `escala` = tamanho (px) de uma célula da grade."""
    escala = max(float(escala), 1.0)
    gh, gw = int(np.ceil(h / escala)) + 2, int(np.ceil(w / escala)) + 2
    ang = rng.uniform(0, 2 * np.pi, (gh, gw)).astype(np.float32)
    gx, gy = np.cos(ang), np.sin(ang)                      # vetores-gradiente da grade

    y = np.arange(h, dtype=np.float32) / escala
    x = np.arange(w, dtype=np.float32) / escala
    yi, xi = y.astype(np.int32), x.astype(np.int32)
    yf, xf = (y - yi)[:, None], (x - xi)[None, :]
    yi, xi = yi[:, None], xi[None, :]

    def canto(dy, dx):                                      # produto escalar de um canto
        return gx[yi + dy, xi + dx] * (xf - dx) + gy[yi + dy, xi + dx] * (yf - dy)

    n00, n10, n01, n11 = canto(0, 0), canto(0, 1), canto(1, 0), canto(1, 1)
    u, v = _fade(xf), _fade(yf)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def fbm(h, w, escala, rng, oitavas=4, ganho=0.5):
    """Ruído fractal (soma de oitavas de Perlin), padronizado: média 0, desvio 1."""
    soma = np.zeros((h, w), np.float32)
    amp, esc = 1.0, float(escala)
    for _ in range(oitavas):
        if esc < 1.0:
            break
        soma += amp * perlin(h, w, esc, rng)
        esc, amp = esc / 2.0, amp * ganho
    return ((soma - soma.mean()) / (soma.std() + 1e-6)).astype(np.float32)


def suave(x, a, b):
    """smoothstep: 0 abaixo de `a`, 1 acima de `b`, transição suave entre os dois."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2) Entrada / saída
# --------------------------------------------------------------------------- #
def carregar_carimbo(caminho):
    """
    Lê o PNG e devolve (rgb, alpha) como float32 em 0..1.
    Se a imagem não tiver transparência (ex.: carimbo preto sobre fundo branco),
    o alfa é derivado do brilho e a cor da tinta é "desmultiplicada" do branco.
    (Pillow é usado para ler/gravar porque lida bem com caminhos com acentos.)
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pil_para_arrays(Image.open(caminho))


def pil_para_arrays(im):
    """Converte uma imagem PIL em (rgb, alpha) float32 0..1 (mesma regra de carregar_carimbo)."""
    tem_alfa = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3]

    if not tem_alfa or alpha.min() > 0.99:
        cinza = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        alpha = np.clip((0.96 - cinza) / 0.5, 0, 1)
        rgb = np.clip((rgb - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 0.05), 0, 1)
    return rgb.astype(np.float32), alpha.astype(np.float32)


def arrays_para_pil(rgb, alpha):
    """(rgb, alpha) float32 0..1 -> imagem PIL RGBA."""
    rgba = np.dstack([rgb, alpha[..., None]])
    return Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8))


def salvar_carimbo(caminho, rgb, alpha):
    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    arrays_para_pil(rgb, alpha).save(caminho)


def aplicar_efeito_em_imagem(img, intensidade=INTENSIDADE_DESGASTE, **kwargs):
    """
    Versão "em memória": recebe uma imagem PIL RGBA (carimbo com fundo transparente)
    e devolve outra imagem PIL RGBA com o desgaste aplicado.
    `kwargs` são os parâmetros de aplicar_efeito_carimbo_realista (falhas, bordas,
    pressao, sangramento, granulacao, irregularidade, cor_tinta, opacidade, semente).
    """
    rgb, alpha = pil_para_arrays(img)
    rgb2, alpha2 = aplicar_efeito_carimbo_realista(rgb, alpha, intensidade=intensidade, **kwargs)
    return arrays_para_pil(rgb2, alpha2)


def sortear_intensidade(minimo=0.1, maximo=0.6, semente=None):
    """Sorteia uma intensidade de desgaste dentro da faixa [minimo, maximo]."""
    return round(float(np.random.default_rng(semente).uniform(minimo, maximo)), 2)


# --------------------------------------------------------------------------- #
# 3) O efeito
# --------------------------------------------------------------------------- #
_NOMES_RNG = ("zona_gran", "gran_md", "gran_hf", "zona_vazios", "vazios_a", "vazios_b",
              "morf", "pressao", "pressao_hf", "zona_borda", "borda", "borda_hf",
              "cortes", "sangra_x", "sangra_y", "sangra_fibra", "extras")


def aplicar_efeito_carimbo_realista(rgb, alpha,
                                    intensidade=INTENSIDADE_DESGASTE,
                                    falhas=QUANT_FALHAS_TINTA,
                                    bordas=DESGASTE_BORDAS,
                                    pressao=FALTA_PRESSAO,
                                    sangramento=SANGRAMENTO,
                                    granulacao=GRANULACAO_PAPEL,
                                    irregularidade=IRREGULARIDADE,
                                    cor_tinta=COR_TINTA,
                                    opacidade=OPACIDADE_GERAL,
                                    semente=SEMENTE):
    """Recebe (rgb, alpha) em float32 0..1 e devolve (rgb, alpha) com o carimbo "gasto"."""
    h, w = alpha.shape
    g = float(np.clip(intensidade, 0, 1))
    F = float(np.clip(falhas, 0, 1)) * g          # intensidades efetivas
    B = float(np.clip(bordas, 0, 1)) * g
    P = float(np.clip(pressao, 0, 1)) * g
    SG = float(np.clip(sangramento, 0, 1)) * g
    G = float(np.clip(granulacao, 0, 1)) * g
    irr = float(np.clip(irregularidade, 0, 1))

    # Um gerador aleatório independente por etapa: mudar um parâmetro não "embaralha" o resto.
    sq = np.random.SeedSequence(semente)
    R = {n: np.random.default_rng(s) for n, s in zip(_NOMES_RNG, sq.spawn(len(_NOMES_RNG)))}

    # ---- Geometria do carimbo --------------------------------------------- #
    mascara0 = (alpha > 0.15).astype(np.uint8)
    ys, xs = np.nonzero(mascara0)
    if len(xs) == 0:
        raise ValueError("O PNG não contém tinta visível (canal alfa vazio).")
    S = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))   # tamanho de referência
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0

    # Todas as escalas são relativas ao tamanho do carimbo -> resultado parecido em qualquer resolução
    esc_hf = max(1.6, 0.006 * S)      # grão fino (fibra do papel)
    esc_md = max(4.0, 0.035 * S)      # manchas médias

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Silhueta externa (casco convexo) e distância até a borda: base do desgaste nas extremidades
    contornos, _ = cv2.findContours(mascara0 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pontos = np.vstack([c.reshape(-1, 2) for c in contornos])
    silhueta = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(silhueta, cv2.convexHull(pontos), 255)
    d_ext = cv2.distanceTransform(silhueta, cv2.DIST_L2, 5)
    peso_borda = (np.exp(-d_ext / (0.07 * S)) * (silhueta > 0)).astype(np.float32)

    def zona(escala, rng):
        """
        Mapa 0..1 de "força local". Com irregularidade=0 vira constante (defeito uniforme);
        com irregularidade=1 há regiões quase limpas e regiões muito degradadas.
        Retorna um multiplicador ~0..2 (média ~1).
        """
        z = suave(fbm(h, w, escala, rng, 2), -0.9, 1.0)
        return 2.0 * ((1 - irr) * 0.5 + irr * z)

    a = alpha.copy()
    if g <= 0:                                   # carimbo "novo": só cor/opacidade
        a_out = np.clip(a * opacidade, 0, 1)
        return _cor_final(rgb, cor_tinta), a_out

    # ---- (4) SANGRAMENTO: bordas irregulares + dispersão pelas fibras ------- #
    if SG > 0:
        amp = SG * max(0.8, 0.004 * S)                            # deslocamento máx. em px
        dx = fbm(h, w, esc_hf * 1.5, R["sangra_x"], 3) * amp * 0.5
        dy = fbm(h, w, esc_hf * 1.5, R["sangra_y"], 3) * amp * 0.5
        a = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)   # contorno "ondulado"
        sigma = 0.4 + SG * max(0.8, 0.006 * S)
        a_bl = cv2.GaussianBlur(a, (0, 0), sigma)
        fibra = fbm(h, w, esc_hf * 0.8, R["sangra_fibra"], 3)
        nucleo = suave(a_bl + 0.14 * SG * fibra, 0.12, 0.55)      # borda "rasgada" por fibra
        halo = cv2.GaussianBlur(a, (0, 0), sigma * 2.6) * (0.45 * SG) * suave(fibra, -1.0, 1.5)
        a = np.maximum(nucleo, halo)                              # tinta que escorre p/ fora

    # ---- (2) FALHAS DE TINTA ------------------------------------------------ #
    if F > 0:
        fz = zona(0.28 * S, R["zona_vazios"])
        # 2a) ruído de contato: onde o ruído fractal passa de um limiar, a tinta não pegou
        ruido_v = 0.65 * fbm(h, w, esc_hf * 0.9, R["vazios_a"], 3) \
                + 0.55 * fbm(h, w, esc_md * 0.8, R["vazios_b"], 4)
        limiar = 2.5 - 1.7 * F - 0.9 * (fz - 1.0)                 # zonas fortes: limiar menor
        vazio_ruido = suave(ruido_v, limiar, limiar + 0.4) * suave(F, 0.0, 0.05)

        # 2b) vazios morfológicos: sementes esparsas + dilatação com elementos estruturantes
        #     elípticos de tamanho/forma aleatórios (pontos, manchas e falhas grandes)
        vazio_morf = np.zeros((h, w), np.uint8)
        rng_m = R["morf"]
        for n_sem, (fa, fb) in ((140, (0.004, 0.012)), (30, (0.012, 0.03)), (5, (0.03, 0.07))):
            dens = np.clip(n_sem * F / (S * S) * fz, 0, 1)        # sementes concentradas em zonas
            sementes = rng_m.random((h, w), dtype=np.float32) < dens
            if not sementes.any():
                continue
            grupo = rng_m.integers(0, 3, (h, w), dtype=np.uint8)
            for gi in range(3):                                   # 3 elementos estruturantes/nível
                sub = (sementes & (grupo == gi)).astype(np.uint8)
                if not sub.any():
                    continue
                kx = max(1, int(S * rng_m.uniform(fa, fb)))
                ky = max(1, int(kx * rng_m.uniform(0.5, 1.6)))
                if rng_m.random() < 0.5:
                    kx, ky = ky, kx
                vazio_morf |= cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky)))
        v = cv2.GaussianBlur(vazio_morf.astype(np.float32), (0, 0), 0.004 * S + 0.6)
        v = suave(v + 0.18 * fbm(h, w, esc_hf, R["vazios_a"], 3), 0.45, 0.75)   # borda irregular

        # 2c) erosão: afina os traços (cv2.erode) apenas nas zonas mais desgastadas
        k_er = max(3, int(0.008 * S)) | 1
        a_fino = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_er, k_er)))
        t = np.clip((fz - 0.8) / 1.2, 0, 1) * min(1.0, F * 1.3)
        a = a * (1 - t) + a_fino * t

        a = a * (1 - vazio_ruido) * (1 - v)

    # ---- (1) FALTA DE PRESSÃO: lado/região onde o carimbo "não pegou" ------- #
    if P > 0:
        ang = R["pressao"].uniform(0, 2 * np.pi)
        inclinacao = R["pressao"].uniform(0.3, 1.0)               # carimbo apoiado mais de um lado
        proj = ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)) / (0.6 * S)
        mapa_p = 0.5 - 0.5 * inclinacao * proj + 0.30 * fbm(h, w, 0.35 * S, R["pressao"], 3)
        limiar_p = -0.9 + 1.7 * P
        # o ruído fino faz a tinta "se dissolver" em pontinhos na transição, não em linha reta
        cobertura = suave(mapa_p + 0.20 * fbm(h, w, esc_hf, R["pressao_hf"], 3),
                          limiar_p, limiar_p + 0.30)
        a = a * cobertura

    # ---- (3) DESGASTE NAS BORDAS: pedaços faltando e fraturas -------------- #
    if B > 0:
        fb = zona(0.30 * S, R["zona_borda"])
        z_b = fbm(h, w, max(esc_md * 1.2, 0.05 * S), R["borda"], 4)
        valor = z_b + 1.5 * peso_borda + 0.30 * fbm(h, w, esc_hf, R["borda_hf"], 3)
        limiar_b = 3.0 - 1.6 * B - 0.9 * (fb - 1.0)
        perda = suave(valor, limiar_b, limiar_b + 0.35) * suave(B, 0.0, 0.08)
        a = a * (1 - perda)

        # fraturas/cortes finos na borracha: riscas sem tinta que entram a partir das bordas
        cortes = np.zeros((h, w), np.uint8)
        cand = np.argwhere((peso_borda > 0.5) & (a > 0.3))
        n_cortes = int(round(B * 16))
        if len(cand) and n_cortes:
            rng_c = R["cortes"]
            idx = rng_c.permutation(len(cand))[: n_cortes * 12]
            escolhidos = [tuple(cand[i]) for i in idx if rng_c.random() < fb[cand[i][0], cand[i][1]] * 0.5]
            for (py, px) in escolhidos[:n_cortes]:
                direcao = np.arctan2(cy - py, cx - px) + rng_c.normal(0, 0.5)   # para dentro
                pts, x_, y_ = [(px, py)], float(px), float(py)
                for _ in range(int(rng_c.integers(5, 12))):
                    direcao += rng_c.normal(0, 0.3)
                    x_ += 0.012 * S * np.cos(direcao)
                    y_ += 0.012 * S * np.sin(direcao)
                    pts.append((x_, y_))
                cv2.polylines(cortes, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)],
                              False, 255, max(1, int(0.004 * S)), cv2.LINE_AA)
            a = a * (1 - 0.95 * cv2.GaussianBlur(cortes.astype(np.float32) / 255, (0, 0), 0.6))

    # ---- (4b) DENSIDADE / GRANULAÇÃO DA TINTA ------------------------------ #
    if G > 0:
        fz = zona(0.35 * S, R["zona_gran"])
        n_md = fbm(h, w, esc_md, R["gran_md"], 4)
        n_hf = fbm(h, w, esc_hf, R["gran_hf"], 3)
        queda = G * fz * (0.20 * suave(n_md, -1.2, 1.8) + 0.25 * suave(n_hf, -1.0, 2.0))
        # miolo de áreas grandes costuma ficar mais claro que as bordas do traço
        d_in = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        queda = queda + 0.15 * G * suave(d_in, 0.008 * S, 0.02 * S)
        a = a * np.clip(1.0 - queda, 0.2, 1.0)

    # ---- Respingos / pontos soltos de tinta (poucos, em zonas desgastadas) --- #
    if g > 0.25:
        rng_e = R["extras"]
        n_pts = int(rng_e.poisson(12 * g * g))
        cnv = np.zeros((h, w), np.uint8)
        for _ in range(n_pts):
            px = int(np.clip(cx + rng_e.normal(0, 0.42 * S), 0, w - 1))
            py = int(np.clip(cy + rng_e.normal(0, 0.42 * S), 0, h - 1))
            cv2.circle(cnv, (px, py), max(1, int(S * rng_e.uniform(0.002, 0.007))),
                       int(rng_e.uniform(120, 255)), -1, cv2.LINE_AA)
        a = np.maximum(a, cv2.GaussianBlur(cnv.astype(np.float32) / 255, (0, 0), 0.6))

    a = cv2.GaussianBlur(a, (0, 0), 0.5)                      # anti-serrilhado final
    a_out = np.clip(a * opacidade, 0, 1).astype(np.float32)
    return _cor_final(rgb, cor_tinta), a_out


def _cor_final(rgb, cor_tinta):
    if cor_tinta is None:
        return rgb
    cor = np.array(cor_tinta, np.float32).reshape(1, 1, 3) / 255.0
    return np.broadcast_to(cor, rgb.shape).copy()


def compor_previa(rgb, alpha, fundo=(243, 241, 234)):
    """Compõe o carimbo sobre um 'papel' claro (mistura multiply) só para visualização."""
    bg = np.empty(alpha.shape + (3,), np.float32)
    bg[:] = np.array(fundo, np.float32) / 255.0
    out = bg * (1 - alpha[..., None]) + (bg * rgb) * alpha[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# 4) Linha de comando
# --------------------------------------------------------------------------- #
def _parse_cor(txt):
    if txt is None:
        return COR_TINTA
    txt = txt.strip()
    if txt.startswith("#") and len(txt) == 7:
        return tuple(int(txt[i:i + 2], 16) for i in (1, 3, 5))
    partes = [int(p) for p in txt.split(",")]
    if len(partes) != 3:
        raise ValueError("Use --cor '#RRGGBB' ou --cor 'R,G,B'")
    return tuple(partes)


def main():
    ap = argparse.ArgumentParser(description="Aplica desgaste realista em um PNG de carimbo.")
    ap.add_argument("entrada", nargs="?", default="Carimbos/c_retangular.png")
    ap.add_argument("-o", "--saida", default=None, help="padrão: <entrada>_imperfeito.png")
    ap.add_argument("--intensidade", type=float, default=INTENSIDADE_DESGASTE)
    ap.add_argument("--falhas", type=float, default=QUANT_FALHAS_TINTA)
    ap.add_argument("--bordas", type=float, default=DESGASTE_BORDAS)
    ap.add_argument("--pressao", type=float, default=FALTA_PRESSAO)
    ap.add_argument("--sangramento", type=float, default=SANGRAMENTO)
    ap.add_argument("--granulacao", type=float, default=GRANULACAO_PAPEL)
    ap.add_argument("--irregularidade", type=float, default=IRREGULARIDADE)
    ap.add_argument("--cor", default=None, help="'#RRGGBB' ou 'R,G,B' (padrão: cor original)")
    ap.add_argument("--opacidade", type=float, default=OPACIDADE_GERAL)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    ap.add_argument("--variantes", type=int, default=1, help="gera N variações (sementes diferentes)")
    ap.add_argument("--previa", action="store_true", help="salva também uma prévia sobre fundo de papel")
    args = ap.parse_args()

    try:
        rgb, alpha = carregar_carimbo(args.entrada)
        base, _ = os.path.splitext(args.saida or args.entrada)
        base = base if args.saida else base + "_imperfeito"
        for i in range(args.variantes):
            semente = None if args.semente is None else args.semente + i
            rgb2, alpha2 = aplicar_efeito_carimbo_realista(
                rgb, alpha, intensidade=args.intensidade, falhas=args.falhas, bordas=args.bordas,
                pressao=args.pressao, sangramento=args.sangramento, granulacao=args.granulacao,
                irregularidade=args.irregularidade, cor_tinta=_parse_cor(args.cor),
                opacidade=args.opacidade, semente=semente)
            destino = f"{base}_{i + 1:02d}.png" if args.variantes > 1 else f"{base}.png"
            salvar_carimbo(destino, rgb2, alpha2)
            print("Salvo:", destino)
            if args.previa:
                compor_previa(rgb2, alpha2).save(destino.replace(".png", "_previa.png"))
    except Exception as e:  # noqa: BLE001
        print(f"Erro: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()