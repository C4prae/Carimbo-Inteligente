#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integracao_documento.py
=======================
Duas tarefas que fazem o carimbo parecer parte da digitalização:

1) ONDE carimbar (posicionamento automático)
   * mascara_texto(): detecta o texto impresso (letras, caixas dos campos, brasão...)
     e IGNORA o fundo decorativo (guilhochê colorido das bordas, marca d'água), que
     pode ser coberto normalmente.
   * procurar_posicao(): varre o documento inteiro e sorteia (com semente) um local onde
     haja espaço. Nenhuma palavra pode ser coberta por inteiro; uma sobreposição mínima
     só é aceita quando não existe lugar melhor (tolerância crescente).

2) COMO carimbar (composição)
   * compor_na_digitalizacao(): em vez de colar a tinta "chapada", dessatura e suaviza a
     cor, faz a tinta reagir à textura do papel, aplica o mesmo desfoque/grão da
     digitalização e mistura por multiplicação (a tinta escurece o papel, não o substitui).

Dependências: numpy, opencv-python, pillow
"""
import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# 1) Mapa de texto impresso
# --------------------------------------------------------------------------- #
def _impar(n):
    n = int(n)
    return n if n % 2 else n + 1


def _moldura(hsv, h, w, limite_frac=0.20, dens_min=0.15):
    """bool (H, W): True na moldura decorativa que contorna a página (cor saturada densa)."""
    sat = (hsv[..., 1] > 90) & (hsv[..., 2] > 110)
    col = sat[int(h * .25):int(h * .75)].mean(0)           # densidade por coluna (faixa central)
    lin = sat[:, int(w * .25):int(w * .75)].mean(1)        # densidade por linha

    def largura(v):
        v = np.convolve(v.astype(np.float32), np.ones(9, np.float32) / 9, mode="same")
        idx = np.nonzero(v[:int(len(v) * limite_frac)] > dens_min)[0]
        return int(idx.max()) + 1 if idx.size else 0

    esq, dir_, topo, base = largura(col), largura(col[::-1]), largura(lin), largura(lin[::-1])
    deco = np.zeros((h, w), bool)
    if esq:
        deco[:, :esq] = True
    if dir_:
        deco[:, w - dir_:] = True
    if topo:
        deco[:topo, :] = True
    if base:
        deco[h - base:, :] = True
    return deco


def mascara_texto(doc_rgb, limiar_escuro=0.30, limiar_contraste=0.28,
                  halo_px=None, return_debug=False):
    """
    doc_rgb : array uint8 (H, W, 3).
    Retorna um dict com:
      texto    : bool (H, W)  pixels de texto/linhas/gráficos impressos
      protegido: bool (H, W)  texto dilatado (margem de segurança)
      palavras : int32 (H, W) rótulo de cada "palavra/linha" (0 = fundo)
      area_pal : array com a área de texto de cada rótulo
      deco     : bool (H, W)  região decorativa ignorada (bordas guilhochê)
    """
    h, w, _ = doc_rgb.shape
    rgb = doc_rgb.astype(np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    hsv = cv2.cvtColor(doc_rgb, cv2.COLOR_RGB2HSV)

    # (a) pixels muito escuros  (b) traços escuros finos em relação ao fundo local (black-hat)
    k = _impar(max(15, w // 70))
    fundo = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bruto = (lum < limiar_escuro) | ((fundo - lum) > limiar_contraste)

    # Região decorativa = a moldura de guilhochê: faixa nas bordas com muita cor saturada.
    # Os limites internos são detectados varrendo de cada borda para dentro; o interior
    # (texto, brasão, holograma, rosetas) continua protegido. Sem moldura -> deco vazio.
    deco = _moldura(hsv, h, w)

    texto = bruto & ~deco
    # remove sujeira de 1-2 px isolados
    n, lab, stats, _ = cv2.connectedComponentsWithStats(texto.astype(np.uint8), connectivity=8)
    pequenos = np.isin(lab, np.nonzero(stats[:, cv2.CC_STAT_AREA] < 3)[0]) & (lab > 0)
    texto &= ~pequenos

    halo = halo_px if halo_px is not None else max(2, w // 550)
    protegido = cv2.dilate(texto.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo + 1, 2 * halo + 1))) > 0

    # "Palavras": junta letras vizinhas (fechamento horizontal) e rotula cada bloco
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 110), max(3, w // 500)))
    blocos = cv2.morphologyEx(texto.astype(np.uint8), cv2.MORPH_CLOSE, kh)
    _, palavras = cv2.connectedComponents(blocos, connectivity=8)
    palavras = palavras.astype(np.int32)
    palavras[~texto] = 0
    area_pal = np.bincount(palavras.ravel(), minlength=palavras.max() + 1).astype(np.float64)
    out = {"texto": texto, "protegido": protegido, "palavras": palavras,
           "area_pal": area_pal, "deco": deco}
    return out


# --------------------------------------------------------------------------- #
# 2) Procura de posição
# --------------------------------------------------------------------------- #
def _silhueta(alpha):
    """Silhueta preenchida do carimbo (casco convexo da tinta): o 'espaço' que ele ocupa."""
    tinta = (alpha > 0.15).astype(np.uint8)
    cnts, _ = cv2.findContours(tinta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(tinta)
    if cnts:
        cv2.fillConvexPoly(sil, cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts])), 1)
    return sil.astype(np.float32), tinta.astype(np.float32)


def procurar_posicao(mapa, carimbo_rgba, rng, ocupado=None, margem_px=0,
                     tolerancias=(0.0, 0.0015, 0.004, 0.008), max_palavra=(0.0, 0.10, 0.25, 0.35),
                     nivel_inicial=0, min_candidatos=40, passo=6, eps=2e-4):
    """
    mapa          : resultado de mascara_texto()
    carimbo_rgba  : imagem PIL RGBA JÁ no tamanho final
    rng           : numpy Generator (sorteio reprodutível)
    ocupado       : bool (H, W) de áreas já carimbadas (opcional)
    tolerancias   : fração máxima da silhueta do carimbo que pode cair sobre texto, por nível.
                    Nível 0 = nenhuma sobreposição; os seguintes permitem um roçar mínimo.
    max_palavra   : fração máxima de QUALQUER palavra/linha que pode ser coberta, por nível
                    (garante que nenhum texto seja coberto por inteiro).
    nivel_inicial : primeiro nível tentado. Se não houver lugar nele, tenta os níveis acima e,
                    por último, os abaixo.
    eps           : folga numérica (a correlação por FFT deixa resíduos ~1e-8).

    Retorna dict(x, y, nivel, cobertura_texto, max_palavra_coberta) ou None se não couber.
    """
    H, W = mapa["texto"].shape
    alpha = np.asarray(carimbo_rgba, dtype=np.float32)[..., 3] / 255.0
    ch, cw = alpha.shape
    if ch >= H or cw >= W:
        return None
    sil, tinta = _silhueta(alpha)
    area_sil = max(float(sil.sum()), 1.0)

    prot = mapa["protegido"].astype(np.float32)
    cob = cv2.matchTemplate(prot, sil, cv2.TM_CCORR) / area_sil          # texto sob a silhueta
    cob_tinta = cv2.matchTemplate(prot, tinta, cv2.TM_CCORR) / max(float(tinta.sum()), 1.0)
    if ocupado is not None and ocupado.any():
        outro = cv2.matchTemplate(ocupado.astype(np.float32), sil, cv2.TM_CCORR) / area_sil
    else:
        outro = np.zeros_like(cob)

    ys, xs = np.mgrid[0:cob.shape[0], 0:cob.shape[1]]
    dentro = ((xs >= margem_px) & (ys >= margem_px) &
              (xs <= W - cw - margem_px) & (ys <= H - ch - margem_px))
    grade = ((xs % passo) == 0) & ((ys % passo) == 0)
    base = dentro & grade & (outro <= eps)            # nunca sobre outro carimbo

    palavras, area_pal, texto = mapa["palavras"], mapa["area_pal"], mapa["texto"]
    n = len(tolerancias)
    ni = int(np.clip(nivel_inicial, 0, n - 1))
    ordem = list(range(ni, n)) + list(range(0, ni))
    for pos_ordem, nivel in enumerate(ordem):
        tol, mp = tolerancias[nivel], max_palavra[nivel]
        valido = base & (cob <= tol + eps) & (cob_tinta <= tol * 2 + eps)
        cand = np.argwhere(valido)
        if len(cand) == 0 or (len(cand) < min_candidatos and pos_ordem < n - 1):
            continue                                         # pouco espaço neste nível: tenta outro
        for i in rng.permutation(len(cand))[:400]:           # sorteia e confere palavra a palavra
            y, x = cand[i]
            sub = palavras[y:y + ch, x:x + cw]
            cobertos = sub[(sil > 0) & texto[y:y + ch, x:x + cw]]
            if cobertos.size:
                cont = np.bincount(cobertos, minlength=len(area_pal)).astype(np.float64)
                fr = cont[1:] / np.maximum(area_pal[1:], 1.0)
                pior = float(fr.max()) if fr.size else 0.0
            else:
                pior = 0.0
            if pior <= mp:
                return {"x": int(x), "y": int(y), "nivel": int(nivel),
                        "cobertura_texto": round(max(float(cob[y, x]), 0.0), 5),
                        "max_palavra_coberta": round(pior, 3)}
    return None


def marcar_ocupado(ocupado, carimbo_rgba, x, y, folga_px=28):
    """Registra a silhueta do carimbo já posicionado, para o próximo não cair em cima."""
    alpha = np.asarray(carimbo_rgba, dtype=np.float32)[..., 3] / 255.0
    sil, _ = _silhueta(alpha)
    sil = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * folga_px + 1,) * 2))
    h, w = sil.shape
    ocupado[y:y + h, x:x + w] |= sil > 0
    return ocupado


# --------------------------------------------------------------------------- #
# 3) Composição "como se fizesse parte da digitalização"
# --------------------------------------------------------------------------- #
def compor_na_digitalizacao(doc_rgb, carimbo_rgba, x0, y0, saturacao=0.62, clareza=0.18,
                            opacidade=0.80, papel=0.55, desfoque=None, grao=0.012, semente=None):
    """
    doc_rgb      : array uint8 (H, W, 3) — a página digitalizada
    carimbo_rgba : imagem PIL RGBA já no tamanho final
    saturacao    : 1 = cor original da tinta; menor = mais acinzentada (tinta "esmaecida")
    clareza      : clareia a tinta em direção ao cinza médio (tinta nunca fica mais escura que o texto)
    opacidade    : opacidade máxima da tinta sobre o papel
    papel        : quanto a textura do papel modula a tinta (0 = nada)
    desfoque     : sigma (px) para igualar a nitidez da digitalização; None = estimado pela resolução
    grao         : desvio do grão adicionado à tinta (imita o ruído do scanner)
    """
    rng = np.random.default_rng(semente)
    H, W, _ = doc_rgb.shape
    st = np.asarray(carimbo_rgba, dtype=np.float32) / 255.0
    ch, cw = st.shape[:2]
    x1, y1 = min(W, x0 + cw), min(H, y0 + ch)
    x0c, y0c = max(0, x0), max(0, y0)
    if x1 <= x0c or y1 <= y0c:
        return doc_rgb
    st = st[y0c - y0:y1 - y0, x0c - x0:x1 - x0]
    reg = doc_rgb[y0c:y1, x0c:x1].astype(np.float32) / 255.0

    a = st[..., 3]
    # cor da tinta: mediana nos pixels opacos -> dessaturada e clareada
    sel = a > 0.5
    cor = np.median(st[..., :3][sel], axis=0) if sel.any() else np.array([0.2, 0.2, 0.4], np.float32)
    cinza = float(cor @ np.array([0.299, 0.587, 0.114], np.float32))
    cor = cinza + (cor - cinza) * saturacao
    cor = cor * (1 - clareza) + 0.62 * clareza

    # a tinta se aloja nas "valetas" da fibra do papel
    lum = reg @ np.array([0.299, 0.587, 0.114], np.float32)
    alta = lum - cv2.GaussianBlur(lum, (0, 0), 2.5)
    alta = alta / (alta.std() + 1e-4)
    a = a * np.clip(1.0 - papel * 0.22 * alta, 0.55, 1.35)

    # mesma nitidez da digitalização
    sig = desfoque if desfoque is not None else 0.5 + W / 4000.0
    a = cv2.GaussianBlur(a, (0, 0), sig)
    a = np.clip(a * opacidade, 0.0, 1.0)

    # mistura por multiplicação (a tinta escurece o papel; o texto de baixo continua visível)
    tinta = 1.0 - a[..., None] + a[..., None] * cor.reshape(1, 1, 3)
    out = reg * tinta
    out += rng.normal(0.0, grao, a.shape).astype(np.float32)[..., None] * a[..., None] * 1.5   # grão só na tinta

    res = doc_rgb.copy()
    res[y0c:y1, x0c:x1] = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    return res