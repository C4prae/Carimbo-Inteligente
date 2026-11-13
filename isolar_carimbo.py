#!/usr/bin/env python3
"""
isolar_carimbo.py — Etapas 1, 3, 4 e 5 do pipeline de carimbos.

Recebe o recorte de um carimbo (saída do detector) ou uma página inteira
(imagem ou PDF) e devolve PNGs contendo apenas o traço do carimbo.

Ideia central: separação de tintas por densidade óptica (lei de Beer-Lambert).
Cada pixel é modelado como   OD = c_carimbo * v_carimbo + Σ c_i * v_interferente_i
onde OD = -log(I / I_papel). Resolver esse sistema por pixel dá a "quantidade
de tinta do carimbo" (c_carimbo) mesmo onde ele cruza a marca d'água — é o
equivalente clássico da classe OV (sobreposição) do artigo de referência.

Uso:
    python isolar_carimbo.py recorte.png -o saida/
    python isolar_carimbo.py pagina.pdf --pagina 1 --bbox 120,340,400,610 -o saida/
    python isolar_carimbo.py recorte.png --mascara-modelo prob.png   # prior do modelo MFM

Dependências: opencv-python, numpy, scikit-image (pdf2image só para PDF).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu


# --------------------------------------------------------------------------- #
# Parâmetros
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    escala: float = 4.0            # super-resolução do recorte (Lanczos)
    lado_max: int = 2400           # limite do maior lado após a escala
    frac_fundo: float = 0.12       # janela do estimador de fundo (fração do menor lado)
    od_min: float = 0.25           # OD mínima para um pixel entrar na estimativa de tintas
    k_tintas: int = 3              # nº de tintas procuradas (carimbo + interferentes)
    cos_fusao: float = 0.985       # tintas mais parecidas que isso são fundidas
    area_min_rel: float = 2.5e-5   # componente menor que isso (fração da imagem) é ruído
    traco: float = 0.5             # espessura do traço: 0 = fino/aberto … 1 = cheio
    margem: int = 24               # margem em px ao redor do carimbo no PNG final
    cor_saida: str = "tinta"       # "tinta" (cor estimada) | "preto" | "#RRGGBB"


# --------------------------------------------------------------------------- #
# Etapa 1 — entrada e normalização
# --------------------------------------------------------------------------- #
def carregar(caminho: str, pagina: int = 1, dpi: int = 400) -> np.ndarray:
    """Lê imagem ou PDF e devolve BGR uint8."""
    p = Path(caminho)
    if p.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pil = convert_from_path(str(p), dpi=dpi, first_page=pagina, last_page=pagina)[0]
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(caminho)
    return img


def normalizar(img: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Amplia, suaviza ruído preservando bordas e estima o papel (fundo)."""
    h, w = img.shape[:2]
    s = min(cfg.escala, cfg.lado_max / max(h, w))
    if s > 1.01:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    # bilateral: remove granulado de JPEG/scanner sem borrar o traço
    img = cv2.bilateralFilter(img, d=5, sigmaColor=18, sigmaSpace=3)

    # Fundo = fechamento morfológico (apaga traços escuros) + mediana (suaviza).
    # A janela precisa ser maior que a espessura do traço mais grosso.
    k = int(min(img.shape[:2]) * cfg.frac_fundo) | 1
    k = max(k, 15)
    pequeno = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    kk = max(k // 4, 5) | 1
    fundo = cv2.morphologyEx(pequeno, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    fundo = cv2.medianBlur(fundo, kk)
    fundo = cv2.resize(fundo, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img, fundo


def densidade_optica(img: np.ndarray, fundo: np.ndarray) -> np.ndarray:
    """OD por canal: 0 = papel limpo; cresce com a quantidade de tinta.
    Dividir pelo fundo cancela cor do papel, iluminação e fundo de segurança claro."""
    i = img.astype(np.float32) + 1.0
    f = fundo.astype(np.float32) + 1.0
    return np.clip(-np.log(i / f), 0.0, None)


# --------------------------------------------------------------------------- #
# Etapa 3 — estimativa das tintas e separação (unmixing)
# --------------------------------------------------------------------------- #
def estimar_tintas(od: np.ndarray, cfg: Config,
                   prior: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Agrupa as direções de OD (k-means) → um vetor unitário por tinta.
    Devolve (vetores [n,3] em BGR, índice da tinta do carimbo).

    A tinta do carimbo é a de maior "massa" de OD ponderada pelo prior
    (máscara do modelo, se houver; senão uma gaussiana centrada no recorte,
    já que o detector centraliza o carimbo)."""
    h, w = od.shape[:2]
    mag = np.linalg.norm(od, axis=2)
    sel = mag > cfg.od_min
    if sel.sum() < 200:
        raise RuntimeError("Pouca tinta encontrada — verifique o recorte/--bbox.")
    dirs = (od[sel] / mag[sel][:, None]).astype(np.float32)

    k = min(cfg.k_tintas, len(dirs))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-5)
    _, rot, centros = cv2.kmeans(dirs, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    centros /= np.linalg.norm(centros, axis=1, keepdims=True)
    rot = rot.ravel()

    if prior is None:
        yy, xx = np.mgrid[0:h, 0:w]
        prior = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - h / 2) / (0.45 * h)) ** 2))
    peso = (mag * prior)[sel]
    massa = np.array([peso[rot == i].sum() for i in range(k)])
    i_car = int(np.argmax(massa))

    # Funde clusters quase colineares com o carimbo (são a mesma tinta, mais
    # clara/escura) — mantê-los separados roubaria traço do carimbo.
    manter = [i_car]
    for i in np.argsort(-massa):
        if i == i_car:
            continue
        if all(float(centros[i] @ centros[j]) < cfg.cos_fusao for j in manter):
            manter.append(int(i))
    return centros[manter], 0


def separar_tintas(od: np.ndarray, vetores: np.ndarray) -> np.ndarray:
    """Mínimos quadrados não-negativos por pixel (conjunto ativo simplificado).
    Devolve concentrações [h,w,n]; o canal 0 é o carimbo."""
    h, w = od.shape[:2]
    y = od.reshape(-1, 3).astype(np.float32)
    v = vetores.astype(np.float32)                       # [n,3]
    n = len(v)
    c = y @ np.linalg.pinv(v)                            # solução irrestrita
    # onde alguma concentração ficou negativa, refaz sem aquela tinta
    for _ in range(n - 1):
        neg = (c < 0).any(axis=1)
        if not neg.any():
            break
        ativo = c[neg] > 0
        c_novo = np.zeros_like(c[neg])
        for combo in np.unique(ativo, axis=0):
            if not combo.any():
                continue
            linhas = (ativo == combo).all(axis=1)
            sub = v[combo]
            c_novo[np.ix_(linhas, combo)] = y[neg][linhas] @ np.linalg.pinv(sub)
        c[neg] = c_novo
    return np.clip(c, 0, None).reshape(h, w, n)


# --------------------------------------------------------------------------- #
# Etapa 4 — binarização do traço
# --------------------------------------------------------------------------- #
def binarizar(c_car: np.ndarray, c_outros: np.ndarray, cfg: Config,
              escala: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Limiar LOCAL relativo ao pico de tinta da vizinhança.

    Um limiar global baixo "enche" o traço: o borrão do scanner em volta de cada
    letra passa do limiar e fecha os miolos (A, O, R, 8...). Aqui o pixel só é
    traço se tiver pelo menos uma fração do pico local — isto é, corta-se cada
    traço na sua meia-altura, seja a tinta forte ou fraca naquele ponto.

    Devolve (máscara, mapa de limiar, limiar global mínimo)."""
    e = float(np.clip(cfg.traco, 0.0, 1.0))
    frac = 0.62 - 0.30 * e                      # fração do pico local exigida
    nitidez = 0.8 * (1.0 - e)                   # realce de borda (unsharp) p/ traço fino

    c = c_car
    if nitidez > 0:
        c = np.clip(c + nitidez * (c - cv2.GaussianBlur(c, (0, 0), 1.2 * escala)), 0, None)

    k = int(5 * escala) | 1                     # vizinhança ≈ 5 px da imagem original
    pico = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pico = cv2.GaussianBlur(pico, (0, 0), escala)

    val = c_car[c_car > 0.05]
    otsu = float(np.clip(threshold_otsu(val), 0.15, 1.5)) if val.size else 0.3
    piso = 0.5 * otsu                           # abaixo disso é papel, não tinta
    limiar = np.maximum(frac * pico, piso)

    masc = c >= limiar
    # pixel dominado por outra tinta não é carimbo (ex.: núcleo da marca d'água)
    masc &= c_car > 0.8 * c_outros
    return masc, (c, limiar), otsu


# --------------------------------------------------------------------------- #
# Etapa 3b — filtragem morfológica e geométrica
# --------------------------------------------------------------------------- #
def limpar(masc: np.ndarray, cfg: Config,
           prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remove ruído e restringe ao corpo do carimbo. Devolve (máscara, ROI)."""
    h, w = masc.shape
    m = masc.astype(np.uint8)

    # 1) componentes minúsculos = poeira, pontilhado de guilhoché, restos de marca d'água
    area_min = max(int(cfg.area_min_rel * h * w), 6)
    n, rot, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ok = np.zeros(n, bool)
    ok[1:] = st[1:, cv2.CC_STAT_AREA] >= area_min
    m = ok[rot].astype(np.uint8)

    # 2) ROI do carimbo: fecha os vãos entre letras/anéis e fica com o maior
    #    aglomerado (ou o que mais coincide com o prior do modelo).
    r = max(int(0.03 * min(h, w)), 5) | 1
    blob = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)), iterations=2)
    n, rot, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n <= 1:
        return m.astype(bool), np.zeros_like(m, bool)
    if prior is not None:
        nota = np.array([prior[rot == i].sum() for i in range(1, n)])
    else:
        nota = st[1:, cv2.CC_STAT_AREA].astype(float)
    alvo = 1 + int(np.argmax(nota))
    cont, _ = cv2.findContours((rot == alvo).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(m)
    cv2.drawContours(roi, [cv2.convexHull(np.vstack(cont))], -1, 1, cv2.FILLED)
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m &= roi

    return m.astype(bool), roi.astype(bool)


# --------------------------------------------------------------------------- #
# Etapa 5 — exportação
# --------------------------------------------------------------------------- #
def _cor_bgr(cfg: Config, v_car: np.ndarray, forte: float) -> tuple[int, int, int]:
    if cfg.cor_saida == "preto":
        return (0, 0, 0)
    if cfg.cor_saida.startswith("#"):
        hx = cfg.cor_saida.lstrip("#")
        r, g, b = (int(hx[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)
    # cor da tinta sobre papel branco, a partir do vetor de absorção estimado
    bgr = 255.0 * np.exp(-v_car * max(forte * 1.6, 0.9))
    return tuple(int(x) for x in np.clip(bgr, 0, 255))


def exportar(c: np.ndarray, limiar: np.ndarray, masc: np.ndarray, forte: float,
             v_car: np.ndarray, cfg: Config, pasta: Path, nome: str) -> dict[str, Path]:
    """Gera: <nome>_rgba.png (traço na cor da tinta, fundo transparente),
             <nome>_bin.png  (preto sobre branco, para OCR/arquivo)."""
    pasta.mkdir(parents=True, exist_ok=True)

    # alfa suave: rampa estreita em torno do limiar local → borda com
    # anti-aliasing sem engrossar o traço; a porta impede a volta do ruído.
    rampa = np.clip((c - limiar) / (0.25 * limiar) + 0.5, 0, 1)
    porta = cv2.dilate(masc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    porta = cv2.GaussianBlur(porta, (0, 0), 0.8)
    alfa = np.clip(rampa * porta, 0, 1)

    ys, xs = np.nonzero(masc)
    if ys.size == 0:
        raise RuntimeError("Nenhum traço de carimbo após a filtragem.")
    y0, y1 = max(ys.min() - cfg.margem, 0), min(ys.max() + cfg.margem + 1, masc.shape[0])
    x0, x1 = max(xs.min() - cfg.margem, 0), min(xs.max() + cfg.margem + 1, masc.shape[1])
    alfa, masc = alfa[y0:y1, x0:x1], masc[y0:y1, x0:x1]

    b, g, r = _cor_bgr(cfg, v_car, forte)
    rgba = np.zeros((*alfa.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = b, g, r
    rgba[..., 3] = np.round(alfa * 255).astype(np.uint8)

    binario = np.where(alfa >= 0.5, 0, 255).astype(np.uint8)

    saidas = {"rgba": pasta / f"{nome}_rgba.png", "bin": pasta / f"{nome}_bin.png"}
    cv2.imwrite(str(saidas["rgba"]), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(saidas["bin"]), binario, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return saidas


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def isolar_carimbo(img_bgr: np.ndarray, pasta: Path, nome: str = "carimbo",
                   cfg: Config | None = None,
                   prob_modelo: np.ndarray | None = None,
                   depurar: bool = False) -> dict[str, Path]:
    """img_bgr: recorte do carimbo. prob_modelo: mapa [0..1] da classe 'carimbo'
    vindo do modelo de segmentação (opcional), no tamanho do recorte."""
    cfg = cfg or Config()
    img, fundo = normalizar(img_bgr, cfg)
    od = densidade_optica(img, fundo)

    prior = None
    if prob_modelo is not None:
        prior = cv2.resize(prob_modelo.astype(np.float32), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    vetores, _ = estimar_tintas(od, cfg, prior)
    conc = separar_tintas(od, vetores)
    c_car = conc[..., 0]
    c_outros = conc[..., 1:].max(axis=2) if conc.shape[2] > 1 else np.zeros_like(c_car)
    if prior is not None:                       # o modelo veta regiões sem carimbo
        c_car = c_car * np.clip(prior * 1.5, 0, 1)

    escala = img.shape[0] / img_bgr.shape[0]
    masc, (c_nit, limiar), forte = binarizar(c_car, c_outros, cfg, escala)
    masc, roi = limpar(masc, cfg, prior)
    saidas = exportar(c_nit, limiar, masc, forte, vetores[0], cfg, pasta, nome)

    if depurar:
        def u8(x, mx):
            return np.clip(x / mx * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(pasta / f"{nome}_dbg_1_normalizada.png"), img)
        cv2.imwrite(str(pasta / f"{nome}_dbg_2_fundo.png"), fundo)
        cv2.imwrite(str(pasta / f"{nome}_dbg_3_tinta_carimbo.png"), 255 - u8(c_car, forte * 2))
        cv2.imwrite(str(pasta / f"{nome}_dbg_4_interferentes.png"), 255 - u8(c_outros, forte * 2))
        cv2.imwrite(str(pasta / f"{nome}_dbg_5_mascara.png"), np.where(masc, 0, 255).astype(np.uint8))
        print("tintas (absorção B,G,R):")
        for i, v in enumerate(vetores):
            print("  ", "carimbo     " if i == 0 else "interferente", np.round(v, 3))
        print(f"otsu={forte:.3f} traco={cfg.traco}")
    return saidas


def main() -> None:
    ap = argparse.ArgumentParser(description="Isola o traço de um carimbo em PNG.")
    ap.add_argument("entrada", help="imagem ou PDF")
    ap.add_argument("-o", "--saida", default="saida")
    ap.add_argument("--pagina", type=int, default=1, help="página do PDF (1 = primeira)")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--bbox", help="x0,y0,x1,y1 do carimbo na página (saída do detector)")
    ap.add_argument("--mascara-modelo", help="PNG de probabilidade da classe carimbo")
    ap.add_argument("--escala", type=float, default=4.0)
    ap.add_argument("--cor", default="tinta", help='"tinta", "preto" ou "#RRGGBB"')
    ap.add_argument("--traco", type=float, default=0.5,
                    help="espessura do traço: 0 = fino/aberto … 1 = cheio (padrão 0.5)")
    ap.add_argument("--depurar", action="store_true", help="salva imagens intermediárias")
    a = ap.parse_args()

    img = carregar(a.entrada, a.pagina, a.dpi)
    if a.bbox:
        x0, y0, x1, y1 = (int(v) for v in a.bbox.split(","))
        img = img[y0:y1, x0:x1]
    prob = None
    if a.mascara_modelo:
        prob = cv2.imread(a.mascara_modelo, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    cfg = Config(escala=a.escala, cor_saida=a.cor, traco=a.traco)
    saidas = isolar_carimbo(img, Path(a.saida), Path(a.entrada).stem, cfg, prob, a.depurar)
    for k, v in saidas.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()