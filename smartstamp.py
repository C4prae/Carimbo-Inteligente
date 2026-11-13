import mimetypes
from enum import Enum
from pathlib import Path

import cv2
import ollama
import pymupdf
from pydantic import BaseModel, ValidationError
from ultralytics import YOLO

PROMPTS_DIR = Path(__file__).parent / "prompts"


class TipoDocumento(str, Enum):
    NASCIMENTO = "nascimento"
    OBITO = "obito"
    CASAMENTO = "casamento"
    DESCONHECIDO = "desconhecido"


class Classificacao(BaseModel):
    """Schema que o Ollama é obrigado a seguir na resposta."""
    tipo: TipoDocumento


class SmartStamp:
    MODELO_YOLO = "best.pt"
    MODELO_VISAO = "qwen3-vl"
    CONFIANCA_MINIMA_YOLO = 0.75
    PROMPT_CLASSIFICACAO = (PROMPTS_DIR / "classificacao.txt").read_text(encoding="utf-8")

    def __init__(self, img):
        self.img_path = Path(img)
        self.saved_img_path = None
        self.results = None
        self.tipo_documento = None
        self.format()

    def format(self):
        """Detecta o formato e converte PDF para imagem (primeira página)."""
        tipo, _ = mimetypes.guess_type(str(self.img_path))

        if tipo is None:
            raise ValueError(f"Tipo de arquivo desconhecido: {self.img_path}")
        elif tipo == "application/pdf":
            print("[-] PDF detectado, convertendo para imagem...")
            self.img_path = self.pdf_para_jpg(self.img_path)[0]
            print(f"[-] Imagem gerada: {self.img_path}")
        elif tipo.startswith("image/"):
            print("[-] Imagem detectada")
        else:
            raise ValueError(f"Formato não suportado: {tipo}")

    @staticmethod
    def pdf_para_jpg(pdf_path, saida_dir=None, dpi=200):
        pdf_path = Path(pdf_path)
        saida_dir = Path(saida_dir) if saida_dir else pdf_path.parent
        saida_dir.mkdir(parents=True, exist_ok=True)

        caminhos = []
        with pymupdf.open(pdf_path) as doc:
            for i, pagina in enumerate(doc, start=1):
                pix = pagina.get_pixmap(dpi=dpi)
                caminho = saida_dir / f"{pdf_path.stem}_pag{i}.jpg"
                pix.save(str(caminho))
                caminhos.append(caminho)

        return caminhos

    def findStamp(self):
        model = YOLO(self.MODELO_YOLO)
        self.results = model(str(self.img_path), save=True, project="images", exist_ok=True)

        save_dir = Path(self.results[0].save_dir)
        self.saved_img_path = save_dir / self.img_path.name

        print("[-] Identificação concluída")

    def cropStamp (self, conf_min=None):
        """Recorta os carimbos detectados com confiança acima de conf_min."""
        if self.results is None:
            print("Erro: Execute primeiro o método findStamp()!")
            return

        conf_min = self.CONFIANCA_MINIMA_YOLO if conf_min is None else conf_min

        img = cv2.imread(str(self.img_path))
        if img is None:
            print(f"Erro ao carregar a imagem em: {self.img_path}")
            return

        save_dir = Path(self.results[0].save_dir)
        output_dir = save_dir / self.img_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        boxes = self.results[0].boxes
        count = 0

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])

            if conf > conf_min:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Garante limites válidos dentro da imagem
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                output_path = output_dir / f"carimbo_{i}_conf_{conf:.2f}.png"
                cv2.imwrite(str(output_path), crop)
                count += 1
                print(f"[-] Carimbo {count} guardado em: {output_path.resolve()}")

        print(f"[-] Processo concluído. {count} recortes guardados na pasta '{output_dir}'.")

    def pln(self):
        """Classifica o documento e guarda o resultado em self.tipo_documento."""
        resposta = ollama.chat(
            model=self.MODELO_VISAO,
            messages=[
                {
                    "role": "user",
                    "content": self.PROMPT_CLASSIFICACAO,
                    "images": [str(self.img_path)],
                }
            ],
            format=Classificacao.model_json_schema(),
            options={"temperature": 0},
        )

        try:
            self.tipo_documento = Classificacao.model_validate_json(
                resposta["message"]["content"]
            ).tipo
        except ValidationError:
            self.tipo_documento = TipoDocumento.DESCONHECIDO

        print(f"[-] Tipo de documento: {self.tipo_documento.value}")
        return self.tipo_documento