import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class SmartStamp:
    def __init__(self, img):
        self.img_path = img
        self.saved_img_path = None
        self.results = None

    def findStamp(self):
        model = YOLO("best.pt")
        # Executa o YOLO e guarda os resultados
        self.results = model(self.img_path, save=True, project="images", exist_ok=True)
        
        save_dir = Path(self.results[0].save_dir)
        filename = Path(self.img_path).name
        self.saved_img_path = save_dir / filename
        
        print("[-] Identificação concluída")

    def isolateStamp(self):
        if self.results is None:
            print("Erro: Execute primeiro o método findStamp()!")
            return

        img = cv2.imread(str(self.img_path))
        if img is None:
            print(f"Erro ao carregar a imagem em: {self.img_path}")
            return

        # Obtém dinamicamente o diretório onde o YOLO salvou a predição
        save_dir = Path(self.results[0].save_dir)

        # Cria a pasta com o nome da imagem (ex: 'teste (3)') DENTRO da pasta de resultados do YOLO
        img_stem = Path(self.img_path).stem
        output_dir = save_dir / img_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        boxes = self.results[0].boxes
        count = 0

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            
            # Filtra por confiança superior a 0.75 (75%)
            if conf > 0.75:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Garantir limites válidos dentro da imagem
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                output_path = output_dir / f"carimbo_{i}_conf_{conf:.2f}.png"
                cv2.imwrite(str(output_path), crop)
                count += 1
                print(f"[-] Carimbo {count} (melhorado) guardado em: {output_path.resolve()}")

        print(f"[-] Processo concluído. {count} recortes guardados na pasta '{output_dir}'.")
        