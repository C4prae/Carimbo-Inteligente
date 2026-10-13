from pathlib import Path
from smartstamp import SmartStamp

if __name__ == "__main__":
    imgpath = Path("test_images/teste (3).jpg")

    doc = SmartStamp(imgpath)
    doc.findStamp()  # 1. Executa o YOLO e guarda o ficheiro na pasta[cite: 1]
    doc.isolateStamp() #2. isole as imagens encontradas pelos carimbos, com IoU superior a 75%