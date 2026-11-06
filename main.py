from pathlib import Path
from smartstamp import SmartStamp
 
if __name__ == "__main__":
    imgpath = Path("test_images/teste (2).pdf")
 
    doc = SmartStamp(imgpath)
    doc.findStamp()
    #doc.isolateStamp()
    doc.pln()
 
    print(f"Tipo do documento: {doc.tipo_documento.value}")
 