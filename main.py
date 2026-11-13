from pathlib import Path
from smartstamp import SmartStamp
 
if __name__ == "__main__":
    imgpath = Path("saida_fluxo/docs/teste (1)_carimbado.png")
 
    doc = SmartStamp(imgpath)
    doc.findStamp()
    doc.cropStamp()
    #doc.pln()

 