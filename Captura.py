import cv2
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
classificadorOlhos = cv2.CascadeClassifier("haarcascade-eye.xml")

camera = cv2.VideoCapture(2)
amostra = 1
numeroAmostras = 5
id = input('Digite seu identificador: ')
largura , altura = 220, 220
print ("Capturando as faces...")

while (True):

    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(
                                                        imagemCinza,
                                                        scaleFactor=1.5,
                                                        minSize=(100,100)
                                                     )

    for (x, y , l, a) in facesDetectadas:
       cv2.rectangle(imagem, (x,y), (x+l , y+a), (0,0,255), 2)
       regiao = imagem[y:y + a, x:x + l]
       regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
       olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
       for (olhox, olhoy, olhol, olhoa) in olhosDetectados:
           cv2.rectangle(regiao, (olhox, olhoy), (olhox + olhol, olhoy + olhoa), (0, 255, 0), 2)

       if cv2.waitKey(1) & 0xFF == ord('q'):
            if np.average(imagemCinza) > 80:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                print("[foto " + str(amostra) + " capturada com sucesso]")
                amostra += 1
            else:
                print('Iluminação Insuficiente')
                break
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break
camera.release()

cv2.destroyAllWindowns()