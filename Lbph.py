import cv2
import webbrowser

detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPHfotoseditadas.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(2)
new = 2

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(55,55))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255,0,0), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""


        if id == 1:
            nome = 'Juliano'
            if confianca <= 5:
                url = "http://localhost/pagina/index1.php?id=1"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 2:
            nome = 'Pirogue'
            if confianca <= 5:
                url = "http://localhost/pagina/index1.php?id=2"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 3:
            nome = 'Herik'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=3"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 4:
            nome = 'Marcos'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=4"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 5:
            nome = 'Tales'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=5"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 6:
            nome = 'Rafael'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=6"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 7:
            nome = 'Guilherme'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=7"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 8:
            nome = 'William'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=8"
                webbrowser.open(url, new=new, )
                camera.release()
        elif id == 9:
            nome = 'Ivan'
            if cv2.waitKey(1) == ord('s'):
                url = "http://localhost/pagina/index1.php?id=9"
                webbrowser.open(url, new=new, )
                camera.release()
        else:
            nome = 'Desconhecido'
        cv2.putText(imagem, nome, ((x -28),y +(a+38)), font, 2, (0,255,0))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (0,0,255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()