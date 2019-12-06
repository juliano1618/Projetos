import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 40)

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        print (id)
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces
ids, faces = getImagemComId()

print("Treinando o Algoritmo...")



lbph.train(faces, ids)
lbph.write('classificadorLBPHfotoseditadas.yml')

print("Treinamento Concluido")
