import  cv2 as cv
import face_recognition
import numpy as np



'''
Ibraheem                =   face_recognition.load_image_file("Faces/Saba Iqbal.jpg")
Ibraheem                =   cv.cvtColor(Ibraheem, cv.COLOR_BGR2RGB)
Ibraheem_location1      =   face_recognition.face_locations(Ibraheem)[0]
cv.rectangle(Ibraheem,())
Ibraheem_encoding1      =   face_recognition.face_encodings(Ibraheem)[0]
cv.imshow("AEB", Ibraheem)'''


saba_iqbal =   face_recognition.load_image_file(("Faces/Saba Iqbal.jpg"))
saba_iqbal=   cv.cvtColor(saba_iqbal,cv.COLOR_BGR2RGB)
face_location   =   face_recognition.face_locations(saba_iqbal)[0]
cv.rectangle(saba_iqbal,(face_location[3],face_location[0]),(face_location[1],face_location[2]),(0,255,0),2)
saba_iqbal_encoding    =   face_recognition.face_encodings(saba_iqbal)[0]

Aish_Fatima =   face_recognition.load_image_file(("Faces/Aish Fatima.jpg"))
Aish_Fatima =   cv.cvtColor(Aish_Fatima,cv.COLOR_BGR2RGB)
face_location   =   face_recognition.face_locations(Aish_Fatima)[0]
cv.rectangle(Aish_Fatima,(face_location[3],face_location[0]),(face_location[1],face_location[2]),(0,255,0),2)
Aish_Fatima_encoding    =   face_recognition.face_encodings(Aish_Fatima)[0]



Ishaal_Noor  =   face_recognition.load_image_file("Faces/Eshaal Fatima.jpg")
Ishaal_Noor  =   cv.cvtColor(Ishaal_Noor,cv.COLOR_BGR2RGB)
Ishaal_Noor_location    =   face_recognition.face_locations(Ishaal_Noor)[0]
cv.rectangle(Ishaal_Noor, (Ishaal_Noor_location[3], Ishaal_Noor_location[0])
             , (Ishaal_Noor_location[1], Ishaal_Noor_location[2]), (0, 255, 0), 2)
Ishaal_Noor_encodings    =   face_recognition.face_encodings(Ishaal_Noor)[0]


x = face_recognition.compare_faces([Aish_Fatima_encoding], saba_iqbal_encoding, tolerance=0.1)
distance    =   face_recognition.face_distance([Aish_Fatima_encoding], saba_iqbal_encoding)
print(x, distance)

cv.putText(saba_iqbal, f"{x} {round(distance[0], 2)}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),4)

#cv.imshow("Aish", Aish_Fatima)
#cv.imshow("Ishal", Ishaal_Noor)
#cv.imshow("saba_iqbal", saba_iqbal)
cv.waitKey(0)