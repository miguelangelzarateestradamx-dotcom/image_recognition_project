import cv2

# Inicia la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                figura = "Triángulo"
            elif len(approx) == 4:
                aspect_ratio = w / float(h)
                figura = "Cuadrado" if 0.95 < aspect_ratio < 1.05 else "Rectángulo"
            elif len(approx) > 4:
                figura = "Círculo"
            else:
                figura = "Desconocida"

            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            cv2.putText(frame, figura, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Detección de Figuras", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
