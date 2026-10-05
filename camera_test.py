from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

print("Camera started. Press Q to exit.")

while True:
    frame = picam2.capture_array()

    # RGB → BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Raspberry Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
