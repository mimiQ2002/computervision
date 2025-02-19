import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('yolov11n.pt')  # Choose the model best suited for your needs

# Open a video stream
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, 1 for an external camera

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Access the first result in the list
    result = results[0]  # Assuming there's only one image/frame being processed

    # Draw results on the frame
    annotated_frame = result.plot()

    # Display the frame
    cv2.imshow('Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
