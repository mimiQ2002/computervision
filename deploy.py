import cv2
from ultralytics import YOLO

# Load the correct model
model = YOLO('yolov11n.pt')  # Ensure the model file exists

# Open a video stream
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection correctly
    results = model.predict(frame, conf=0.5)  # Ensure correct function usage

    # Access the first result in the list
    result = results[0] if results else None

    if result:
        # Draw results on the frame
        annotated_frame = result.plot()
    else:
        annotated_frame = frame  # If no detection, show the original frame

    # Display the frame
    cv2.imshow('Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
