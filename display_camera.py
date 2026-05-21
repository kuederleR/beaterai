import cv2
import sys

def main():
    # Attempt to open the external UVC camera (often index 1 on a MacBook if it has a built-in camera)
    camera_index = 2
    
    # We can explicitly use AVFoundation on macOS if default fails, 
    # but the default usually handles this.
    cap = cv2.VideoCapture(camera_index)

    # If the camera at index 1 is not opened, fallback to index 0
    if not cap.isOpened():
        print(f"Warning: Could not open camera at index {camera_index}. Trying index 0...")
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open any camera. Please make sure the camera is connected and you have granted camera permissions.")
        sys.exit(1)

    print(f"Successfully opened camera at index {camera_index}.")
    print("Press 'q' to quit the application.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow('External UVC Camera', frame)

        # Wait for 'q' key to stop
        # 1ms delay is necessary for cv2.imshow to process GUI events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and clean up windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
