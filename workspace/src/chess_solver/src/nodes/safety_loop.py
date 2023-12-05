import cv2
import numpy as np
import time

def is_blue_present(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    """
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    hsv_values = hsv[center_y, center_x]
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
    print(f"HSV values at the center pixel: {hsv_values}")
    """ 

  
    lower_blue = np.array([100, 180, 140])
    upper_blue = np.array([120, 230, 220])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return np.any(result)


def open_camera():
    return cv2.VideoCapture(0)

def release_camera(cap):
    cap.release()

def safety_cam(cap):
    blue_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        blue_present = is_blue_present(frame)
        if blue_present and not blue_detected:
            print("Blue is present in the frame.")

        elif not blue_present and blue_detected:
            print("Blue is no longer present. Waiting for 3 seconds...")
            elapsed_time = 0
            while elapsed_time < 3:
                ret, frame = cap.read()
                blue_present = is_blue_present(frame)

                if blue_present:
                    print("Blue detected again. Aborting move.")
                    break

                cv2.imshow('Original Frame', frame)
                elapsed_time += 1
                time.sleep(1)

            if not blue_present:
                print("Making move!")

        blue_detected = blue_present
        cv2.imshow('Original Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    cap = open_camera()

    if cap is None:
        print("Error: Could not open camera.")
        return

    try:
        safety_cam(cap)

    finally:
        release_camera(cap)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
