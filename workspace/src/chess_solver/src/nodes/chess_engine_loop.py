from safety_loop import open_camera, release_camera, safety_cam

def main_loop():
    cap = open_camera()

    if cap is None:
        print("Error: Could not open camera.")
        return

    try:
        safety_cam(cap)

    finally:
        release_camera(cap)

if __name__ == "__main__":
    main_loop()
