import cv2
import numpy as np

def detect_water_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame, None

    bottle_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(bottle_contour) < 1000:
        return None, frame, None

    x, y, w, h = cv2.boundingRect(bottle_contour)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return None, frame, (x, y, w, h)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)

    heights = []
    for i in range(thresh.shape[0]):
        row = thresh[i, :]
        if cv2.countNonZero(row) > 0.5 * thresh.shape[1]:
            heights.append(i)

    if heights:
        water_level = max(heights)
        fill_ratio = (thresh.shape[0] - water_level) / thresh.shape[0]
        percentage = fill_ratio * 100
        return percentage, frame, (x, y, w, h)
    else:
        return 0, frame, (x, y, w, h)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Water Bottle Fill Level Detector\nPress 'q' to quit | 's' to save image\n")

    cv2.namedWindow('Water Detector', cv2.WINDOW_NORMAL)  # 👈 allows resizing

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        display_frame = frame.copy()

        if frame_count % 3 == 0:
            percentage, processed_frame, bbox = detect_water_level(display_frame)

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if percentage is not None:
                    water_y = int(y + h * (1 - percentage / 100))
                    cv2.line(display_frame, (x, water_y), (x+w, water_y), (255, 0, 0), 3)

                    fill_text = f"{percentage:.1f}% full"
                    cv2.putText(display_frame, fill_text, (x, max(30, y-15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Add status text
                    if percentage > 80:
                        status, color = "FULL", (0, 255, 0)
                    elif percentage > 50:
                        status, color = "HALF FULL", (0, 255, 255)
                    elif percentage > 20:
                        status, color = "LOW", (0, 165, 255)
                    else:
                        status, color = "EMPTY", (0, 0, 255)

                    cv2.putText(display_frame, status, (x, y + h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display_frame, "No bottle detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Instruction footer
            cv2.putText(display_frame, "Press 'q' to quit | 's' to save", 
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show final image
        cv2.imshow("Water Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"water_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Session closed.")

if __name__ == "__main__":
    main()
