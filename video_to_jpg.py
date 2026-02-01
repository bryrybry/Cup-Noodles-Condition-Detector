import cv2
import os
from tkinter import Tk, filedialog


FRAMES_PER_SECOND = 6 


Tk().withdraw()
video_paths = filedialog.askopenfilenames(
    title="Select video files",
    filetypes=[("Video files", "*.mp4 *.mov")]
)

while True:
    classes = ["Finished", "Opened", "Sealed"]
    print("0 - Finished")
    print("1 - Opened")
    print("2 - Sealed")
    inputclass = input("Choose: ")
    inputclass = int(inputclass)
    if inputclass >= 0 and inputclass <= 2:
        output_dir = classes[inputclass]
        break
    else:
        print("Invalid!")

output_dir = f"./data/{output_dir}"
os.makedirs(output_dir, exist_ok=True)

for video_path in video_paths:

    filename = os.path.splitext(os.path.basename(video_path))[0]

    if not video_path:
        raise SystemExit("No file selected")

    cap = cv2.VideoCapture(video_path) # Opens video file
    if not cap.isOpened():
        raise SystemExit("Cannot open video")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    if FRAMES_PER_SECOND <= 0 or FRAMES_PER_SECOND > orig_fps:
        FRAMES_PER_SECOND = orig_fps 

    step = int(round(orig_fps / FRAMES_PER_SECOND))
    if step < 1:
        step = 1

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read() # read NEXT frame
        if not ret: 
            break # no more frames in the video

        if frame_idx % step == 0:
            cv2.imwrite(f"{output_dir}/{filename}_{saved_idx}.jpg", frame)
            saved_idx += 1

        frame_idx += 1

cap.release()
