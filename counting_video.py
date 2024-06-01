from pathlib import Path
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import time

# ---------------------------------------------------------
source_path = 'SampleVideo3.mp4'   #path to sample video, for webcam : 0, for external webcam : 1
classes_lst = [0]  #objects to count, 0 : person
region_points_lst = [(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]  #points of region
device = "cpu" # enter 0 to use gpu
limit_time = 20 #the time limit for webcam (seconds)
# ---------------------------------------------------------

track_history = defaultdict(list)


# fix performance of this function
def run(
    weights="yolov8n.pt",
    source=source_path,
    device=device,
    view_img=True,
    save_img=True,
    exist_ok=False,
    classes=classes_lst,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
    region_points = region_points_lst
):
    counting_regions = [
        {
            "name": "YOLOv8 Rectangle Region",
            "polygon": Polygon(region_points),
            "counts": 0,
            "dragging": False,
            "region_color": (37, 255, 225),  # BGR Value
            "text_color": (0, 0, 0),  # Region Text Color
        },
    ]
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Check source path
    if not source in [0, 1]:
        if not Path(source).exists():
            raise FileNotFoundError(f"Source path '{source}' does not exist.")


    # Setup Model
    model = YOLO("yolov8n.pt")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    #save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    #save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
    s_time = time.time()
    # Iterate over video frames
    
    limit_ok = True
    while videocapture.isOpened() and limit_ok:
        # check the limit time for webcam
        if source in [0, 1]:
            if time.time() - s_time < limit_time:
                limit_ok = True
            else:
                limit_ok = False
        else:
            limit_ok = True
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")

            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
    
run() 
    
    
    
    
    
    
    
