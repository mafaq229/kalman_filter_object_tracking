import cv2
import numpy as np
from src.kalman import KalmanFilter
from src.tracker import ObjectTracker, TrackingConfig


def process_video(video_path, template_region, detector_type="template", save_frames=None):
    """Process a video file using Kalman Filter tracking.

    Args:
        video_path (str): Path to the input video file.
        template_region (dict): Dictionary containing template region {'x': int, 'y': int, 'w': int, 'h': int}
        detector_type (str): Type of detector to use ("hog" or "template").
        save_frames (dict): Dictionary mapping frame numbers to save paths.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize configuration
    config = TrackingConfig(
        noise_std_x=5.0,
        noise_std_y=5.0,
        template_size={'w': template_region['w'], 'h': template_region['h']}
    )

    # Initialize tracker
    tracker = ObjectTracker(config)
    
    # Get first frame for template initialization
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Initialize detector with template
    tracker.initialize_detector(detector_type, first_frame, template_region)
    
    # Initialize Kalman Filter with template center
    initial_x = template_region['x'] + template_region['w'] // 2
    initial_y = template_region['y'] + template_region['h'] // 2
    kf = KalmanFilter(init_x=initial_x, init_y=initial_y)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect object and update Kalman filter
        detection = tracker.detect_object(frame, detector_type)
        if detection is not None:
            z_x, z_y, z_w, z_h = detection
            pred_x, pred_y = kf.process(z_x, z_y)
            
            # Visualize tracking
            out_frame = tracker.visualize_tracking(frame, (z_x, z_y, z_w, z_h), (pred_x, pred_y))
            cv2.imshow('Tracking', out_frame)
            
            # Save frame if requested
            if save_frames and frame_count in save_frames:
                cv2.imwrite(save_frames[frame_count], out_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 20 == 0:
            print(f'Processing frame {frame_count}')

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to demonstrate Kalman Filter tracking."""
    # Example usage
    video_path = "test1.mp4"  # Replace with your video path
    
    # Define template region
    template_region = {
        'x': 140,
        'y': 72,
        'w': 50,
        'h': 50
    }
    
    # Optional: specify frames to save
    # save_frames = {
    #     10: "output/frame_10.jpg",
    #     20: "output/frame_20.jpg"
    # }

    process_video(video_path, template_region, detector_type="template")


if __name__ == "__main__":
    main()
