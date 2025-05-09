"""Object tracking module with multiple tracking methods."""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass, field


@dataclass
class TrackingConfig:
    """Configuration for object tracking."""
    noise_std_x: float = 5.0
    noise_std_y: float = 5.0
    process_noise: Optional[np.ndarray] = None
    measurement_noise: Optional[np.ndarray] = None
    template_size: Optional[Dict[str, int]] = None
    detection_threshold: float = 0.5

    def __post_init__(self):
        if self.process_noise is None:
            self.process_noise = 0.1 * np.eye(4)
        if self.measurement_noise is None:
            self.measurement_noise = 0.1 * np.eye(2)


class ObjectTracker:
    """Main object tracking class with multiple tracking methods."""

    def __init__(self, config: TrackingConfig):
        """Initialize the tracker with configuration."""
        self.config = config
        self.hog_detector: Optional[cv2.HOGDescriptor] = None
        self.template: Optional[np.ndarray] = None
        self.frame_count: int = 0

    def initialize_detector(self, detector_type: str, template_frame: Optional[np.ndarray] = None,
                          template_region: Optional[Dict[str, int]] = None) -> None:
        """Initialize the appropriate detector based on type."""
        if detector_type == "hog":
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif detector_type == "template":
            if template_frame is None or template_region is None:
                raise ValueError("Template frame and region required for template matching")
            self.template = template_frame[
                template_region['y']:template_region['y'] + template_region['h'],
                template_region['x']:template_region['x'] + template_region['w']
            ]
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def detect_object(self, frame: np.ndarray, detector_type: str) -> Optional[Tuple[float, float, int, int]]:
        """Detect object in the frame using specified detector."""
        if detector_type == "hog":
            return self._detect_hog(frame)
        elif detector_type == "template":
            return self._detect_template(frame)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def _detect_hog(self, frame: np.ndarray) -> Optional[Tuple[float, float, int, int]]:
        """Detect objects using HOG detector."""
        if self.hog_detector is None:
            raise RuntimeError("HOG detector not initialized")

        rects, weights = self.hog_detector.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        if len(weights) == 0:
            return None

        max_w_id = np.argmax(weights)
        rect = rects[int(max_w_id)]
        z_x: float = float(rect[0])
        z_y: float = float(rect[1])
        z_w: int = int(rect[2])
        z_h: int = int(rect[3])
        
        # Add noise to measurements
        z_x = float(z_x) + z_w // 2 + np.random.normal(0, self.config.noise_std_x)
        z_y = float(z_y) + z_h // 2 + np.random.normal(0, self.config.noise_std_y)
        
        return float(z_x), float(z_y), z_w, z_h

    def _detect_template(self, frame: np.ndarray) -> Optional[Tuple[float, float, int, int]]:
        """Detect object using template matching."""
        if self.template is None:
            raise RuntimeError("Template not initialized")

        corr_map = cv2.matchTemplate(frame, self.template, cv2.TM_SQDIFF)
        z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
        
        if self.config.template_size is None:
            raise RuntimeError("Template size not configured")
            
        z_w = self.config.template_size['w']
        z_h = self.config.template_size['h']
        
        # Add noise to measurements
        z_x = float(z_x) + z_w // 2 + np.random.normal(0, self.config.noise_std_x)
        z_y = float(z_y) + z_h // 2 + np.random.normal(0, self.config.noise_std_y)
        
        return float(z_x), float(z_y), z_w, z_h

    def visualize_tracking(self, frame: np.ndarray, measurement: Tuple[float, float, int, int],
                          prediction: Tuple[float, float]) -> np.ndarray:
        """Visualize tracking results on the frame."""
        out_frame = frame.copy()
        
        # Draw measurement
        z_x, z_y, z_w, z_h = measurement
        cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
        cv2.rectangle(out_frame, 
                     (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                     (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                     (0, 0, 255), 2)
        
        # Draw prediction
        pred_x, pred_y = prediction
        cv2.circle(out_frame, (int(pred_x), int(pred_y)), 10, (255, 0, 0), 2)
        
        return out_frame

    def process_sequence(self, image_dir: str, kalman_filter, detector_type: str,
                        save_frames: Optional[Dict[int, str]] = None) -> None:
        """Process a sequence of images for tracking."""
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith('.jpg') and not f.startswith('.')])
        
        for img_file in image_files:
            frame = cv2.imread(os.path.join(image_dir, img_file))
            if frame is None:
                print(f"Failed to read image: {img_file}")
                continue

            # Detect object
            detection = self.detect_object(frame, detector_type)
            if detection is None:
                print(f"No detection in frame {self.frame_count}")
                continue

            z_x, z_y, z_w, z_h = detection
            
            # Update Kalman filter
            pred_x, pred_y = kalman_filter.process(z_x, z_y)
            
            # Visualize and display
            out_frame = self.visualize_tracking(frame, (z_x, z_y, z_w, z_h), (pred_x, pred_y))
            cv2.imshow('Tracking', out_frame)
            
            # Save frame if requested
            if save_frames and self.frame_count in save_frames:
                cv2.imwrite(save_frames[self.frame_count], out_frame)
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
            if self.frame_count % 20 == 0:
                print(f'Processing frame {self.frame_count}') 