#!/usr/bin/env python3


import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from typing import Tuple, Optional, List, Dict

class GroundingDINO: 
    """
    Enhanced vision module with Grounding DINO + 3D localization
    
    Pipeline:
    1. Grounding DINO: Detect object with bounding box + confidence
    2. Depth: Get depth values for detected region
    3. 3D Projection: Convert to world coordinates
    """
    
    # Confidence thresholds 
    # :SHOUD change based on experiments 
    HIGH_CONFIDENCE = 0.70
    MEDIUM_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.40
    
    # Color ranges for verification
    COLOR_RANGES = {
        'red': [((0, 50, 50), (10, 255, 255)), ((170, 50, 50), (180, 255, 255))],
        'blue': [((90, 50, 50), (130, 255, 255))],
        'green': [((35, 50, 50), (85, 255, 255))],
        'yellow': [((15, 50, 50), (35, 255, 255))],
        'brown': [((10, 50, 20), (20, 255, 200))],
        'orange': [((5, 50, 50), (15, 255, 255))],
        'purple': [((125, 50, 50), (165, 255, 255))],
        'white': [((0, 0, 180), (180, 30, 255))],
        'black': [((0, 0, 0), (180, 255, 50))],
    }
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", debug_mode=True):
        """
        Initialize Grounding DINO model
        
        Args:
            model_id: "grounding-dino-tiny" (faster) or "grounding-dino-base" (more accurate)
            debug_mode: Enable visualization
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_mode = debug_mode
        
        print(f"Loading Grounding DINO model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        print("✓ Grounding DINO model loaded")
        
        # Detection thresholds
        self.box_threshold = 0.30
        self.text_threshold = 0.25
        
        print(f"\n3D Localization Pipeline:")
        print(f"  1. Grounding DINO detection (open-vocabulary)")
        print(f"  2. Depth from PyBullet")
        print(f"  3. 3D projection to world coordinates")
        print(f"\nDebug mode: {'ENABLED' if debug_mode else 'disabled'}")
    
    def capture_panoramic_views_with_depth(self, env, robot_id, num_views=8):
        """
        Capture RGB + Depth from multiple angles
        
        Returns list of dicts with:
        - 'image': RGB array
        - 'depth': depth array (meters)
        - 'angle': camera angle (radians)
        - 'camera_pos': 3D camera position
        - 'target_pos': 3D look-at target
        - 'view_matrix': camera view matrix
        - 'projection_matrix': camera projection matrix
        """
        robot_state = env.get_robot_state()
        robot_pos = robot_state['position']
        
        print(f"\nCapturing {num_views} panoramic views with depth...")
        
        views = []
        width, height = 320, 320
        
        for i in range(num_views):
            angle = (2 * np.pi * i) / num_views
            
            cam_height = 1.5
            cam_pos = [robot_pos[0], robot_pos[1], cam_height]
            
            look_distance = 5.0
            target_pos = [
                robot_pos[0] + look_distance * np.cos(angle),
                robot_pos[1] + look_distance * np.sin(angle),
                cam_height - 0.5
            ]
            
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=[0, 0, 1]
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=90.0, aspect=1.0, nearVal=0.1, farVal=10.0
            )
            
            # Get RGB + Depth
            img_arr = p.getCameraImage(
                width, height, view_matrix, projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            rgb_array = np.array(img_arr[2]).reshape(height, width, 4)[:, :, :3]
            depth_buffer = np.array(img_arr[3]).reshape(height, width)
            
            near = 0.1
            far = 10.0
            
            # Convert depth buffer to meters
            depth_standard = far * near / (far - (far - near) * depth_buffer)
            depth_linear = near + (far - near) * depth_buffer
            depth_direct = depth_buffer * far
            
            center_standard = depth_standard[height//2, width//2]
            center_linear = depth_linear[height//2, width//2]
            center_direct = depth_direct[height//2, width//2]
            
            # Choose most reasonable depth
            options = [
                ('standard', depth_standard, center_standard),
                ('linear', depth_linear, center_linear),
                ('direct', depth_direct, center_direct)
            ]
            
            best_option = min(options, key=lambda x: abs(x[2] - 2.0) if 0.5 < x[2] < 8 else 999)
            depth_meters = best_option[1]
            
            views.append({
                'image': rgb_array,
                'depth': depth_meters,
                'angle': angle,
                'camera_pos': np.array(cam_pos),
                'target_pos': np.array(target_pos),
                'view_matrix': view_matrix,
                'projection_matrix': projection_matrix,
                'width': width,
                'height': height,
                'fov': 90.0,
                'near': near,
                'far': far
            })
        
        print(f"✓ Captured {num_views} views with depth maps")
        return views
    
    def detect_object_with_grounding_dino(self, instruction: str, views: List[Dict]) -> List[Tuple]:
        """
        Use Grounding DINO to find object across all views
        
        Returns:
            List of (view_dict, detection_dict, score) tuples sorted by score
        """
        print(f"\nGrounding DINO Search: '{instruction}'")
        
        # Extract object name
        object_name = self._parse_instruction(instruction)
        print(f"Target: '{object_name}'")
        
        # Prepare text query
        text_labels = [[object_name]]
        
        print(f"\n Detection Results:")
        
        all_results = []
        
        for idx, view in enumerate(views):
            pil_image = Image.fromarray(view['image'].astype('uint8'))
            
            # Run Grounding DINO
            inputs = self.processor(
                images=pil_image,
                text=text_labels,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]]  # [height, width]
            )
            
            result = results[0]
            
            if len(result["boxes"]) > 0:
                # Get best detection in this view
                best_idx = result["scores"].argmax()
                score = result["scores"][best_idx].item()
                box = result["boxes"][best_idx].cpu().numpy()  # [x1, y1, x2, y2]
                label = result["labels"][best_idx]
                
                x1, y1, x2, y2 = box
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                detection = {
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                    'bbox_xyxy': box.tolist(),
                    'score': score,
                    'label': label,
                    'center': center,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                angle_deg = np.degrees(view['angle'])
                print(f"  ✓ View {angle_deg:6.1f}°: Score={score:.3f}, Label='{label}', BBox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                
                all_results.append((view, detection, score))
            else:
                angle_deg = np.degrees(view['angle'])
                print(f"  ✗ View {angle_deg:6.1f}°: No detection")
        
        # Sort by score
        all_results.sort(key=lambda x: x[2], reverse=True)
        
        return all_results
    
    def get_object_center_3d(self, view: Dict, detection: Dict) -> Optional[np.ndarray]:
        """
        Get 3D world coordinates of detected object using depth
        
        Args:
            view: View dictionary with depth, camera matrices, etc.
            detection: Detection dict with bbox, center, etc.
            
        Returns:
            3D position in world coordinates [x, y, z] or None if failed
        """
        x, y, w, h = detection['bbox']
        cx, cy = detection['center']
        
        # Extract depth in bounding box region
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Ensure bounds
        h_img, w_img = view['depth'].shape
        x1, x2 = max(0, x1), min(w_img, x2)
        y1, y2 = max(0, y1), min(h_img, y2)
        
        depth_crop = view['depth'][y1:y2, x1:x2]
        
        # Get valid depths
        valid_depths = depth_crop[(depth_crop > view['near']) & (depth_crop < view['far'])]
        
        if len(valid_depths) == 0:
            print("⚠ No valid depth measurements!")
            return None
        
        # Use 10th percentile (robust to background pixels)
        percentile_10_depth = np.percentile(valid_depths, 10)
        median_depth = np.median(valid_depths)
        
        print(f"\n   Depth measurement:")
        print(f"     Min: {valid_depths.min():.2f}m, Max: {valid_depths.max():.2f}m")
        print(f"     Median: {median_depth:.2f}m, 10th percentile: {percentile_10_depth:.2f}m")
        print(f"     → Using: {percentile_10_depth:.2f}m")
        
        # Use detection center
        center_pixel = np.array([cx, cy])
        
        print(f"  Detection center: ({cx:.0f}, {cy:.0f})")
        print(f"  Camera angle: {np.degrees(view['angle']):.1f}°")
        
        # Convert to 3D camera frame
        point_camera = self.pixel_to_camera_frame(
            center_pixel, percentile_10_depth, view
        )
        
        print(f"  Point in camera frame: ({point_camera[0]:.2f}, {point_camera[1]:.2f}, {point_camera[2]:.2f})")
        
        # Convert to world frame
        point_world = self.camera_to_world_frame(point_camera, view)
        
        print(f"  ✓ Object 3D position (world): ({point_world[0]:.2f}, {point_world[1]:.2f}, {point_world[2]:.2f})")
        
        return point_world
    
    def pixel_to_camera_frame(self, pixel: np.ndarray, depth: float, view: Dict) -> np.ndarray:
        """Convert pixel + depth to 3D camera coordinates"""
        u, v = pixel
        width = view['width']
        height = view['height']
        fov = view['fov']
        
        focal_length = (width / 2.0) / np.tan(np.radians(fov / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        
        x_cam = (u - cx) * depth / focal_length
        y_cam = (v - cy) * depth / focal_length
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def camera_to_world_frame(self, point_camera: np.ndarray, view: Dict) -> np.ndarray:
        """Transform point from camera frame to world frame"""
        angle = view['angle']
        x_cam, y_cam, z_cam = point_camera
        
        # Camera directions
        forward_x = np.cos(angle)
        forward_y = np.sin(angle)
        right_x = np.cos(angle - np.pi/2)
        right_y = np.sin(angle - np.pi/2)
        
        # Transform to world
        world_x = view['camera_pos'][0] + z_cam * forward_x + x_cam * right_x
        world_y = view['camera_pos'][1] + z_cam * forward_y + x_cam * right_y
        world_z = view['camera_pos'][2] - y_cam
        
        return np.array([world_x, world_y, world_z])
    
    def search_current_room(self, instruction: str, env, robot_id):
        """
        COMPATIBLE WITH ORIGINAL INTERFACE
        Phase 1: Search current room with Grounding DINO + 3D localization
        
        Returns:
            (decision, best_view, best_score)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: GROUNDING DINO SEARCH WITH 3D LOCALIZATION")
        print(f"{'='*60}")
        print(f"Query: '{instruction}'")
        
        # Step 1: Capture views with depth
        views = self.capture_panoramic_views_with_depth(env, robot_id, num_views=8)
        
        # Step 2: Grounding DINO detection
        matches = self.detect_object_with_grounding_dino(instruction, views)
        
        if not matches:
            print("\n No detections found")
            self._cached_object_position = None
            return "EXPLORE_OTHER_ROOMS", None, 0.0
        
        best_view, best_detection, best_score = matches[0]
        
        # Step 3: Assess confidence
        confidence_level = self._assess_confidence_level(best_score)
        print(f"\nConfidence: {confidence_level} (score: {best_score:.3f})")
        print(f"Label: '{best_detection['label']}'")
        
        # Step 4: Localize in 3D
        if best_score >= self.LOW_CONFIDENCE:
            print(f"\nAttempting 3D localization...")
            
            object_position_3d = self.get_object_center_3d(best_view, best_detection)
            
            if object_position_3d is not None:
                self._cached_object_position = object_position_3d
                
                if self.debug_mode:
                    self._visualize_detection(best_view, best_detection, object_position_3d, instruction)
            else:
                print("⚠ Could not localize in 3D")
                self._cached_object_position = None
        else:
            self._cached_object_position = None
        
        # Decision logic
        if confidence_level == "HIGH":
            decision = "NAVIGATE"
        elif confidence_level == "MEDIUM":
            decision = "NAVIGATE"
        elif confidence_level == "LOW":
            decision = "EXPLORE_NEARBY"
        else:
            decision = "EXPLORE_OTHER_ROOMS"
        
        print(f"Decision: {decision}")
        
        # Visualize confidence
        self._visualize_confidence_assessment(best_view, instruction, best_score,
                                             confidence_level, decision, best_detection['label'])
        
        return decision, best_view, best_score
    
    def navigate_to_detected_object(self, instruction, env, robot_id, map_builder,
                                    best_view, best_score):
        """
        COMPATIBLE WITH ORIGINAL INTERFACE
        Phase 2: Navigate to detected object using TRUE 3D position
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: NAVIGATION TO OBJECT")
        print(f"{'='*60}")
        print(f"Confidence score: {best_score:.3f}")
        
        # Use cached 3D position if available
        if hasattr(self, '_cached_object_position') and self._cached_object_position is not None:
            object_center = self._cached_object_position
            print(f"✓ Using TRUE 3D position: ({object_center[0]:.2f}, {object_center[1]:.2f}, {object_center[2]:.2f})")
        else:
            # Fallback to direction estimate
            angle = best_view['angle']
            robot_state = env.get_robot_state()
            robot_pos = robot_state['position']
            
            distance_estimate = 2.5 if best_score > 0.5 else 3.5
            
            object_center = np.array([
                robot_pos[0] + distance_estimate * np.cos(angle),
                robot_pos[1] + distance_estimate * np.sin(angle),
                0.4
            ])
            print(f"⚠ Using direction estimate: ({object_center[0]:.2f}, {object_center[1]:.2f})")
        
        print(f"\nSampling candidate positions...")
        candidates = self._sample_candidate_positions_around_object(object_center, env, num_samples=100)
        
        print(f"Finding best navigable goal...")
        best_position = self._find_best_navigable_goal(candidates, object_center, map_builder, env)
        
        print(f"✓ Goal selected: ({best_position[0]:.2f}, {best_position[1]:.2f})")
        
        self._visualize_navigation_plan(best_view, candidates, object_center,
                                       best_position, map_builder, env)
        
        return best_position
    
    def _sample_candidate_positions_around_object(self, object_center, env, num_samples=100):
        """Sample candidate positions in a ring around object"""
        candidates = []
        
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples + np.random.uniform(-0.2, 0.2)
            distance = np.random.uniform(1.0, 2.5)
            
            x = object_center[0] + distance * np.cos(angle)
            y = object_center[1] + distance * np.sin(angle)
            z = 0.4
            
            candidates.append(np.array([x, y, z]))
        
        return candidates
    
    def _find_best_navigable_goal(self, candidates, object_center, map_builder, env):
        """Find navigable position closest to object"""
        valid_candidates = []
        
        for candidate in candidates:
            candidate_grid = map_builder.world_to_grid(candidate)
            
            if not (0 <= candidate_grid[0] < map_builder.height_cells and
                    0 <= candidate_grid[1] < map_builder.width):
                continue
            
            if map_builder.is_free(candidate_grid):
                distance_to_object = np.linalg.norm(candidate[:2] - object_center[:2])
                valid_candidates.append({
                    'position': candidate,
                    'distance_to_object': distance_to_object
                })
        
        if not valid_candidates:
            distances = [np.linalg.norm(c[:2] - object_center[:2]) for c in candidates]
            best_idx = np.argmin(distances)
            return candidates[best_idx]
        
        valid_candidates.sort(key=lambda x: x['distance_to_object'])
        return valid_candidates[0]['position']
    
    def _assess_confidence_level(self, score: float) -> str:
        """Assess confidence level from Grounding DINO score"""
        if score >= self.HIGH_CONFIDENCE:
            return "HIGH"
        elif score >= self.MEDIUM_CONFIDENCE:
            return "MEDIUM"
        elif score >= self.LOW_CONFIDENCE:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _parse_instruction(self, instruction: str) -> str:
        """Extract object name from instruction"""
        instruction = instruction.lower().strip()
        
        remove_words = ['find', 'go to', 'navigate to', 'locate', 'search for',
                       'look for', 'move to', 'head to', 'the', 'a', 'an']
        
        for word in remove_words:
            instruction = instruction.replace(f' {word} ', ' ').replace(f'{word} ', '')
        
        return instruction.strip()
    
    def _visualize_detection(self, view: Dict, detection: Dict,
                            object_pos: np.ndarray, target_name: str):
        """Visualize Grounding DINO detection + 3D localization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image with bounding box
        ax1 = axes[0]
        ax1.imshow(view['image'])
        
        x, y, w, h = detection['bbox']
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor='lime',
            facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Add label
        ax1.text(
            x, y - 5,
            f"{detection['label']} ({detection['score']:.2f})",
            color='white',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.8)
        )
        
        ax1.set_title(f"Detection: {target_name}\nAngle: {np.degrees(view['angle']):.0f}°",
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Depth map
        ax2 = axes[1]
        depth_viz = np.copy(view['depth'])
        depth_viz[depth_viz > 8] = 8
        im = ax2.imshow(depth_viz, cmap='plasma')
        
        # Mark detection center
        cx, cy = detection['center']
        ax2.plot(cx, cy, 'r*', markersize=20)
        
        ax2.set_title(f"Depth Map\nObject depth: {np.median(view['depth'][int(y):int(y+h), int(x):int(x+w)]):.2f}m",
                     fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Depth (m)')
        ax2.axis('off')
        
        # Overlay
        ax3 = axes[2]
        ax3.imshow(view['image'])
        rect2 = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='red',
            alpha=0.3
        )
        ax3.add_patch(rect2)
        ax3.plot(cx, cy, 'r*', markersize=20)
        
        ax3.set_title(f"3D Position\n({object_pos[0]:.1f}, {object_pos[1]:.1f}, {object_pos[2]:.1f})",
                     fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig('maps/grounding_dino_detection.png', dpi=150, bbox_inches='tight')
        print("✓ Saved detection visualization to maps/grounding_dino_detection.png")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    
    def _visualize_confidence_assessment(self, view, target_object, score,
                                        confidence_level, decision, label):
        """Visualize confidence assessment"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(view['image'])
        title = f"Best Match: '{target_object}'\n"
        title += f"Detected: '{label}'\n"
        title += f"Score: {score:.3f} ({confidence_level})"
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2.barh(['Score'], [score], 
                color='green' if score >= self.HIGH_CONFIDENCE else 
                      'orange' if score >= self.MEDIUM_CONFIDENCE else 'red')
        ax2.axvline(self.HIGH_CONFIDENCE, color='green', linestyle='--', linewidth=2,
                   label=f'High ({self.HIGH_CONFIDENCE:.2f})')
        ax2.axvline(self.MEDIUM_CONFIDENCE, color='orange', linestyle='--', linewidth=2,
                   label=f'Medium ({self.MEDIUM_CONFIDENCE:.2f})')
        ax2.axvline(self.LOW_CONFIDENCE, color='red', linestyle='--', linewidth=2,
                   label=f'Low ({self.LOW_CONFIDENCE:.2f})')
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel('Grounding DINO Confidence Score', fontsize=11)
        ax2.set_title(f'Decision: {decision}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('maps/confidence_assessment.png', dpi=150, bbox_inches='tight')
        print("✓ Saved confidence assessment to maps/confidence_assessment.png")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    
    def _visualize_navigation_plan(self, view, candidates, object_center,
                                  best_position, map_builder, env):
        """Visualize navigation plan"""
        robot_state = env.get_robot_state()
        robot_pos = robot_state['position']
        robot_grid = map_builder.world_to_grid(robot_pos)
        object_grid = map_builder.world_to_grid(object_center)
        goal_grid = map_builder.world_to_grid(best_position)
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(view['image'])
        ax1.set_title('Object Detection View', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(map_builder.grid, cmap='gray_r', origin='lower')
        
        ax2.plot(robot_grid[1], robot_grid[0], 'go', markersize=15, label='Robot', zorder=10)
        ax2.plot(object_grid[1], object_grid[0], 'mo', markersize=12,
                label=f'Object\n({object_center[0]:.1f}, {object_center[1]:.1f})', zorder=9)
        
        for candidate in candidates:
            cand_grid = map_builder.world_to_grid(candidate)
            if map_builder.is_free(cand_grid):
                ax2.plot(cand_grid[1], cand_grid[0], 'b.', markersize=2, alpha=0.3)
        
        ax2.plot(goal_grid[1], goal_grid[0], 'r*', markersize=25,
                label=f'Goal\n({best_position[0]:.1f}, {best_position[1]:.1f})', zorder=11)
        
        ax2.set_title('Navigation Plan', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('maps/navigation_plan.png', dpi=150, bbox_inches='tight')
        print("✓ Saved navigation plan to maps/navigation_plan.png")
        plt.show(block=False)
        plt.pause(2)
        plt.close()