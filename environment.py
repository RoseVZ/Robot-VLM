#!/usr/bin/env python3
"""
PyBullet simulation environment - Drop-in replacement with Kenny models
Compatible with existing navigation pipeline
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class RobotEnvironment:
    """Manages PyBullet simulation environment - Compatible with navigation system"""
    
    def __init__(self, gui=True):
        """Initialize PyBullet environment"""
        self.gui = gui
        
        if gui:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=45,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Enable better rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = None
        self.objects = {}
        
        # Bounds for navigation system
        self.bounds = {
            'x_min': -10, 'x_max': 10,
            'y_min': -10, 'y_max': 10,
            'z_ground': 0
        }
        
        print("✓ PyBullet environment initialized")
    
    def load_robot(self, position=[0, -5, 0.5], orientation=[0, 0, 0, 1]):
        """Load robot at a safe starting position"""
        try:
            self.robot_id = p.loadURDF("husky/husky.urdf", position, orientation)
            print(f"✓ Husky robot loaded at position ({position[0]:.1f}, {position[1]:.1f})")
        except:
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.5)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.5,
                                               rgbaColor=[1, 0.5, 0, 1])
            self.robot_id = p.createMultiBody(baseMass=5,
                                             baseCollisionShapeIndex=collision_shape,
                                             baseVisualShapeIndex=visual_shape,
                                             basePosition=position)
            print(f"✓ Orange robot loaded at position ({position[0]:.1f}, {position[1]:.1f})")
        
        return self.robot_id
    
    def _load_kenny_model(self, obj_path, position=[0, 0, 0], 
                         orientation=[0, 0, 0, 1], scale=10.0, 
                         fixed=True, color=None):
        """
        Load Kenny's OBJ model
        
        Args:
            obj_path: Path to .obj file
            position: [x, y, z] position
            orientation: [x, y, z, w] quaternion
            scale: Scale multiplier
            fixed: If True, object won't move
            color: Optional [r, g, b, a] color override
        
        Returns:
            Object ID or None if failed
        """
        try:
            # Create visual shape
            visual_kwargs = {
                'shapeType': p.GEOM_MESH,
                'fileName': obj_path,
                'meshScale': [scale, scale, scale],
            }
            
            if color is not None:
                visual_kwargs['rgbaColor'] = color
            
            visual_shape = p.createVisualShape(**visual_kwargs)
            
            # Create collision shape
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=obj_path,
                meshScale=[scale, scale, scale]
            )
            
            # Create multi-body
            mass = 0 if fixed else 1.0
            
            obj_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation
            )
            
            return obj_id
            
        except Exception as e:
            print(f"  ✗ Failed to load {obj_path}: {e}")
            return None
    
    # ========================================
    # SCENE CREATION METHODS (For compatibility)
    # ========================================
    
    def create_realistic_scene(self):
        """
        🔥 COMPATIBILITY METHOD
        Creates Kenny apartment (alias for create_multiroom_apartment)
        """
        print("\n✓ Using Kenny realistic apartment scene")
        return self.create_multiroom_apartment()
    
    def create_colorful_scene(self):
        """
        🔥 COMPATIBILITY METHOD
        Creates Kenny apartment (alias for create_multiroom_apartment)
        """
        print("\n✓ Using Kenny colorful apartment scene")
        return self.create_multiroom_apartment()
    
    def create_simple_scene(self):
        """
        🔥 COMPATIBILITY METHOD
        Creates Kenny apartment (alias for create_multiroom_apartment)
        """
        print("\n✓ Using Kenny simple apartment scene")
        return self.create_multiroom_apartment()
    
    def create_multiroom_apartment(self):
        """
        Create apartment with Kenny models
        
        Layout:
        ┌─────────────┐     ┌─────────────┐
        │   BEDROOM   │🚪   │   KITCHEN   │
        │  (BLUE)     │     │  (GREEN)    │
        │  (-4, 4)    │     │  (4, 4)     │
        └──────🚪─────┘     └──────🚪─────┘
                │               │
        ┌───────┴───────────────┴───────┐
        │      LIVING ROOM              │
        │      (BROWN)                  │
        │      (0, -4)                  │
        └───────────────────────────────┘
        """
        
        print("\n" + "="*60)
        print("CREATING KENNY 3-ROOM APARTMENT")
        print("="*60 + "\n")
        
        wall_height = 3.0
        wall_thickness = 0.2
        
        # Room colors
        colors = {
            'living': [0.8, 0.6, 0.3, 1],    # Brown
            'bedroom': [0.3, 0.5, 0.9, 1],   # Blue
            'kitchen': [0.4, 0.8, 0.4, 1],   # Green
        }
        
        # ========================================
        # WALLS
        # ========================================
        
        walls = []
        
        # Living room walls
        living_walls = [
            ([-6, -4, wall_height/2], [wall_thickness, 4, wall_height/2]),
            ([6, -4, wall_height/2], [wall_thickness, 4, wall_height/2]),
            ([0, -8, wall_height/2], [6, wall_thickness, wall_height/2]),
        ]
        
        for pos, half_extents in living_walls:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                        rgbaColor=colors['living'])
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            walls.append(wall)
        
        # Bedroom walls
        bedroom_walls = [
            ([-8, 4, wall_height/2], [wall_thickness, 4, wall_height/2]),
            ([-4, 8, wall_height/2], [4, wall_thickness, wall_height/2]),
            ([-6.5, 0, wall_height/2], [1.5, wall_thickness, wall_height/2]),
            ([-1.5, 0, wall_height/2], [1.5, wall_thickness, wall_height/2]),
            ([0, 1, wall_height/2], [wall_thickness, 1, wall_height/2]),
            ([0, 5.5, wall_height/2], [wall_thickness, 2.5, wall_height/2]),
        ]
        
        for pos, half_extents in bedroom_walls:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                        rgbaColor=colors['bedroom'])
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            walls.append(wall)
        
        # Kitchen walls
        kitchen_walls = [
            ([8, 4, wall_height/2], [wall_thickness, 4, wall_height/2]),
            ([4, 8, wall_height/2], [4, wall_thickness, wall_height/2]),
            ([1.5, 0, wall_height/2], [1.5, wall_thickness, wall_height/2]),
            ([6.5, 0, wall_height/2], [1.5, wall_thickness, wall_height/2]),
        ]
        
        for pos, half_extents in kitchen_walls:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                        rgbaColor=colors['kitchen'])
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            walls.append(wall)
        
        # Store walls
        for i, wall in enumerate(walls):
            self.objects[f'wall_{i}'] = wall
        
        # ========================================
        # COLORED FLOORS
        # ========================================
        
        floor_height = 0.02
        floor_markers = [
            ('living_floor', [0, -4, floor_height], [5.5, 3.5, 0.01], colors['living']),
            ('bedroom_floor', [-4, 4, floor_height], [3.5, 3.5, 0.01], colors['bedroom']),
            ('kitchen_floor', [4, 4, floor_height], [3.5, 3.5, 0.01], colors['kitchen']),
        ]
        
        for name, pos, half_extents, color in floor_markers:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                      baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[name] = marker
        
        print(f"✓ Created {len(walls)} walls + 3 colored floors")
        
        # ========================================
        # KENNY FURNITURE
        # ========================================
        
        print("\n📦 Loading furniture...")
        
        # Check if Kenny models exist
        kenny_available = os.path.exists('./object_models/loungeSofa.obj')
        
        if kenny_available:
            print("  ✓ Kenny models found - loading realistic furniture")
            self._load_kenny_furniture()
        else:
            print("  ⚠ Kenny models not found at './object_models/'")
            print("  ✓ Loading fallback PyBullet objects")
            self._load_fallback_furniture()
        
        # ========================================
        # SUMMARY
        # ========================================
        
        print("\n" + "="*60)
        print("3-ROOM APARTMENT SUMMARY:")
        print("="*60)
        print("""
        ┌─────────────┐     ┌─────────────┐
        │   BEDROOM   │🚪   │   KITCHEN   │
        │  (BLUE)     │     │  (GREEN)    │
        └──────🚪─────┘     └──────🚪─────┘
                │               │
        ┌───────┴───────────────┴───────┐
        │      LIVING ROOM              │
        │      (BROWN)                  │
        └───────────────────────────────┘
        """)
        print("🟤 BROWN floor = LIVING ROOM")
        print("🔵 BLUE floor = BEDROOM")
        print("🟢 GREEN floor = KITCHEN")
        
        # Print available objects for CLIP/GroundingDINO
        obj_names = [k for k in self.objects.keys() if not k.startswith('wall') and 'floor' not in k]
        print(f"\n📋 Available objects: {obj_names}")
        print("="*60 + "\n")
        
        return self.objects
    
    def _load_kenny_furniture(self):
        """Load Kenny .obj furniture models"""
        
        # Base rotation to fix Y-up to Z-up
        base_rotation = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
        
        furniture_config = [
            # LIVING ROOM
            # {
            #     'name': 'sofa',
            #     'model': './object_models/loungeSofa.obj',
            #     'position': [-3, -5, 0],
            #     'scale': 3.0,
            #     'yaw': 180,
            #     'color': [1, 0.2, 0.1, 1]
            # },
            {
                'name': 'coffee_table',
                'model': './object_models/table.obj',
                'position': [-3, -6, 0],
                'scale': 2.0,
                'yaw': 180,
                'color': [0.3, 0.2, 0.1, 1]
            },
            {
                'name': 'lamp',
                'model': './object_models/lampSquareFloor.obj',
                'position': [2, -5, 0],
                'scale': 2.0,
                'yaw': 0,
                'color': None
            },
            
            # BEDROOM
            {
                'name': 'bed',
                'model': './object_models/bedDouble.obj',
                'position': [-5, 5, 0],
                'scale': 2.0,
                'yaw': 180,
                'color': None
            },
            {
                'name': 'nightstand_left',
                'model': './object_models/cabinetBedDrawer.obj',
                'position': [-6, 7, 0],
                'scale': 3.0,
                'yaw': 180,
                'color': [0.3, 0.2, 0.1, 1]
            },
            {
                'name': 'nightstand_right',
                'model': './object_models/cabinetBedDrawer.obj',
                'position': [-3, 7, 0],
                'scale': 3.0,
                'yaw': 180,
                'color': [0.3, 0.2, 0.1, 1]
            },
            {
                'name': 'coat_rack',
                'model': './object_models/coatRackStanding.obj',
                'position': [-7, 2, 0],
                'scale': 3.0,
                'yaw': 0,
                'color': [173/255, 216/255, 230/255, 1]
            },
            
            # KITCHEN
            {
                'name': 'fridge',
                'model': './object_models/kitchenFridgeLarge.obj',
                'position': [6, 5, 0],
                'scale': 3.0,
                'yaw': 180,
                'color': [0.5, 0.5, 0.5, 1]
            },
            {
                'name': 'stove',
                'model': './object_models/kitchenStove.obj',
                'position': [3, 5, 0],
                'scale': 3.0,
                'yaw': 180,
                'color': [0.5, 0.5, 0.5, 1]
            },
            {
                'name': 'sink',
                'model': './object_models/kitchenSink.obj',
                'position': [2, 5, 0],
                'scale': 3.0,
                'yaw': 180,
                'color': [0.5, 0.5, 0.5, 1]
            },
            {
                'name': 'toaster',
                'model': './object_models/toaster.obj',
                'position': [5, 5, 2],
                'scale': 3.0,
                'yaw': 180,
                'color': [173/255, 216/255, 230/255, 1]
            },
        ]
        
        loaded_count = 0
        for config in furniture_config:
            if not os.path.exists(config['model']):
                continue
            
            # Apply rotations
            yaw_rad = np.radians(config['yaw'])
            user_rotation = p.getQuaternionFromEuler([0, 0, yaw_rad])
            orientation = p.multiplyTransforms([0,0,0], base_rotation, 
                                              [0,0,0], user_rotation)[1]
            
            obj_id = self._load_kenny_model(
                obj_path=config['model'],
                position=config['position'],
                orientation=orientation,
                scale=config['scale'],
                fixed=True,
                color=config['color']
            )
            
            if obj_id is not None:
                self.objects[config['name']] = obj_id
                loaded_count += 1
        
        print(f"  ✓ Loaded {loaded_count} Kenny furniture pieces")
    
    def _load_fallback_furniture(self):
        """Load PyBullet URDF furniture as fallback"""
        
        fallback_furniture = [
            # Living room
            ('sofa', 'table/table.urdf', [-3, -5, 0], 0.8),
            ('coffee_table', 'table_square/table_square.urdf', [-3, -6, 0], 0.6),
            ('lamp', 'sphere2.urdf', [2, -5, 0.3], 0.3),
            
            # Bedroom
            ('bed', 'table/table.urdf', [-5, 5, 0], 1.2),
            ('nightstand_left', 'table_square/table_square.urdf', [-6, 7, 0], 0.5),
            ('nightstand_right', 'table_square/table_square.urdf', [-3, 7, 0], 0.5),
            ('teddy', 'teddy_vhacd.urdf', [-7, 2, 0.5], 1.2),
            
            # Kitchen
            ('fridge', 'cube.urdf', [6, 5, 0], 1.0),
            ('stove', 'cube.urdf', [3, 5, 0], 0.8),
            ('sink', 'cube.urdf', [2, 5, 0], 0.7),
            ('toaster', 'sphere2.urdf', [5, 5, 0.5], 0.3),
        ]
        
        loaded_count = 0
        for name, urdf, pos, scale in fallback_furniture:
            try:
                obj_id = p.loadURDF(urdf, basePosition=pos, globalScaling=scale, useFixedBase=True)
                self.objects[name] = obj_id
                loaded_count += 1
            except:
                pass
        
        print(f"  ✓ Loaded {loaded_count} fallback furniture pieces")
    
    # ========================================
    # COMPATIBILITY METHODS (Keep for pipeline)
    # ========================================
    
    def get_robot_state(self):
        """Get robot position and orientation"""
        if self.robot_id is None:
            return None
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        return {
            'position': np.array(pos),
            'orientation': np.array(orn),
            'euler': np.array(euler),
            'yaw': euler[2]
        }
    
    def get_object_position(self, object_name):
        """Get position of named object"""
        if object_name in self.objects:
            pos, _ = p.getBasePositionAndOrientation(self.objects[object_name])
            return np.array(pos)
        return None
    
    def set_robot_position(self, position, yaw=0):
        """Set robot position and orientation"""
        if self.robot_id is None:
            return
        
        orientation = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_id, position, orientation)
    
    def get_camera_image(self, width=640, height=480):
        """Get camera image from robot's perspective"""
        if self.robot_id is None:
            return None, None
        
        robot_state = self.get_robot_state()
        pos = robot_state['position']
        yaw = robot_state['yaw']
        
        camera_pos = [
            pos[0] + 0.3 * np.cos(yaw),
            pos[1] + 0.3 * np.sin(yaw),
            pos[2] + 0.5
        ]
        
        target_pos = [
            pos[0] + 2.0 * np.cos(yaw),
            pos[1] + 2.0 * np.sin(yaw),
            pos[2] + 0.5
        ]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height,
            nearVal=0.1, farVal=10.0
        )
        
        img = p.getCameraImage(
            width, height,
            view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb = np.array(img[2])[:, :, :3]
        depth = np.array(img[3])
        
        return rgb, depth
    
    def get_camera_image_at_angle(self, position, yaw, width=640, height=480):
        """Get camera image from specific position and angle"""
        camera_pos = [
            position[0] + 0.3 * np.cos(yaw),
            position[1] + 0.3 * np.sin(yaw),
            position[2] + 0.5
        ]
        
        target_pos = [
            position[0] + 2.0 * np.cos(yaw),
            position[1] + 2.0 * np.sin(yaw),
            position[2] + 0.5
        ]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height,
            nearVal=0.1, farVal=10.0
        )
        
        img = p.getCameraImage(
            width, height,
            view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb = np.array(img[2])[:, :, :3]
        depth = np.array(img[3])
        
        return rgb, depth
    
    def step_simulation(self):
        """Step the simulation forward"""
        p.stepSimulation()
        if self.gui:
            time.sleep(1./240.)
    
    def close(self):
        """Close PyBullet connection"""
        p.disconnect()
        print("✓ PyBullet disconnected")


# Quick test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING KENNY APARTMENT ENVIRONMENT")
    print("="*60 + "\n")
    
    env = RobotEnvironment(gui=True)
    env.load_robot(position=[0, -5, 0.5])
    
    # Test that all scene creation methods work
    env.create_realistic_scene()
    
    print("\n▶ Simulation running... Close window to exit.")
    
    try:
        while True:
            env.step_simulation()
    except KeyboardInterrupt:
        print("\nStopped")
    
    env.close()