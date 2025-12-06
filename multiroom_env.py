#!/usr/bin/env python3
"""PyBullet simulation environment with multi-room apartment"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class RobotEnvironment:
    """Manages PyBullet simulation environment"""
    
    def __init__(self, gui=True):
        """Initialize PyBullet environment"""
        self.gui = gui
        
        if gui:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=45,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = None
        self.objects = {}
        
        # Larger bounds for multi-room environment
        self.bounds = {
            'x_min': -10, 'x_max': 10,
            'y_min': -10, 'y_max': 10,
            'z_ground': 0
        }
        
        print("✓ PyBullet environment initialized")
    
    def load_robot(self, position=[-3, -3, 0.3], orientation=[0, 0, 0, 1]):
        """Load robot at a safe starting position"""
        try:
            self.robot_id = p.loadURDF("husky/husky.urdf", position, orientation)
            print(f"✓ Husky robot loaded at position ({position[0]:.1f}, {position[1]:.1f})")
        except:
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.5)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.5,
                                               rgbaColor=[0, 0.5, 1, 1])
            self.robot_id = p.createMultiBody(baseMass=5,
                                             baseCollisionShapeIndex=collision_shape,
                                             baseVisualShapeIndex=visual_shape,
                                             basePosition=position)
            print(f"✓ Simple robot loaded at position ({position[0]:.1f}, {position[1]:.1f})")
        
        return self.robot_id
    
    def create_multiroom_apartment(self):
        """
        Create a LARGE L-shaped 3-room apartment
        NO interior partitions - each room is a complete separate space
        
        Layout (L-shape):
        ┌─────────────────┬─────────────┐
        │                 │             │
        │   KITCHEN       │   BEDROOM   │
        │   (West)        │   (East)    │
        │                 │             │
        ├─────────────────┴─────────────┘
        │
        │   LIVING ROOM
        │   (South - L extension)
        │   Robot starts here
        │
        └─────────────────
        
        Each room is completely separate with its own walls and doorway
        """
        
        print("\n" + "="*60)
        print("CREATING L-SHAPED MULTI-ROOM APARTMENT")
        print("="*60 + "\n")
        
        # ========================================
        # LIVING ROOM (South - L extension)
        # Dimensions: 10m x 8m
        # ========================================
        
        print("🏠 Creating LIVING ROOM (South)...")
        
        # Living room walls (complete enclosure)
        living_walls = [
            # West wall
            ([-9, -4, 1], [0.2, 4, 2], [0.7, 0.7, 0.7, 1]),
            # East wall
            ([1, -4, 1], [0.2, 4, 2], [0.7, 0.7, 0.7, 1]),
            # South wall
            ([-4, -8, 1], [5, 0.2, 2], [0.7, 0.7, 0.7, 1]),
            # North wall - LEFT of doorway
            ([-6, 0, 1], [2.8, 0.2, 2], [0.7, 0.7, 0.7, 1]),
            # North wall - RIGHT of doorway
            ([-1, 0, 1], [1.8, 0.2, 2], [0.7, 0.7, 0.7, 1]),
            # Doorway at x=-3.5, width=2m leads to kitchen
        ]
        
        for i, (pos, half_extents, color) in enumerate(living_walls):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[f'living_wall_{i}'] = wall
        
        # Living room furniture
        living_objects = [
            ('sofa', 'table/table.urdf', [-6, -5, 0], 1.2, 'Large sofa'),
            ('coffee_table', 'table_square/table_square.urdf', [-4, -5, 0], 0.8, 'Coffee table'),
            ('tv_stand', 'table/table.urdf', [-6, -2, 0], 0.7, 'TV stand'),
            ('red_block', 'cube.urdf', [-1, -5, 0.5], 1.2, 'Red toy block'),
            ('blue_block', 'cube.urdf', [0, -6, 0.5], 1.2, 'Blue toy block'),
            ('duck', 'duck_vhacd.urdf', [-2, -7, 0.3], 1.2, 'Yellow duck toy'),
        ]
        
        for name, urdf, pos, scale, desc in living_objects:
            try:
                obj_id = p.loadURDF(urdf, basePosition=pos, globalScaling=scale, useFixedBase=True)
                self.objects[name] = obj_id
                if name == 'red_block':
                    p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1])
                elif name == 'blue_block':
                    p.changeVisualShape(obj_id, -1, rgbaColor=[0, 0, 1, 1])
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        # ========================================
        # KITCHEN (Northwest)
        # Dimensions: 8m x 8m
        # ========================================
        
        print("\n🍳 Creating KITCHEN (Northwest)...")
        
        # Kitchen walls (complete enclosure)
        kitchen_walls = [
            # West wall
            ([-9, 4, 1], [0.2, 4, 2], [0.8, 0.8, 0.75, 1]),
            # East wall - TOP of doorway to bedroom
            ([-1, 6.5, 1], [0.2, 1.3, 2], [0.8, 0.8, 0.75, 1]),
            # East wall - BOTTOM of doorway to bedroom
            ([-1, 1.5, 1], [0.2, 1.3, 2], [0.8, 0.8, 0.75, 1]),
            # Doorway at y=4, width=2m leads to bedroom
            # North wall
            ([-4, 8, 1], [5, 0.2, 2], [0.8, 0.8, 0.75, 1]),
            # South wall - LEFT of doorway to living
            ([-6, 0, 1], [2.8, 0.2, 2], [0.8, 0.8, 0.75, 1]),
            # South wall - RIGHT of doorway to living
            ([-1, 0, 1], [1.8, 0.2, 2], [0.8, 0.8, 0.75, 1]),
            # Doorway at x=-3.5, width=2m leads to living room
        ]
        
        for i, (pos, half_extents, color) in enumerate(kitchen_walls):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[f'kitchen_wall_{i}'] = wall
        
        # Kitchen furniture
        kitchen_objects = [
            ('kitchen_table', 'table/table.urdf', [-5, 4, 0], 1.3, 'Large kitchen table'),
            ('kitchen_chair_1', 'table_square/table_square.urdf', [-3.5, 4, 0], 0.6, 'Kitchen chair 1'),
            ('kitchen_chair_2', 'table_square/table_square.urdf', [-6.5, 4, 0], 0.6, 'Kitchen chair 2'),
            ('kitchen_chair_3', 'table_square/table_square.urdf', [-5, 5.5, 0], 0.6, 'Kitchen chair 3'),
            ('kitchen_chair_4', 'table_square/table_square.urdf', [-5, 2.5, 0], 0.6, 'Kitchen chair 4'),
            ('mug', 'sphere2.urdf', [-5, 4.5, 1.0], 0.35, 'Mug'),
            ('counter', 'table/table.urdf', [-7.5, 6, 0], 0.8, 'Kitchen counter'),
        ]
        
        for name, urdf, pos, scale, desc in kitchen_objects:
            try:
                obj_id = p.loadURDF(urdf, basePosition=pos, globalScaling=scale, useFixedBase=True)
                self.objects[name] = obj_id
                if name == 'mug':
                    p.changeVisualShape(obj_id, -1, rgbaColor=[0.6, 0.3, 0.1, 1])
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        # ========================================
        # BEDROOM (Northeast)
        # Dimensions: 8m x 8m
        # ========================================
        
        print("\n🛏️ Creating BEDROOM (Northeast)...")
        
        # Bedroom walls (complete enclosure)
        bedroom_walls = [
            # West wall - TOP of doorway from kitchen
            ([-1, 6.5, 1], [0.2, 1.3, 2], [0.85, 0.82, 0.75, 1]),
            # West wall - BOTTOM of doorway from kitchen
            ([-1, 1.5, 1], [0.2, 1.3, 2], [0.85, 0.82, 0.75, 1]),
            # Doorway at y=4, width=2m from kitchen
            # East wall
            ([7, 4, 1], [0.2, 4, 2], [0.85, 0.82, 0.75, 1]),
            # North wall
            ([3, 8, 1], [5, 0.2, 2], [0.85, 0.82, 0.75, 1]),
            # South wall
            ([3, 0, 1], [5, 0.2, 2], [0.85, 0.82, 0.75, 1]),
        ]
        
        for i, (pos, half_extents, color) in enumerate(bedroom_walls):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[f'bedroom_wall_{i}'] = wall
        
        # Bedroom furniture
        bedroom_objects = [
            ('bed', 'table/table.urdf', [3, 4, 0], 1.5, 'Large bed'),
            ('pillow_1', 'cube.urdf', [2.5, 5, 1.0], 0.5, 'Pillow 1'),
            ('pillow_2', 'cube.urdf', [3.5, 5, 1.0], 0.5, 'Pillow 2'),
            ('nightstand_left', 'table_square/table_square.urdf', [1, 4, 0], 0.6, 'Nightstand (left)'),
            ('nightstand_right', 'table_square/table_square.urdf', [5, 4, 0], 0.6, 'Nightstand (right)'),
            ('dresser', 'table/table.urdf', [5.5, 1.5, 0], 0.9, 'Dresser'),
            ('teddy', 'teddy_vhacd.urdf', [1.5, 7, 0.5], 1.5, 'Large teddy bear'),
        ]
        
        for name, urdf, pos, scale, desc in bedroom_objects:
            try:
                obj_id = p.loadURDF(urdf, basePosition=pos, globalScaling=scale, useFixedBase=True)
                self.objects[name] = obj_id
                if 'pillow' in name:
                    p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        # ========================================
        # SUMMARY
        # ========================================
        
        print("\n" + "="*60)
        print("L-SHAPED APARTMENT LAYOUT:")
        print("="*60)
        print("""
        ┌─────────────────┬─────────────┐
        │                 │             │
        │   KITCHEN       │   BEDROOM   │
        │   8x8m          │   8x8m      │
        │                 │             │
        ├─────────────────┴─────────────┘
        │
        │   LIVING ROOM
        │   10x8m
        │   🤖 Robot starts here
        │
        └─────────────────
        """)
        
        print("🏠 3-Room L-Shaped Apartment Created!")
        print("\n  Each room is COMPLETELY SEPARATE - no shared spaces")
        print(f"\n  Living Room: {len([k for k in self.objects.keys() if any(x in k for x in ['sofa', 'red_block', 'blue_block', 'duck', 'coffee', 'tv'])])} objects")
        print(f"  Kitchen: {len([k for k in self.objects.keys() if 'kitchen' in k or k in ['mug', 'counter']])} objects")
        print(f"  Bedroom: {len([k for k in self.objects.keys() if any(x in k for x in ['bed', 'pillow', 'teddy', 'nightstand', 'dresser'])])} objects")
        
        print("\n📋 Object Inventory:")
        print("  Living Room: sofa, coffee_table, tv_stand, red_block, blue_block, duck")
        print("  Kitchen: kitchen_table, 4× chairs, mug, counter")
        print("  Bedroom: bed, 2× pillows, 2× nightstands, dresser, teddy")
        
        print("\n🚪 Doorways (2m wide):")
        print("  Living → Kitchen: North doorway (~x=-3.5, y=0)")
        print("  Kitchen → Bedroom: East doorway (~x=-1, y=4)")
        
        print("\n📍 Room Centers:")
        print("  Living Room: (-4, -4)")
        print("  Kitchen: (-5, 4)")
        print("  Bedroom: (3, 4)")
        
        print("\n🤖 Robot starts in: Living Room")
        print("="*60 + "\n")
        
        return self.objects
    
    def add_room_floor_markers(self):
        """Add colored floor patches to visually distinguish rooms"""
        
        print("\nAdding room floor markers...")
        
        floor_markers = [
            # Living room (light blue carpet)
            ('living_floor', [-4, -4, 0.01], [4.8, 3.8, 0.01], [0.7, 0.8, 0.9, 0.5]),
            
            # Kitchen (white tiles)
            ('kitchen_floor', [-5, 4, 0.01], [3.8, 3.8, 0.01], [0.95, 0.95, 0.95, 0.7]),
            
            # Bedroom (warm beige carpet)
            ('bedroom_floor', [3, 4, 0.01], [3.8, 3.8, 0.01], [0.9, 0.85, 0.7, 0.5]),
        ]
        
        for name, pos, half_extents, color in floor_markers:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                      baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[name] = marker
        
        print("✓ Room floor markers added")
    
    def create_simple_scene(self):
        """Create a simple scene with basic shapes (kept for compatibility)"""
        
        # Walls - outer boundary
        wall_positions = [
            ([-4.8, 0, 1], [0.2, 4.8, 2]),    # West wall
            ([4.8, 0, 1], [0.2, 4.8, 2]),     # East wall
            ([0, -4.8, 1], [4.8, 0.2, 2]),    # South wall
            ([0, 4.8, 1], [4.8, 0.2, 2]),     # North wall
        ]
        
        for i, (pos, half_extents) in enumerate(wall_positions):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=[0.7, 0.7, 0.7, 1])
            wall = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=pos)
            self.objects[f'wall_{i}'] = wall
        
        # Furniture - well-spaced for navigation
        furniture = [
            ('table', [2.5, 2.5, 0.4], [0.6, 0.6, 0.4], [0.6, 0.3, 0.1, 1]),
            ('chair', [1.5, 2.5, 0.25], [0.35, 0.35, 0.5], [0.8, 0.4, 0.2, 1]),
            ('sofa', [-2.5, 2.5, 0.3], [1.0, 0.6, 0.6], [0.2, 0.6, 0.8, 1]),
            ('desk', [2.5, -2.0, 0.4], [0.8, 0.5, 0.4], [0.4, 0.25, 0.1, 1]),
            ('bookshelf', [3.5, -1.0, 0.8], [0.3, 0.6, 1.6], [0.3, 0.2, 0.1, 1]),
        ]
        
        for name, pos, half_extents, color in furniture:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=color)
            obj_id = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=collision,
                                      baseVisualShapeIndex=visual,
                                      basePosition=pos)
            self.objects[name] = obj_id
        
        print(f"✓ Scene created with {len(furniture)} objects")
        print(f"  Available: {[name for name in self.objects.keys() if not name.startswith('wall')]}")
        return self.objects
    
    def create_realistic_scene(self):
        """Create scene with realistic 3D URDF models from PyBullet"""
        
        print("\nLoading realistic 3D objects...")
        
        # Walls (same as before)
        wall_positions = [
            ([-4.8, 0, 1], [0.2, 4.8, 2]),
            ([4.8, 0, 1], [0.2, 4.8, 2]),
            ([0, -4.8, 1], [4.8, 0.2, 2]),
            ([0, 4.8, 1], [4.8, 0.2, 2]),
        ]
        
        for i, (pos, half_extents) in enumerate(wall_positions):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=[0.7, 0.7, 0.7, 1])
            wall = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=pos)
            self.objects[f'wall_{i}'] = wall
        
        # Realistic 3D objects using PyBullet's built-in URDFs
        realistic_objects = [
            # Tables
            ('table', 'table/table.urdf', [2.5, 2.5, 0], [0, 0, 0, 1], 1.0),
            ('desk', 'table_square/table_square.urdf', [2.5, -2.0, 0], [0, 0, 0, 1], 0.8),
            
            # Chairs
            ('chair', 'table_square/table_square.urdf', [1.5, 2.5, 0], [0, 0, 0, 1], 0.5),
            ('chair_2', 'table_square/table_square.urdf', [3.5, 2.5, 0], [0, 0, 0, 1], 0.5),
            
            # Decorative objects
            ('teddy', 'teddy_vhacd.urdf', [-2.5, 2.5, 0.5], [0, 0, 0, 1], 1.5),
            ('duck', 'duck_vhacd.urdf', [-3, -2, 0.3], [0, 0, 0.707, 0.707], 1.0),
            
            # Boxes
            ('red_box', 'cube.urdf', [3.5, -1.0, 0.5], [0, 0, 0, 1], 1.0),
            ('blue_box', 'sphere2.urdf', [-1, -3, 0.5], [0, 0, 0, 1], 1.0),
        ]
        
        loaded_count = 0
        for name, urdf_file, pos, orn, scale in realistic_objects:
            try:
                obj_id = p.loadURDF(urdf_file, basePosition=pos, baseOrientation=orn,
                                   globalScaling=scale, useFixedBase=True)
                self.objects[name] = obj_id
                loaded_count += 1
                print(f"  ✓ Loaded {name} ({urdf_file})")
            except Exception as e:
                print(f"  ✗ Could not load {name}: {e}")
                self._create_fallback_object(name, pos)
        
        print(f"\n✓ Scene created with {loaded_count} realistic objects")
        print(f"  Available: {[name for name in self.objects.keys() if not name.startswith('wall')]}")
        return self.objects
    
    def create_colorful_scene(self):
        """Create scene with colorful objects (good for CLIP testing)"""
        
        print("\nCreating colorful scene...")
        
        # Walls
        wall_positions = [
            ([-4.8, 0, 1], [0.2, 4.8, 2]),
            ([4.8, 0, 1], [0.2, 4.8, 2]),
            ([0, -4.8, 1], [4.8, 0.2, 2]),
            ([0, 4.8, 1], [4.8, 0.2, 2]),
        ]
        
        for i, (pos, half_extents) in enumerate(wall_positions):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=[0.7, 0.7, 0.7, 1])
            wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual, basePosition=pos)
            self.objects[f'wall_{i}'] = wall
        
        # Colorful blocks
        colorful_objects = [
            ('red_block', 'cube.urdf', [2.5, 2.5, 0.5], 'red'),
            ('blue_block', 'cube.urdf', [-2.5, 2.5, 0.5], 'blue'),
            ('brown_teddy', 'teddy_vhacd.urdf', [0, 3, 0.5], 'brown'),
            ('yellow_duck', 'duck_vhacd.urdf', [0, -3, 0.3], 'yellow'),
        ]
        
        color_map = {
            'red': [1, 0, 0, 1],
            'blue': [0, 0, 1, 1],
            'brown': [0.6, 0.3, 0.1, 1],
            'yellow': [1, 1, 0, 1],
        }
        
        loaded_count = 0
        for name, urdf_file, pos, color in colorful_objects:
            try:
                obj_id = p.loadURDF(urdf_file, basePosition=pos, useFixedBase=True)
                self.objects[name] = obj_id
                
                if color in color_map and 'block' in name:
                    p.changeVisualShape(obj_id, -1, rgbaColor=color_map[color])
                
                loaded_count += 1
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Could not load {name}: {e}")
        
        print(f"\n✓ Colorful scene created with {loaded_count} objects")
        return self.objects
    
    def _create_fallback_object(self, name, position):
        """Create a simple box as fallback if URDF loading fails"""
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3],
                                     rgbaColor=[0.5, 0.5, 0.5, 1])
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                   baseVisualShapeIndex=visual, basePosition=position)
        self.objects[name] = obj_id
        print(f"  ⚠ Created fallback box for {name}")
    
    def list_available_urdfs(self):
        """List all available URDF files in pybullet_data"""
        data_path = pybullet_data.getDataPath()
        print(f"\nSearching for URDF files in: {data_path}\n")
        
        available_urdfs = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.urdf'):
                    rel_path = os.path.relpath(os.path.join(root, file), data_path)
                    available_urdfs.append(rel_path)
        
        print("Available URDF files:")
        for i, urdf in enumerate(sorted(available_urdfs), 1):
            print(f"  {i}. {urdf}")
        
        return available_urdfs
    
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


def test_multiroom_scene():
    """Test the L-shaped multi-room apartment"""
    
    print("\n" + "="*60)
    print("L-SHAPED MULTI-ROOM APARTMENT TEST")
    print("="*60 + "\n")
    
    env = RobotEnvironment(gui=True)
    
    # Load robot in living room
    print("Loading robot in living room...")
    env.load_robot(position=[-4, -4, 0.3])
    
    # Create L-shaped apartment
    env.create_multiroom_apartment()
    
    # Optional: Add floor markers
    add_floors = input("\nAdd colored floor markers? (y/n): ").strip().lower()
    if add_floors == 'y':
        env.add_room_floor_markers()
    
    # Display room coordinates
    print("\n📍 NAVIGATION TARGETS:")
    print("  Living Room center: (-4, -4)")
    print("  Living→Kitchen doorway: (-3.5, 0)")
    print("  Kitchen center: (-5, 4)")
    print("  Kitchen→Bedroom doorway: (-1, 4)")
    print("  Bedroom center: (3, 4)")
    
    # Keep simulation running
    print("\n▶ Simulation running... Close window to exit.")
    print("  L-shaped layout - each room is completely separate!")
    
    try:
        while True:
            env.step_simulation()
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()


def test_scenes():
    """Test different scene types"""
    
    print("\n" + "="*60)
    print("TESTING PYBULLET 3D OBJECTS")
    print("="*60 + "\n")
    
    env = RobotEnvironment(gui=True)
    
    print("Loading robot...")
    env.load_robot(position=[-3, -3, 0.3])
    
    print("\n" + "="*60)
    print("Choose scene type:")
    print("1. Simple scene (basic shapes)")
    print("2. Realistic scene (URDF models)")
    print("3. Colorful scene (for CLIP testing)")
    print("4. L-shaped Multi-room apartment (for semantic exploration)")
    print("="*60)
    
    choice = input("\nEnter 1, 2, 3, or 4: ").strip()
    
    if choice == "1":
        env.create_simple_scene()
    elif choice == "2":
        env.create_realistic_scene()
    elif choice == "3":
        env.create_colorful_scene()
    elif choice == "4":
        env.create_multiroom_apartment()
        add_floors = input("\nAdd colored floor markers? (y/n): ").strip().lower()
        if add_floors == 'y':
            env.add_room_floor_markers()
    else:
        print("Invalid choice, using colorful scene")
        env.create_colorful_scene()
    
    print("\nSimulation running... Close PyBullet window to exit.")
    try:
        while True:
            env.step_simulation()
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()


if __name__ == "__main__":
    test_multiroom_scene()