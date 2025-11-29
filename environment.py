#!/usr/bin/env python3
"""PyBullet simulation environment with realistic 3D objects"""

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
                cameraDistance=12,
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
        
        # Larger bounds for better navigation
        self.bounds = {
            'x_min': -5, 'x_max': 5,
            'y_min': -5, 'y_max': 5,
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
            # Format: (name, urdf_file, position, orientation, scale)
            
            # Tables
            ('table', 'table/table.urdf', [2.5, 2.5, 0], [0, 0, 0, 1], 1.0),
            ('desk', 'table_square/table_square.urdf', [2.5, -2.0, 0], [0, 0, 0, 1], 0.8),
            
            # Chairs (using small tables as chairs)
            ('chair', 'table_square/table_square.urdf', [1.5, 2.5, 0], [0, 0, 0, 1], 0.5),
            ('chair_2', 'table_square/table_square.urdf', [3.5, 2.5, 0], [0, 0, 0, 1], 0.5),
            
            # Decorative objects
            ('teddy', 'teddy_vhacd.urdf', [-2.5, 2.5, 0.5], [0, 0, 0, 1], 1.5),
            ('duck', 'duck_vhacd.urdf', [-3, -2, 0.3], [0, 0, 0.707, 0.707], 1.0),
            
            # Boxes/containers
            ('red_box', 'cube.urdf', [3.5, -1.0, 0.5], [0, 0, 0, 1], 1.0),
            ('blue_box', 'sphere2.urdf', [-1, -3, 0.5], [0, 0, 0, 1], 1.0),
        ]
        
        loaded_count = 0
        for name, urdf_file, pos, orn, scale in realistic_objects:
            try:
                # Load URDF
                obj_id = p.loadURDF(
                    urdf_file,
                    basePosition=pos,
                    baseOrientation=orn,
                    globalScaling=scale,
                    useFixedBase=True  # Make objects static
                )
                self.objects[name] = obj_id
                loaded_count += 1
                print(f"  ✓ Loaded {name} ({urdf_file})")
            except Exception as e:
                print(f"  ✗ Could not load {name}: {e}")
                # Fallback to simple box
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
        
        # Colorful blocks for CLIP testing
        colorful_objects = [
            # (name, urdf, position, color_description)
            ('red_block', 'cube.urdf', [2.5, 2.5, 0.5], 'red'),
            ('blue_block', 'cube.urdf', [-2.5, 2.5, 0.5], 'blue'),
            # ('yellow_sphere', 'sphere2.urdf', [2.5, -2.5, 0.5], 'yellow'),
            # ('green_sphere', 'sphere2.urdf', [-2.5, -2.5, 0.5], 'green'),
            ('brown_teddy', 'teddy_vhacd.urdf', [0, 3, 0.5], 'brown'),
            ('yellow_duck', 'duck_vhacd.urdf', [0, -3, 0.3], 'yellow'),
        ]
        
        # Color mapping for blocks
        color_map = {
            'red': [1, 0, 0, 1],
            'blue': [0, 0, 1, 1],
            'green': [0, 1, 0, 1],
            'yellow': [1, 1, 0, 1],
            'brown': [0.6, 0.3, 0.1, 1],
        }
        
        loaded_count = 0
        for name, urdf_file, pos, color in colorful_objects:
            try:
                obj_id = p.loadURDF(urdf_file, basePosition=pos, useFixedBase=True)
                self.objects[name] = obj_id
                
                # Try to change color if it's a basic shape
                if color in color_map and 'block' in name:
                    p.changeVisualShape(obj_id, -1, rgbaColor=color_map[color])
                
                loaded_count += 1
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Could not load {name}: {e}")
        
        print(f"\n✓ Colorful scene created with {loaded_count} objects")
        print(f"  Available: {[name for name in self.objects.keys() if not name.startswith('wall')]}")
        return self.objects
    
    def _create_fallback_object(self, name, position):
        """Create a simple box as fallback if URDF loading fails"""
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3],
                                     rgbaColor=[0.5, 0.5, 0.5, 1])
        obj_id = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=collision,
                                   baseVisualShapeIndex=visual,
                                   basePosition=position)
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
    
    def step_simulation(self):
        """Step the simulation forward"""
        p.stepSimulation()
        if self.gui:
            time.sleep(1./240.)
    
    def close(self):
        """Close PyBullet connection"""
        p.disconnect()
        print("✓ PyBullet disconnected")


def test_scenes():
    """Test different scene types"""
    
    print("\n" + "="*60)
    print("TESTING PYBULLET 3D OBJECTS")
    print("="*60 + "\n")
    
    env = RobotEnvironment(gui=True)
    
    # Test 1: List available URDFs
    print("\n1. Discovering available URDF files...")
    input("Press Enter to list all available URDFs...")
    env.list_available_urdfs()
    
    # Test 2: Load robot
    input("\nPress Enter to load robot...")
    env.load_robot(position=[-3, -3, 0.3])
    
    # Test 3: Choose scene type
    print("\n" + "="*60)
    print("Choose scene type:")
    print("1. Simple scene (basic shapes)")
    print("2. Realistic scene (URDF models)")
    print("3. Colorful scene (for CLIP testing)")
    print("="*60)
    
    choice = input("\nEnter 1, 2, or 3: ").strip()
    
    if choice == "1":
        env.create_simple_scene()
    elif choice == "2":
        env.create_realistic_scene()
    else:
        env.create_colorful_scene()
    
    # Keep simulation running
    print("\nSimulation running... Close PyBullet window to exit.")
    try:
        while True:
            env.step_simulation()
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()


if __name__ == "__main__":
    test_scenes()