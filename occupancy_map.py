#!/usr/bin/env python3
"""Occupancy map generation using raycasting"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

class OccupancyMapBuilder:
    """Build 2D occupancy grid from 3D environment"""
    
    def __init__(self, bounds, resolution=0.1, height=1.0):
        """
        Args:
            bounds: dict with x_min, x_max, y_min, y_max
            resolution: meters per grid cell
            height: height at which to raycast
        """
        self.bounds = bounds
        self.resolution = resolution
        self.height = height
        
        # Calculate grid dimensions
        self.width = int((bounds['x_max'] - bounds['x_min']) / resolution)
        self.height_cells = int((bounds['y_max'] - bounds['y_min']) / resolution)
        
        # Initialize grid (0 = free, 1 = occupied)
        self.grid = np.zeros((self.height_cells, self.width), dtype=np.uint8)
        
        print(f"✓ Occupancy map: {self.width}x{self.height_cells} cells @ {resolution}m resolution")
    
    def build_from_raycasting(self, raycast_height=5.0):
        """Build occupancy map using vertical raycasting"""
        print("Building occupancy map via raycasting...")
        
        # Create grid of raycast origins
        x_coords = np.linspace(self.bounds['x_min'], 
                              self.bounds['x_max'], 
                              self.width)
        y_coords = np.linspace(self.bounds['y_min'], 
                              self.bounds['y_max'], 
                              self.height_cells)
        
        total_rays = self.width * self.height_cells
        rays_done = 0
        
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Raycast from above to ground
                ray_from = [x, y, raycast_height]
                ray_to = [x, y, self.bounds.get('z_ground', 0)]
                
                result = p.rayTest(ray_from, ray_to)[0]
                hit_fraction = result[2]
                
                # If ray hits something above ground, mark as occupied
                if hit_fraction < 1.0:
                    hit_position = result[3]
                    if hit_position[2] > 0.1:  # Above ground threshold
                        self.grid[i, j] = 1
                
                rays_done += 1
                if rays_done % 1000 == 0:
                    progress = 100 * rays_done / total_rays
                    print(f"  Progress: {progress:.1f}%", end='\r')
        
        print(f"\n✓ Occupancy map built: {np.sum(self.grid)} occupied cells")
        return self.grid
    
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid indices"""
        x, y = world_pos[0], world_pos[1]
        
        grid_x = int((x - self.bounds['x_min']) / self.resolution)
        grid_y = int((y - self.bounds['y_min']) / self.resolution)
        
        # Clamp to grid bounds
        grid_x = np.clip(grid_x, 0, self.width - 1)
        grid_y = np.clip(grid_y, 0, self.height_cells - 1)
        
        return (grid_y, grid_x)  # Note: (row, col) format
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        grid_y, grid_x = grid_pos
        
        x = self.bounds['x_min'] + grid_x * self.resolution
        y = self.bounds['y_min'] + grid_y * self.resolution
        
        return np.array([x, y, 0])
    
    def is_free(self, grid_pos):
        """Check if grid cell is free"""
        row, col = grid_pos
        if 0 <= row < self.height_cells and 0 <= col < self.width:
            return self.grid[row, col] == 0
        return False
    
    def inflate_obstacles(self, radius_cells=2):
        """Inflate obstacles for robot safety margin"""
        from scipy.ndimage import binary_dilation
        
        structure = np.ones((2*radius_cells+1, 2*radius_cells+1))
        self.grid = binary_dilation(self.grid, structure=structure).astype(np.uint8)
        print(f"✓ Obstacles inflated by {radius_cells} cells")
    
    def visualize(self, path=None, start=None, goal=None, save_path=None):
        """Visualize occupancy grid"""
        plt.figure(figsize=(10, 10))
        
        # Show grid
        plt.imshow(self.grid, cmap='gray_r', origin='lower')
        
        # Plot path
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            plt.plot(path_array[:, 1], path_array[:, 0], 
                    'b-', linewidth=2, label='Path')
        
        # Plot start and goal
        if start is not None:
            plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        if goal is not None:
            plt.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
        
        plt.title('Occupancy Grid')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Map saved to {save_path}")
        
        plt.show()


def test_occupancy_map():
    """Test occupancy map generation"""
    import sys
    sys.path.append('.')
    from environment import RobotEnvironment
    
    env = RobotEnvironment(gui=True)
    env.load_robot()
    env.create_simple_scene()
    
    # Build occupancy map
    map_builder = OccupancyMapBuilder(env.bounds, resolution=0.1)
    occupancy_grid = map_builder.build_from_raycasting()
    
    # Inflate obstacles
    map_builder.inflate_obstacles(radius_cells=3)
    
    # Visualize
    robot_state = env.get_robot_state()
    start_grid = map_builder.world_to_grid(robot_state['position'])
    
    map_builder.visualize(start=start_grid, save_path='maps/occupancy_map.png')
    
    env.close()


if __name__ == "__main__":
    test_occupancy_map()