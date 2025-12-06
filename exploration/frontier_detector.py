#!/usr/bin/env python3
"""
Frontier detection for autonomous exploration
Uses wavefront algorithm to find only reachable frontiers
"""

import numpy as np
from scipy.ndimage import label
from collections import deque
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class FrontierDetector:
    """
    Detect frontiers (boundaries between known and unknown space)
    using wavefront algorithm for reachability
    
    Only finds frontiers robot can actually reach!
    """
    
    def __init__(self, occupancy_map, min_frontier_size=5):
        """
        Args:
            occupancy_map: OptimizedOccupancyMapBuilder instance
            min_frontier_size: Minimum cells for valid frontier (default 5)
        """
        self.map = occupancy_map
        self.min_size = min_frontier_size
        
        print(f"✓ Frontier detector initialized")
        print(f"  Grid size: {self.map.width}x{self.map.height_cells}")
        print(f"  Resolution: {self.map.resolution}m")
        print(f"  Min frontier size: {min_frontier_size} cells")
    
    def detect_frontiers(self, robot_position) -> List[Dict]:
        """
        🔥 MAIN METHOD: Detect all reachable frontiers in current map
        
        Uses wavefront algorithm to only find frontiers in connected
        free space (automatically excludes areas outside walls!)
        
        Returns list of frontier regions with metadata:
        - id: Unique identifier
        - center_grid: (row, col) grid coordinates
        - center_world: (x, y, z) world coordinates  
        - cells: List of all frontier cells
        - size: Number of cells in frontier
        - distance: Distance from robot (meters)
        
        Args:
            robot_position: [x, y, z] robot position
            
        Returns:
            List[dict]: Frontier regions sorted by distance
        """
        
        print("\n" + "="*60)
        print("FRONTIER DETECTION (WAVEFRONT REACHABILITY)")
        print("="*60)
        
        # Step 1: Find frontier cells (only reachable ones!)
        frontier_cells = self._find_frontier_cells_reachable_only(robot_position)
        
        if len(frontier_cells) == 0:
            print("⚠ No reachable frontier cells detected")
            print("  (Map may be fully explored in reachable areas)")
            return []
        
        print(f"✓ Found {len(frontier_cells)} reachable frontier cells")
        
        # Step 2: Cluster into regions
        clusters = self._cluster_frontiers(frontier_cells)
        print(f"✓ Grouped into {len(clusters)} clusters")
        
        # Step 3: Filter by size
        valid_clusters = [c for c in clusters if len(c) >= self.min_size]
        removed = len(clusters) - len(valid_clusters)
        
        if removed > 0:
            print(f"  Removed {removed} small clusters (< {self.min_size} cells)")
        
        print(f"✓ {len(valid_clusters)} valid frontier regions")
        
        if len(valid_clusters) == 0:
            print("⚠ No valid frontiers after filtering")
            print("  (All frontiers too small - likely noise)")
            return []
        
        # Step 4: Compute metadata
        frontiers = self._compute_frontier_metadata(valid_clusters, robot_position)
        
        # Step 5: Sort by distance (nearest first)
        frontiers.sort(key=lambda f: f['distance'])
        
        print(f"\n📍 Detected Frontiers (nearest first):")
        for f in frontiers:
            print(f"  Frontier {f['id']}: "
                  f"size={f['size']:3d} cells, "
                  f"dist={f['distance']:5.1f}m, "
                  f"pos=({f['center_world'][0]:5.1f}, {f['center_world'][1]:5.1f})")
        
        return frontiers
    
    def _find_frontier_cells_reachable_only(self, robot_position) -> List[Tuple[int, int]]:
        """
        🔥 WAVEFRONT ALGORITHM: Find frontiers in connected free space only
        
        Two-stage process:
        1. Flood fill from robot to find all reachable free cells
        2. Check which reachable cells border unknown space
        
        This naturally excludes:
        - Areas outside apartment walls
        - Disconnected rooms (until doorway discovered)
        - Unreachable spaces
        
        Args:
            robot_position: [x, y, z] robot position
            
        Returns:
            List[(row, col)]: Frontier cells robot can reach
        """
        
        robot_grid = self.map.world_to_grid(robot_position)
        height, width = self.map.grid.shape
        
        print(f"\n  Starting wavefront from robot at grid {robot_grid}")
        
        # ═══════════════════════════════════════════════════════
        # STAGE 1: Flood fill to find reachable free space
        # ═══════════════════════════════════════════════════════
        
        queue = deque([robot_grid])
        visited = set([robot_grid])
        reachable_free = []
        
        # BFS from robot position
        while queue:
            current = queue.popleft()
            row, col = current
            
            # Mark as reachable free space
            reachable_free.append(current)
            
            # Explore 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = row + dr, col + dc
                    neighbor = (nr, nc)
                    
                    # Check bounds
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    
                    # Already visited?
                    if neighbor in visited:
                        continue
                    
                    visited.add(neighbor)
                    
                    # Is this cell free?
                    cell_value = self.map.grid[nr, nc]
                    if cell_value < 50:  # Free space (0-49)
                        queue.append(neighbor)
        
        print(f"  ✓ Flood fill found {len(reachable_free)} reachable free cells")
        
        # ═══════════════════════════════════════════════════════
        # STAGE 2: Find which reachable cells border unknown
        # ═══════════════════════════════════════════════════════
        
        frontier_cells = []
        
        for (row, col) in reachable_free:
            # Check if this free cell has unknown neighbor
            has_unknown_neighbor = False
            
            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = row + dr, col + dc
                    
                    # Check bounds
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    
                    # Is neighbor unknown?
                    if self.map.grid[nr, nc] == -1:  # Unknown
                        has_unknown_neighbor = True
                        break
                
                if has_unknown_neighbor:
                    break
            
            if has_unknown_neighbor:
                frontier_cells.append((row, col))
        
        print(f"  ✓ Found {len(frontier_cells)} cells bordering unknown space")
        
        return frontier_cells
    
    def _cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Group adjacent frontier cells into regions using connected components
        
        Uses 8-connectivity (cells touching at corners are connected)
        
        Args:
            frontier_cells: List of (row, col) tuples
            
        Returns:
            List of clusters, each cluster is List[(row, col)]
        """
        
        if len(frontier_cells) == 0:
            return []
        
        # Create binary map of frontier cells
        height, width = self.map.grid.shape
        frontier_map = np.zeros((height, width), dtype=bool)
        
        for row, col in frontier_cells:
            frontier_map[row, col] = True
        
        # Label connected components (8-connectivity)
        structure = np.ones((3, 3), dtype=int)
        labeled, num_features = label(frontier_map, structure=structure)
        
        # Extract clusters
        clusters = []
        for label_id in range(1, num_features + 1):
            cluster_coords = np.where(labeled == label_id)
            cluster_cells = list(zip(cluster_coords[0], cluster_coords[1]))
            clusters.append(cluster_cells)
        
        return clusters
    
    def _compute_frontier_metadata(self, clusters: List[List[Tuple[int, int]]], 
                                   robot_position) -> List[Dict]:
        """
        Add metadata to each frontier cluster
        
        Computes:
        - Center position (grid and world coordinates)
        - Size (number of cells)
        - Distance from robot (Euclidean distance in meters)
        
        Args:
            clusters: List of frontier clusters
            robot_position: [x, y, z] robot position
            
        Returns:
            List[dict]: Frontiers with complete metadata
        """
        
        frontiers = []
        
        robot_grid = self.map.world_to_grid(robot_position)
        
        for i, cluster in enumerate(clusters):
            # Compute center of mass
            rows = [c[0] for c in cluster]
            cols = [c[1] for c in cluster]
            
            center_row = int(np.mean(rows))
            center_col = int(np.mean(cols))
            
            # Convert to world coordinates
            world_pos = self.map.grid_to_world((center_row, center_col))
            world_pos[2] = robot_position[2]  # Keep same z as robot
            
            # Compute distance from robot (grid space first, then convert)
            distance_grid = np.sqrt(
                (center_row - robot_grid[0])**2 + 
                (center_col - robot_grid[1])**2
            )
            distance_meters = distance_grid * self.map.resolution
            
            # Create frontier dict
            frontier = {
                'id': i,
                'center_grid': (center_row, center_col),
                'center_world': world_pos,
                'cells': cluster,
                'size': len(cluster),
                'distance': distance_meters
            }
            
            frontiers.append(frontier)
        
        return frontiers
    
    def visualize_frontiers(self, frontiers: List[Dict], 
                           robot_position=None,
                           save_path='maps/frontiers.png'):
        """
        Visualize detected frontiers on occupancy map
        
        Args:
            frontiers: List of frontier dicts from detect_frontiers()
            robot_position: Optional [x, y, z] robot position to show
            save_path: Where to save visualization
        """
        
        if len(frontiers) == 0:
            print("⚠ No frontiers to visualize")
            return
        
        print(f"\n📊 Visualizing {len(frontiers)} frontiers...")
        
        # Create colored visualization
        grid = self.map.grid
        colored = np.zeros((*grid.shape, 3))
        
        # Base map colors
        colored[grid == -1] = [0.5, 0.5, 0.5]          # Unknown = gray
        colored[(grid >= 0) & (grid < 50)] = [1, 1, 1]  # Free = white
        colored[grid >= 50] = [0, 0, 0]                 # Occupied = black
        
        # Color palette for frontiers (distinct colors)
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]
        
        # Color each frontier region
        for i, frontier in enumerate(frontiers):
            color = colors[i % len(colors)]
            for row, col in frontier['cells']:
                colored[row, col] = color
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(colored, origin='lower')
        
        # Mark frontier centers with X
        for frontier in frontiers:
            row, col = frontier['center_grid']
            ax.scatter(col, row, c='red', s=300, marker='x', 
                      linewidths=4, zorder=10)
            
            # Add text label
            label = f"F{frontier['id']}\n{frontier['size']}c\n{frontier['distance']:.1f}m"
            ax.text(col+5, row+5, label,
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.9, edgecolor='red', linewidth=2),
                   zorder=11)
        
        # Mark robot position if provided
        if robot_position is not None:
            robot_grid = self.map.world_to_grid(robot_position)
            ax.scatter(robot_grid[1], robot_grid[0], c='lime', s=400, 
                      marker='o', edgecolors='darkgreen', linewidths=3, zorder=12)
            ax.text(robot_grid[1], robot_grid[0]-10, '🤖 Robot',
                   color='darkgreen', fontsize=12, fontweight='bold',
                   ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Title and info
        explored_pct = self.map._get_explored_percentage()
        ax.set_title(
            f'Frontier Detection: {len(frontiers)} reachable frontiers found\n'
            f'Map explored: {explored_pct:.1f}% | Resolution: {self.map.resolution}m',
            fontsize=14, fontweight='bold', pad=20
        )
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Grid X', fontsize=12)
        ax.set_ylabel('Grid Y', fontsize=12)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free Space'),
            Patch(facecolor='black', edgecolor='black', label='Occupied'),
            Patch(facecolor='gray', edgecolor='black', label='Unknown'),
            Patch(facecolor='red', edgecolor='black', label='Frontiers'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved frontier visualization to {save_path}")
        
        plt.show(block=False)
    
    def get_frontier_summary(self, frontiers: List[Dict]) -> str:
        """
        Get text summary of detected frontiers
        
        Args:
            frontiers: List of frontier dicts
            
        Returns:
            Formatted string summary
        """
        
        if len(frontiers) == 0:
            return "No frontiers detected"
        
        summary = f"Detected {len(frontiers)} frontier regions:\n"
        
        for f in frontiers:
            summary += (f"  • Frontier {f['id']}: "
                       f"{f['size']} cells at {f['distance']:.1f}m "
                       f"({f['center_world'][0]:.1f}, {f['center_world'][1]:.1f})\n")
        
        total_cells = sum(f['size'] for f in frontiers)
        avg_distance = np.mean([f['distance'] for f in frontiers])
        
        summary += f"\nTotal frontier cells: {total_cells}\n"
        summary += f"Average distance: {avg_distance:.1f}m\n"
        
        return summary
