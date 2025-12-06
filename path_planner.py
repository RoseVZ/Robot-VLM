#!/usr/bin/env python3
"""Path planning algorithms (A*) with robot footprint"""

import numpy as np
import heapq
from typing import List, Tuple, Optional

class AStarPlanner:
    """A* path planning on occupancy grid with robot footprint consideration"""
    
    def __init__(self, occupancy_map, robot_radius_cells=3):
        """
        Args:
            occupancy_map: OccupancyMapBuilder instance
            robot_radius_cells: Robot radius in grid cells (default=3)
                               For 0.1m resolution: 3 cells = 0.3m radius
                               Husky is ~0.5m radius, so 3 cells is safe
        """
        self.map = occupancy_map
        self.robot_radius = robot_radius_cells
        
        print(f"✓ A* Planner initialized with robot radius = {robot_radius_cells} cells "
              f"({robot_radius_cells * occupancy_map.resolution:.2f}m)")
    
    def plan(self, start_grid: Tuple[int, int], 
             goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding with robot footprint
        Returns: List of (row, col) waypoints or None if no path
        """
        # Convert to standard Python ints
        start_grid = (int(start_grid[0]), int(start_grid[1]))
        goal_grid = (int(goal_grid[0]), int(goal_grid[1]))
        
        print(f"\nPlanning path from {start_grid} to {goal_grid}...")
        print(f"  Robot footprint: {self.robot_radius} cells radius")
        
        # Check if start is free (with footprint)
        if not self._is_position_safe(start_grid):
            print(f"❌ Start position {start_grid} not safe for robot!")
            nearest = self._find_nearest_safe(start_grid)
            if nearest:
                print(f"   Using nearest safe cell: {nearest}")
                start_grid = nearest
            else:
                print("   No nearby safe cells found!")
                return None
        
        # Check if goal is free (with footprint)
        if not self._is_position_safe(goal_grid):
            print(f"❌ Goal position {goal_grid} not safe for robot!")
            nearest = self._find_nearest_safe(goal_grid)
            if nearest:
                print(f"   Using nearest safe cell: {nearest}")
                goal_grid = nearest
            else:
                print("   No nearby safe cells found!")
                return None
        
        # A* search
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            
            if current == goal_grid:
                path = self._reconstruct_path(came_from, current)
                print(f"✓ Path found with {len(path)} waypoints (explored {nodes_explored} nodes)")
                return path
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # 🔥 CHECK FOOTPRINT - Not just single cell!
                if not self._is_position_safe(neighbor):
                    continue
                
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("❌ No path found!")
        return None
    
    def _is_position_safe(self, pos: Tuple[int, int]) -> bool:
        """
        Check if robot can safely be at this position
        Checks all cells within robot's footprint radius
        
        Args:
            pos: (row, col) center position
            
        Returns:
            True if all cells within footprint are free
        """
        row, col = pos
        
        # Check all cells in circular footprint
        for dr in range(-self.robot_radius, self.robot_radius + 1):
            for dc in range(-self.robot_radius, self.robot_radius + 1):
                # Check if within circular radius
                if dr*dr + dc*dc <= self.robot_radius * self.robot_radius:
                    check_row = row + dr
                    check_col = col + dc
                    
                    # Check if cell is in bounds
                    if not (0 <= check_row < self.map.height_cells and 
                           0 <= check_col < self.map.width):
                        return False
                    
                    # Check if cell is free
                    if not self.map.is_free((check_row, check_col)):
                        return False
        
        return True
    
    def _find_nearest_safe(self, pos: Tuple[int, int], 
                          max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        Find nearest safe position for robot (considering footprint)
        
        Args:
            pos: (row, col) starting position
            max_radius: Maximum search radius in cells
            
        Returns:
            Nearest safe position or None
        """
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:  # Check perimeter only
                        candidate = (pos[0] + dr, pos[1] + dc)
                        if self._is_position_safe(candidate):
                            return candidate
        return None
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors"""
        row, col = pos
        neighbors = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < self.map.height_cells and 
                    0 <= new_col < self.map.width):
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance between grid cells"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Heuristic (Euclidean distance to goal)"""
        return self._distance(pos, goal)
    
    def _reconstruct_path(self, came_from: dict, 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from chain"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class AStarPlannerWithSmoothing(AStarPlanner):
    """
    A* planner with path smoothing
    Useful for smoother robot motion
    """
    
    def plan(self, start_grid: Tuple[int, int], 
             goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path and apply smoothing"""
        
        # Get raw A* path
        raw_path = super().plan(start_grid, goal_grid)
        
        if raw_path is None:
            return None
        
        # Apply path smoothing
        smoothed_path = self._smooth_path(raw_path)
        
        print(f"  Path smoothed: {len(raw_path)} → {len(smoothed_path)} waypoints")
        
        return smoothed_path
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints
        Uses line-of-sight checks with footprint
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to connect to farthest visible point
            farthest_idx = current_idx + 1
            
            for test_idx in range(len(path) - 1, current_idx, -1):
                if self._has_line_of_sight(path[current_idx], path[test_idx]):
                    farthest_idx = test_idx
                    break
            
            smoothed.append(path[farthest_idx])
            current_idx = farthest_idx
        
        return smoothed
    
    def _has_line_of_sight(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> bool:
        """
        Check if robot can move in straight line between two positions
        Uses Bresenham's line algorithm with footprint checking
        """
        # Get all cells on line
        line_cells = self._bresenham_line(pos1, pos2)
        
        # Check if all positions on line are safe for robot
        for cell in line_cells:
            if not self._is_position_safe(cell):
                return False
        
        return True
    
    def _bresenham_line(self, start: Tuple[int, int], 
                       end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm"""
        cells = []
        
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        
        return cells