#!/usr/bin/env python3
"""Path planning algorithms (A*)"""

import numpy as np
import heapq
from typing import List, Tuple, Optional

class AStarPlanner:
    """A* path planning on occupancy grid"""
    
    def __init__(self, occupancy_map):
        self.map = occupancy_map
        
    def plan(self, start_grid: Tuple[int, int], 
             goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding
        Returns: List of (row, col) waypoints or None if no path
        """
        # Convert to standard Python ints to avoid numpy types
        start_grid = (int(start_grid[0]), int(start_grid[1]))
        goal_grid = (int(goal_grid[0]), int(goal_grid[1]))
        
        print(f"\nPlanning path from {start_grid} to {goal_grid}...")
        
        # Check if start is free
        if not self.map.is_free(start_grid):
            print(f"❌ Start position {start_grid} is occupied!")
            print(f"   Grid value at start: {self.map.grid[start_grid[0], start_grid[1]]}")
            # Try to find nearest free cell
            nearest = self._find_nearest_free(start_grid)
            if nearest:
                print(f"   Using nearest free cell: {nearest}")
                start_grid = nearest
            else:
                print("   No nearby free cells found!")
                return None
        
        # Check if goal is free
        if not self.map.is_free(goal_grid):
            print(f"❌ Goal position {goal_grid} is occupied!")
            print(f"   Grid value at goal: {self.map.grid[goal_grid[0], goal_grid[1]]}")
            # Try to find nearest free cell
            nearest = self._find_nearest_free(goal_grid)
            if nearest:
                print(f"   Using nearest free cell: {nearest}")
                goal_grid = nearest
            else:
                print("   No nearby free cells found!")
                return None
        
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
                
                if not self.map.is_free(neighbor):
                    continue
                
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("❌ No path found!")
        return None
    
    def _find_nearest_free(self, pos: Tuple[int, int], max_radius: int = 5) -> Optional[Tuple[int, int]]:
        """Find nearest free cell within max_radius"""
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:  # Check perimeter only
                        candidate = (pos[0] + dr, pos[1] + dc)
                        if self.map.is_free(candidate):
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
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from chain"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
