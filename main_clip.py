#!/usr/bin/env python3


import os
import numpy as np
import time
from environment import RobotEnvironment
from occupancy_map import OccupancyMapBuilder
from path_planner import AStarPlanner
from dino_vision_module import GroundingDINO
from navigation_controller import NavigationController, NavigationControllerSmooth

class ConfidenceBasedNavigationSystem:
    """Navigation system with confidence-based exploration + 3D localization"""
    
    def __init__(self, gui=True, smooth_mode=True):
        print("\n" + "="*60)
        print("NAVIGATION SYSTEM WITH 3D OBJECT LOCALIZATION")
        print("="*60 + "\n")
        
        self.env = RobotEnvironment(gui=gui)
        self.robot_start_position = [-3, -3, 0.2]
        self.smooth_mode = smooth_mode
        
        # Create scene no robot first
        self.objects = self.env.create_realistic_scene()
        # self.objects = self.env.create_colorful_scene()
        
        self.map_builder = None
        self.planner = None
        self.clip_module = GroundingDINO()  
        self.controller = None
        
        os.makedirs('maps', exist_ok=True)
        
        print("\n✓ System initialized with 3D localization!\n")
    
    def build_map(self, resolution=0.1, inflate_radius=1):
        """Build occupancy map"""
        print("="*60)
        print("BUILDING MAP")
        print("="*60)
        
        self.map_builder = OccupancyMapBuilder(self.env.bounds, resolution=resolution)
        self.map_builder.build_from_raycasting()
        self.map_builder.inflate_obstacles(radius_cells=inflate_radius)
        
        print("\nLoading robot...")
        self.env.load_robot(position=self.robot_start_position)
        
        robot_state = self.env.get_robot_state()
        start_grid = self.map_builder.world_to_grid(robot_state['position'])
        
        if not self.map_builder.is_free(start_grid):
            print("⚠ Clearing robot area...")
            self._clear_area(start_grid, radius=3)
        else:
            print("✓ Robot position is FREE")
        
        self.planner = AStarPlanner(self.map_builder)
        
        if self.smooth_mode:
            self.controller = NavigationControllerSmooth(self.env, self.env.robot_id)
        else:
            self.controller = NavigationController(self.env, self.env.robot_id)
        
        print("")
    
    def _clear_area(self, center_grid, radius=3):
        """Clear circular area"""
        row, col = center_grid
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr*dr + dc*dc <= radius*radius:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.map_builder.height_cells and 0 <= c < self.map_builder.width:
                        self.map_builder.grid[r, c] = 0
    
    def navigate_with_confidence(self, instruction: str, visualize=True):
        """
        Main navigation with confidence-based decision making + 3D localization
        
        Flow:
        1. CLIP search current room (now with depth + segmentation)
        2. Check confidence score
        3. If HIGH/MEDIUM: Navigate to object (using TRUE 3D position!)
        4. If LOW/VERY_LOW: Explore other rooms (placeholder for now)
        """
        
        if self.map_builder is None:
            print(" Map not built!")
            return False
        
        # PHASE 1: Search current room with CLIP + 3D localization
        decision, best_view, best_score = self.clip_module.search_current_room(
            instruction, self.env, self.env.robot_id
        )
        
        # PHASE 2: Act based on confidence
        if decision == "NAVIGATE":
            print(f"\n{'='*60}")
            print("ACTION: Navigating to detected object")
            print(f"{'='*60}")
            
            # Get goal position (now uses TRUE 3D position if available!)
            goal_position = self.clip_module.navigate_to_detected_object(
                instruction, self.env, self.env.robot_id, 
                self.map_builder, best_view, best_score
            )
            
            # Plan path
            robot_state = self.env.get_robot_state()
            start_grid = self.map_builder.world_to_grid(robot_state['position'])
            goal_grid = self.map_builder.world_to_grid(goal_position)
            
            if not self.map_builder.is_free(goal_grid):
                print("⚠ Goal occupied, clearing...")
                self._clear_area(goal_grid, radius=3)
            
            path = self.planner.plan(start_grid, goal_grid)
            if path is None:
                print("❌ No path found!")
                return False
            
            # Visualize
            self.map_builder.visualize(path=path, start=start_grid, goal=goal_grid,
                                      save_path='maps/final_path.png')
            
            # Execute
            print("\n Executing navigation...")
            time.sleep(1)
            success = self.controller.execute_path(path, self.map_builder, visualize=visualize)
            
            return success
            
        elif decision == "EXPLORE_NEARBY":
            print(f"\n{'='*60}")
            print("ACTION: Exploring nearby areas")
            print(f"{'='*60}")
            print("⚠ Moving closer to investigate...")
            
            # For now, try to navigate toward the weak signal
            goal_position = self.clip_module.navigate_to_detected_object(
                instruction, self.env, self.env.robot_id,
                self.map_builder, best_view, best_score
            )
            
            robot_state = self.env.get_robot_state()
            start_grid = self.map_builder.world_to_grid(robot_state['position'])
            goal_grid = self.map_builder.world_to_grid(goal_position)
            
            if not self.map_builder.is_free(goal_grid):
                self._clear_area(goal_grid, radius=3)
            
            path = self.planner.plan(start_grid, goal_grid)
            if path:
                self.map_builder.visualize(path=path, start=start_grid, goal=goal_grid,
                                          save_path='maps/explore_path.png')
                success = self.controller.execute_path(path, self.map_builder, visualize=visualize)
                return success
            
            return False
            
        else:  # EXPLORE_OTHER_ROOMS
            print(f"\n{'='*60}")
            print("ACTION: Multi-room exploration needed")
            print(f"{'='*60}")
            print("⚠ Object not in current room")
            print("📍 Next step: Implement frontier detection and multi-room exploration")
            
            return False
    
    def demo(self):
        """Demo with different confidence scenarios"""
        
        test_cases = [
            "find the red block", 
            "find the blue block",
            "find the purple block",  
        ]
        
        for instruction in test_cases:
            print("\n" + "="*60)
            print(f"TEST: {instruction}")
            print("="*60)
            
            input("\nPress Enter to start search...")
            
            success = self.navigate_with_confidence(instruction, visualize=True)
            
            if success:
                print(f"\n✓ SUCCESS: Found and navigated to object")
            else:
                print(f"\nCould not complete navigation")
            
            input("\nPress Enter for next test...")
    
    def interactive_mode(self):
        """Interactive mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE - 3D LOCALIZATION")
        print("="*60)
        print("\nAvailable objects in scene:")
        for obj_name in self.objects.keys():
            if not obj_name.startswith('wall'):
                print(f"  - {obj_name}")
        print("\nTry queries like:")
        print("  - 'find the red block'")
        print("  - 'go to the blue block'")
        print("  - 'find the purple block' (doesn't exist)")
        print("\nType 'quit' to exit\n")
        
        while True:
            instruction = input("Query: ").strip()
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            if instruction:
                self.navigate_with_confidence(instruction, visualize=True)
    
    def close(self):
        self.env.close()


def main():
    print("\n" + "-"*20)
    print("NAVIGATION WITH 3D OBJECT LOCALIZATION")
    print("- "*20 + "\n")
    
    print("   • CLIP finds object direction")
    print("   • Segmentation finds object pixels")
    print("   • Depth measures distance")
    print("   • 3D projection gives world coordinates")
    
    system = ConfidenceBasedNavigationSystem(gui=True, smooth_mode=True)
    system.build_map(resolution=0.1, inflate_radius=1)
    
    print("\n" + "="*60)
    print("CHOOSE MODE")
    print("="*60)
    print("1. Demo (test different confidence levels)")
    print("2. Interactive (try your own queries)")
    print("3. Single test (find red block)")
    print("="*60)
    
    choice = input("\nEnter 1, 2, or 3 [default: 1]: ").strip() or "1"
    
    if choice == "1":
        system.demo()
    elif choice == "2":
        system.interactive_mode()
    else:
        system.navigate_with_confidence("find the red block", visualize=True)
    
    system.close()
    print("\nDone!")


if __name__ == "__main__":
    main()