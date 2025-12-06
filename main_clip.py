import os
import numpy as np
import time
from environment import RobotEnvironment
from occupancy_map import OccupancyMapBuilder
from path_planner import AStarPlanner
from dino_vision_module import GroundingDINO
from navigation_controller import NavigationController, NavigationControllerSmooth

class ConfidenceBasedNavigationSystem:
    """Navigation system with progressive confidence refinement"""
    
    def __init__(self, gui=True, smooth_mode=True):
        print("\n" + "="*60)
        print("PROGRESSIVE CONFIDENCE NAVIGATION SYSTEM")
        print("="*60 + "\n")
        
        self.env = RobotEnvironment(gui=gui)
        self.robot_start_position = [3, 1, 0.2]
        self.smooth_mode = smooth_mode
        
        # Create scene
        self.objects = self.env.create_realistic_scene()
        
        self.map_builder = None
        self.planner = None
        self.clip_module = GroundingDINO()  
        self.controller = None

        self.vlm_verifier = None 
        
        # CONFIDENCE THRESHOLDS
        self.HIGH_CONFIDENCE = 0.80   # Definite match - navigate directly
        self.MID_CONFIDENCE = 0.60    # Likely match - approach & verify
        self.LOW_CONFIDENCE = 0.40    # Weak signal - local exploration
        
        os.makedirs('maps', exist_ok=True)
        
        print("✓ Progressive confidence system initialized!")
        print(f"  Thresholds: HIGH={self.HIGH_CONFIDENCE}, MID={self.MID_CONFIDENCE}, LOW={self.LOW_CONFIDENCE}\n")
    
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
        self.map_builder.perform_initial_scan(
            robot_state['position'],
            robot_state['yaw']
        )
        
        start_grid = self.map_builder.world_to_grid(robot_state['position'])
        
        if not self.map_builder.is_free(start_grid):
            print("⚠ Clearing robot area...")
            self._clear_area(start_grid, radius=3)
        else:
            print("✓ Robot position is FREE")

        self.map_builder.inflate_obstacles(radius_cells=inflate_radius)
        robot_radius_meters = 0.5  # Husky robot radius
        robot_radius_cells = int(robot_radius_meters / resolution)
        self.planner = AStarPlanner(self.map_builder, robot_radius_cells=robot_radius_cells)
        
        if self.smooth_mode:
            self.controller = NavigationControllerSmooth(self.env, self.env.robot_id)
        else:
            self.controller = NavigationController(self.env, self.env.robot_id)
        
        print(f"\n✓ SLAM system ready!")
        print(f"  Map coverage: {self.map_builder._get_explored_percentage():.1f}%")
    
    def _clear_area(self, center_grid, radius=3):
        """Clear circular area"""
        row, col = center_grid
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr*dr + dc*dc <= radius*radius:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.map_builder.height_cells and 0 <= c < self.map_builder.width:
                        self.map_builder.grid[r, c] = 0
    
    def navigate_with_progressive_confidence(self, instruction: str, 
                                        visualize=True, 
                                        max_refinement_attempts=3):
        """
            PROGRESSIVE CONFIDENCE NAVIGATION WITH DYNAMIC THRESHOLDS
        
        Strategy:
        - Attempt 1: Only explore if confidence < 0.60 (LOW)
        - Attempt 2+: Explore if confidence < 0.80 (LOW or MID)
        
        Gets stricter with each attempt!
        """
        
        if self.map_builder is None:
            print("       Map not built!")
            return False
        
        print(f"\n{'='*60}")
        print(f"QUERY: {instruction}")
        print(f"{'='*60}")
        
        # Track refinement attempts
        refinement_count = 0
        
        while refinement_count <= max_refinement_attempts:
            
            # STEP 1: Detect from current position
            print(f"\n{'─'*60}")
            print(f"DETECTION ATTEMPT {refinement_count + 1}")
            print(f"{'─'*60}")
            
            decision, best_view, best_score = self.clip_module.search_current_room(
                instruction, self.env, self.env.robot_id
            )
            
            print(f"\nDetection Result:")
            print(f"   Confidence: {best_score:.3f}")
            print(f"   Decision: {decision}")
            
            #    DYNAMIC THRESHOLDS BASED ON ATTEMPT NUMBER
            
            if refinement_count == 0:
                # ATTEMPT 1: Be lenient, only explore on LOW
                exploration_threshold = self.MID_CONFIDENCE  # 0.60
                print(f"   Threshold: First attempt - explore if < {exploration_threshold:.2f}")
            else:
                # ATTEMPT 2+: Be strict, explore on MID or LOW
                exploration_threshold = self.HIGH_CONFIDENCE  # 0.80
                print(f"   Threshold: Later attempt - explore if < {exploration_threshold:.2f}")
            
            # STEP 2: Decide action based on confidence
            
            # ═══════════════════════════════════════
            # CASE A: HIGH CONFIDENCE (≥0.80)
            # ═══════════════════════════════════════
            if best_score >= self.HIGH_CONFIDENCE:
                print(f"\n✓ HIGH CONFIDENCE ({best_score:.3f}) - Navigating directly to goal!")
                
                goal_position = self.clip_module.navigate_to_detected_object(
                    instruction, self.env, self.env.robot_id, 
                    self.map_builder, best_view, best_score
                )
                
                success = self._execute_navigation(goal_position, visualize, 
                                                path_name='final_path.png')
                
                if success:
                    print(f"\n🎯 SUCCESS: Reached object with HIGH confidence!")
                    return True
                else:
                    print(f"\n⚠ Navigation failed, retrying detection...")
                    refinement_count += 1
                    continue
            
            # ═══════════════════════════════════════
            # CASE B: BELOW EXPLORATION THRESHOLD
            # ═══════════════════════════════════════
            elif best_score < exploration_threshold:
                print(f"\n{'='*60}")
                print("TRIGGERING EXPLORATION")
                print(f"{'='*60}")
                
                # Initialize VLM verifier
                if self.vlm_verifier is None:
                    try:
                        from exploration.vlm_verifier import VLMRoomVerifier
                        self.vlm_verifier = VLMRoomVerifier()
                    except Exception as e:
                        print(f"       VLM verifier failed: {e}")
                        return False
                
                # Initialize frontier detector
                from exploration.frontier_detector import FrontierDetector
                detector = FrontierDetector(self.map_builder, min_frontier_size=5)
                
                #     NEW: Track visited cells for this query
                if not hasattr(self, '_visited_cells'):
                    self._visited_cells = {}  # query -> set of cells
                
                if instruction not in self._visited_cells:
                    self._visited_cells[instruction] = set()
                
                #     NEW: Multi-frontier loop
                max_frontier_attempts = 5
                frontier_attempt = 0
                
                while frontier_attempt < max_frontier_attempts:
                    
                    print(f"\n{'─'*60}")
                    print(f"FRONTIER ATTEMPT {frontier_attempt + 1}/{max_frontier_attempts}")
                    print(f"{'─'*60}")
                    
                    # STEP 1: Detect frontiers
                    frontiers = detector.detect_frontiers(
                        robot_position=self.env.get_robot_state()['position']
                    )
                    
                    if len(frontiers) == 0:
                        print("\n       No frontiers detected!")
                        return False
                    
                    # Visualize
                    detector.visualize_frontiers(
                        frontiers,
                        robot_position=self.env.get_robot_state()['position'],
                        save_path=f'maps/frontiers_attempt_{frontier_attempt+1}.png'
                    )
                    
                    # STEP 2: Filter to unvisited cells
                    unvisited_frontiers = []
                    
                    for frontier in frontiers:
                        # Convert position to cell
                        x, y = frontier['center_world'][:2]
                        cell = (int(x / 2.0), int(y / 2.0))  # 2m cell size
                        
                        # Check if visited
                        if cell not in self._visited_cells[instruction]:
                            unvisited_frontiers.append(frontier)
                        else:
                            print(f"  🚫 Skipping frontier {frontier['id']} at cell {cell} - already visited")
                    
                    if len(unvisited_frontiers) == 0:
                        print(f"\n       No unvisited frontiers for '{instruction}'")
                        print(f"   Visited {len(self._visited_cells[instruction])} cells")
                        return False
                    
                    print(f"\n✓ {len(unvisited_frontiers)} unvisited frontiers")
                    
                    # STEP 3: Pick nearest unvisited
                    next_frontier = unvisited_frontiers[0]  # Already sorted by distance
                    
                    print(f"\n🎯 Trying frontier {next_frontier['id']}:")
                    print(f"   Position: ({next_frontier['center_world'][0]:.1f}, "
                        f"{next_frontier['center_world'][1]:.1f})")
                    print(f"   Distance: {next_frontier['distance']:.1f}m")
                    print(f"   Size: {next_frontier['size']} cells")
                    
                    # STEP 4: Navigate to frontier
                    success = self._execute_navigation(
                        next_frontier['center_world'],
                        visualize=True,
                        path_name=f'frontier_attempt_{frontier_attempt+1}.png'
                    )
                    
                    if not success:
                        print(f"\n       Navigation failed!")
                        frontier_attempt += 1
                        continue
                    
                    print(f"\n✓ Reached frontier!")
                    
                    # STEP 5: Mark cell as visited
                    x, y = next_frontier['center_world'][:2]
                    cell = (int(x / 2.0), int(y / 2.0))
                    self._visited_cells[instruction].add(cell)
                    print(f"  📍 Marked cell {cell} as visited")
                    
                    # STEP 6: VLM verification
                    print(f"\n{'='*60}")
                    print(f"POST-FRONTIER SEMANTIC VERIFICATION")
                    print(f"{'='*60}")
                    
                    verification = self.vlm_verifier.verify_room(
                        target_object=instruction,
                        env=self.env,
                        robot_id=self.env.robot_id
                    )
                    
                    # STEP 7: Decision
                    if verification['is_promising']:
                        # ✅ Right room!
                        print(f"\n{'='*60}")
                        print(f"✅ ROOM VERIFICATION PASSED")
                        print(f"{'='*60}")
                        print(f"  Room: {verification['room_type']}")
                        print(f"  VLM: {verification['reasoning']}")
                        
                        # Re-detect object
                        print(f"\n🔄 Re-detecting from promising room...")
                        time.sleep(1)
                        refinement_count += 1
                        continue  #     LOOP BACK TO OUTER LOOP (detection)
                    
                    else:
                        #        Wrong room - try next frontier
                        print(f"\n{'='*60}")
                        print(f"       ROOM VERIFICATION FAILED")
                        print(f"{'='*60}")
                        print(f"  Room: {verification['room_type']}")
                        print(f"  VLM: {verification['reasoning']}")
                        print(f"\n🔄 Trying next frontier...")
                        
                        frontier_attempt += 1
                        #     LOOP CONTINUES TO NEXT FRONTIER
                
                # Exhausted all attempts
                print(f"\n       Tried {max_frontier_attempts} frontiers, none promising")
                return False
            
            # ═══════════════════════════════════════
            # CASE C: ABOVE THRESHOLD BUT NOT HIGH
            # (Only on first attempt - MID confidence 0.60-0.79)
            # ═══════════════════════════════════════
            else:
                # This only happens on first attempt with MID confidence
                print(f"\n⚠ MID CONFIDENCE ({best_score:.3f}) - Approaching for verification...")
                
                # Get detected object position
                detected_position = self.clip_module.navigate_to_detected_object(
                    instruction, self.env, self.env.robot_id,
                    self.map_builder, best_view, best_score
                )
                
                # Go halfway to get closer
                halfway_position = self._compute_halfway_point(
                    self.env.get_robot_state()['position'],
                    detected_position
                )
                
                print(f"\n📍 Moving to verification position (halfway)...")
                print(f"   Current: {self.env.get_robot_state()['position'][:2]}")
                print(f"   Detected at: {detected_position[:2]}")
                print(f"   Halfway point: {halfway_position[:2]}")
                
                # Navigate halfway
                success = self._execute_navigation(halfway_position, visualize,
                                                path_name=f'approach_{refinement_count}.png')
                
                if not success:
                    print(f"\n       Could not reach verification point")
                    return False
                
                # RE-DETECT from new position
                print(f"\n🔄 Re-detecting from closer position...")
                time.sleep(1)  # Brief pause for stability
                
                refinement_count += 1
                # Loop continues - will re-detect from new position
                continue
        
        # Max refinement attempts reached
        print(f"\n⚠ Max refinement attempts ({max_refinement_attempts}) reached")
        print(f"   Final confidence: {best_score:.3f}")
        print(f"   Triggering exploration...")
        return False
    
    def _compute_halfway_point(self, start_pos, goal_pos):
        """
        Compute halfway point between start and goal
        
        Args:
            start_pos: [x, y, z] starting position
            goal_pos: [x, y, z] goal position
        
        Returns:
            halfway_pos: [x, y, z] halfway point
        """
        start = np.array(start_pos)
        goal = np.array(goal_pos)
        
        # Compute midpoint
        halfway = (start + goal) / 2.0
        
        # Keep z coordinate same as start (stay on ground)
        halfway[2] = start_pos[2]
        
        return halfway
    
    def _execute_navigation(self, goal_position, visualize=True, path_name='path.png'):
        """
        Execute navigation to goal position with dynamic replanning support
        
        Args:
            goal_position: [x, y, z] target position
            visualize: Show visualization
            path_name: Name for saved path image
        
        Returns:
            success: bool
        """
        robot_state = self.env.get_robot_state()
        start_grid = self.map_builder.world_to_grid(robot_state['position'])
        goal_grid = self.map_builder.world_to_grid(goal_position)
        
        # Clear goal if needed
        if not self.map_builder.is_free(goal_grid):
            print("   ⚠ Goal occupied, clearing...")
            self._clear_area(goal_grid, radius=3)
        
        # Plan initial path
        path = self.planner.plan(start_grid, goal_grid)
        if path is None:
            print("          No path found!")
            return False
        
        print(f"   ✓ Path planned: {len(path)} waypoints")
        
        # Visualize
        if visualize:
            self.map_builder.visualize(path=path, start=start_grid, goal=goal_grid,
                                    save_path=f'maps/{path_name}')
        
        #     EXECUTE WITH REPLANNING SUPPORT
        success = self.controller.execute_path(
            path, 
            self.map_builder, 
            visualize=visualize,
            planner=self.planner,        #     Pass planner for replanning
            goal_position=goal_grid      #     Pass goal for replanning
        )
        
        return success
    
    # ═══════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ═══════════════════════════════════════
    
    def navigate_with_confidence(self, instruction: str, visualize=True):
        """
        Wrapper for backward compatibility
        Calls new progressive confidence method
        """
        return self.navigate_with_progressive_confidence(
            instruction, 
            visualize=visualize,
            max_refinement_attempts=3
        )
    
    def demo(self):
        """Demo with different confidence scenarios"""
        
        test_cases = [
            "find the sofa",       # Should have high confidence
            "find the bed",        # Should have mid confidence (different room)
            "find the fridge",     # Should have mid confidence (different room)
            "find the purple block",  # Should fail (doesn't exist)
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
                print(f"\n       Could not complete navigation")
            
            input("\nPress Enter for next test...")
    
    def interactive_mode(self):
        """Interactive mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE - PROGRESSIVE CONFIDENCE")
        print("="*60)
        print("\nAvailable objects in scene:")
        for obj_name in self.objects.keys():
            if not obj_name.startswith('wall') and 'floor' not in obj_name:
                print(f"  - {obj_name}")
        print("\nHow it works:")
        print("  HIGH confidence (>0.80): Navigate directly")
        print("  MID confidence (0.60-0.80): Approach halfway, re-detect, verify")
        print("  LOW confidence (<0.60): Exploration mode")
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
    print("\n" + "🤖 "*20)
    print("PROGRESSIVE CONFIDENCE NAVIGATION")
    print("🤖 "*20 + "\n")
    
    print("Strategy:")
    print("  • HIGH confidence (>0.80): Go directly to goal")
    print("  • MID confidence (0.60-0.80): Approach, re-detect, verify")
    print("  • LOW confidence (<0.60): Trigger exploration")
    
    system = ConfidenceBasedNavigationSystem(gui=True, smooth_mode=True)
    system.build_map(resolution=0.1, inflate_radius=1)
    
    print("\n" + "="*60)
    print("CHOOSE MODE")
    print("="*60)
    print("1. Demo (test different confidence levels)")
    print("2. Interactive (try your own queries)")
    print("3. Single test (find sofa)")
    print("="*60)
    
    choice = input("\nEnter 1, 2, or 3 [default: 2]: ").strip() or "2"
    
    if choice == "1":
        system.demo()
    elif choice == "2":
        system.interactive_mode()
    else:
        system.navigate_with_confidence("find the sofa", visualize=True)
    
    system.close()
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()