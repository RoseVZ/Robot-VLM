#!/usr/bin/env python3
"""Robot navigation controllers with dynamic replanning"""

import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt

class NavigationControllerRealTime:
    """
    Real-time velocity control for Husky robot with dynamic replanning
    Handles obstacle detection during execution and automatically replans
    """
    
    def __init__(self, env, robot_id):
        self.env = env
        self.robot_id = robot_id
        
        # Husky parameters
        self.max_linear_velocity = 1.0   # m/s
        self.max_angular_velocity = 1.5  # rad/s
        
        # PID gains
        self.kp_linear = 1.5
        self.kp_angular = 3.0
        
        # Thresholds
        self.waypoint_threshold = 0.3
        self.goal_threshold = 0.2
        
        # 🔥 NEW: Replanning parameters
        self.replan_threshold = 5       # Replan if blocked for N waypoints
        self.max_replans = 3            # Max replans per navigation
        self.replan_check_interval = 3  # Check blockage every N waypoints
        
        # Get wheel joints
        self.wheel_joints = self._find_wheel_joints()
        
        print("✓ Real-time navigation controller with dynamic replanning")
        print(f"  Max replans per navigation: {self.max_replans}")
    
    def _find_wheel_joints(self):
        """Find wheel joint indices for Husky"""
        joints = {}
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8').lower()
            
            if 'front_left_wheel' in joint_name or 'left_wheel' in joint_name:
                if 'front' in joint_name:
                    joints['front_left'] = i
                else:
                    joints['rear_left'] = i
            elif 'front_right_wheel' in joint_name or 'right_wheel' in joint_name:
                if 'front' in joint_name:
                    joints['front_right'] = i
                else:
                    joints['rear_right'] = i
        
        if not joints:
            print("Warning: Could not find wheel joints, using fallback indices")
            joints = {'front_left': 2, 'front_right': 3, 'rear_left': 4, 'rear_right': 5}
        
        print(f"  Wheel joints: {joints}")
        return joints
    
    def execute_path(self, waypoints_grid, map_builder, visualize=True, 
                    planner=None, goal_position=None):
        """
        🔥 ENHANCED: Execute path with dynamic replanning
        
        Args:
            waypoints_grid: List of waypoints in grid coordinates
            map_builder: OccupancyMapBuilder for SLAM updates
            visualize: Show live visualization
            planner: AStarPlanner instance (for replanning)
            goal_position: Original goal in grid coords (for replanning)
        
        Returns:
            success: bool
        """
        if not waypoints_grid:
            print("No waypoints!")
            return False
        
        # Convert to world coordinates
        waypoints_world = [map_builder.grid_to_world(wp) for wp in waypoints_grid]
        
        print(f"\n{'='*60}")
        print(f"EXECUTING PATH: {len(waypoints_world)} waypoints")
        print(f"Mode: Real-time Velocity Control + SLAM + Dynamic Replanning")
        print(f"{'='*60}\n")
        
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 10))
        
        trajectory = []
        current_waypoint_idx = 1
        start_time = time.time()
        
        # Counters
        scan_counter = 0
        replan_count = 0
        consecutive_blocked = 0
        
        while current_waypoint_idx < len(waypoints_world):
            # Get current state
            robot_state = self.env.get_robot_state()
            current_pos = np.array(robot_state['position'][:2])
            current_yaw = robot_state['yaw']
            
            trajectory.append(current_pos.copy())
            
            # 🔥 UPDATE MAP PERIODICALLY
            scan_counter += 1
            if scan_counter % 15 == 0:
                if hasattr(map_builder, 'update_map_from_scan'):
                    num_hits = map_builder.update_map_from_scan(
                        robot_state['position'],
                        current_yaw
                    )
                    
                    if scan_counter % 60 == 0:
                        if hasattr(map_builder, '_get_explored_percentage'):
                            explored = map_builder._get_explored_percentage()
                            obstacles = np.sum(map_builder.grid >= 50)
                            print(f"  📡 SLAM update: {explored:.1f}% explored, {obstacles} obstacles")
            
            # 🔥 CHECK IF NEXT WAYPOINT IS BLOCKED
            if current_waypoint_idx % self.replan_check_interval == 0:
                next_waypoint_grid = waypoints_grid[current_waypoint_idx]
                
                if not map_builder.is_free(next_waypoint_grid):
                    consecutive_blocked += 1
                    print(f"  ⚠ Waypoint {current_waypoint_idx} blocked! "
                          f"({consecutive_blocked}/{self.replan_threshold})")
                    
                    # Trigger replanning if blocked for too long
                    if consecutive_blocked >= self.replan_threshold:
                        
                        if replan_count >= self.max_replans:
                            print(f"\n❌ Max replans ({self.max_replans}) reached!")
                            print(f"   Path is blocked, cannot reach goal")
                            if visualize:
                                plt.ioff()
                                plt.close()
                            return False
                        
                        if planner is None or goal_position is None:
                            print(f"\n⚠ Cannot replan: planner or goal not provided")
                            print(f"   Continuing with current path...")
                            consecutive_blocked = 0
                        else:
                            # 🔥 TRIGGER REPLANNING
                            success = self._replan_path(
                                map_builder, planner, goal_position,
                                robot_state['position'], replan_count
                            )
                            
                            if success:
                                # Update path
                                waypoints_grid = success['path']
                                waypoints_world = success['waypoints_world']
                                current_waypoint_idx = 1
                                consecutive_blocked = 0
                                replan_count += 1
                                
                                print(f"  ✓ Replanning successful! New path: {len(waypoints_grid)} waypoints")
                                
                                # Visualize new path
                                if visualize:
                                    self._update_visualization(
                                        ax, map_builder, waypoints_grid,
                                        current_pos, waypoints_world[current_waypoint_idx][:2],
                                        trajectory, current_waypoint_idx, len(waypoints_world)
                                    )
                                
                                continue  # Restart loop with new path
                            else:
                                print(f"  ❌ Replanning failed!")
                                if visualize:
                                    plt.ioff()
                                    plt.close()
                                return False
                else:
                    consecutive_blocked = 0  # Reset if waypoint is free
            
            # Target waypoint
            target_waypoint = np.array(waypoints_world[current_waypoint_idx][:2])
            distance_to_waypoint = np.linalg.norm(target_waypoint - current_pos)
            
            # Check if reached
            threshold = self.goal_threshold if current_waypoint_idx == len(waypoints_world) - 1 else self.waypoint_threshold
            
            if distance_to_waypoint < threshold:
                print(f"✓ Reached waypoint {current_waypoint_idx}/{len(waypoints_world)-1}")
                current_waypoint_idx += 1
                if current_waypoint_idx >= len(waypoints_world):
                    break
                continue
            
            # Calculate velocities
            linear_vel, angular_vel = self._calculate_velocities(
                current_pos, current_yaw, target_waypoint
            )
            
            # Send to wheels
            self._set_wheel_velocities(linear_vel, angular_vel)
            
            # Step simulation
            p.stepSimulation()
            if self.env.gui:
                time.sleep(1./240.)
            
            # Visualize
            if visualize and len(trajectory) % 10 == 0:
                self._update_visualization(
                    ax, map_builder, waypoints_grid,
                    current_pos, target_waypoint, trajectory,
                    current_waypoint_idx, len(waypoints_world)
                )
        
        # Stop robot
        self._set_wheel_velocities(0, 0)
        
        elapsed = time.time() - start_time
        distance = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                      for i in range(len(trajectory)-1))
        
        # 🔥 FINAL 360° SCAN
        print("\n📡 Performing final detailed scan...")
        if hasattr(map_builder, 'update_map_from_scan'):
            robot_state = self.env.get_robot_state()
            for i in range(8):
                angle = robot_state['yaw'] + (i * np.pi / 4)
                map_builder.update_map_from_scan(
                    robot_state['position'],
                    angle
                )
            
            if hasattr(map_builder, '_get_explored_percentage'):
                final_explored = map_builder._get_explored_percentage()
                print(f"✓ Final map coverage: {final_explored:.1f}%")
        
        if visualize:
            self._update_visualization(
                ax, map_builder, waypoints_grid,
                current_pos, target_waypoint, trajectory,
                len(waypoints_world), len(waypoints_world)
            )
            
            plt.ioff()
            print("\n✓ Path complete! (matplotlib window still open)")
            time.sleep(2)
            plt.close()
        
        print(f"\n✓ Navigation complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Distance: {distance:.2f}m")
        print(f"  Avg speed: {distance/elapsed:.2f}m/s")
        if replan_count > 0:
            print(f"  Replans: {replan_count}")
        print()
        
        return True
    
    def _replan_path(self, map_builder, planner, goal_position, 
                    robot_position, replan_count):
        """
        🔥 REPLAN PATH when blocked
        
        Args:
            map_builder: OccupancyMapBuilder
            planner: AStarPlanner instance
            goal_position: Original goal (grid coords)
            robot_position: Current robot position [x,y,z]
            replan_count: Current replan attempt number
            
        Returns:
            dict with 'path' and 'waypoints_world', or None if failed
        """
        
        print(f"\n{'='*60}")
        print(f"🔄 REPLANNING (attempt {replan_count + 1}/{self.max_replans})")
        print(f"{'='*60}")
        
        # Get current position in grid
        current_grid = map_builder.world_to_grid(robot_position)
        
        print(f"  Current position: {current_grid}")
        print(f"  Original goal: {goal_position}")
        
        # Try planning to original goal
        new_path = planner.plan(current_grid, goal_position)
        
        if new_path is not None:
            print(f"  ✓ Found new path to original goal")
            waypoints_world = [map_builder.grid_to_world(wp) for wp in new_path]
            return {
                'path': new_path,
                'waypoints_world': waypoints_world
            }
        
        # Original goal unreachable, try to find alternative
        print(f"  ⚠ Original goal blocked, searching for alternative...")
        
        alternative_goal = self._find_nearest_free_cell(
            goal_position, map_builder, max_radius=15
        )
        
        if alternative_goal is None:
            print(f"  ❌ No alternative goal found")
            return None
        
        print(f"  ✓ Found alternative goal: {alternative_goal}")
        
        # Plan to alternative
        new_path = planner.plan(current_grid, alternative_goal)
        
        if new_path is None:
            print(f"  ❌ Cannot reach alternative goal either")
            return None
        
        print(f"  ✓ Found path to alternative goal")
        
        waypoints_world = [map_builder.grid_to_world(wp) for wp in new_path]
        
        return {
            'path': new_path,
            'waypoints_world': waypoints_world
        }
    
    def _find_nearest_free_cell(self, blocked_goal, map_builder, max_radius=15):
        """
        🔥 Find nearest free cell to blocked goal
        
        Spiral search outward from goal
        
        Args:
            blocked_goal: (row, col) original goal
            map_builder: OccupancyMapBuilder
            max_radius: Maximum search radius in cells
            
        Returns:
            (row, col) or None
        """
        
        center_row, center_col = blocked_goal
        height, width = map_builder.grid.shape
        
        print(f"    Searching for free cell near {blocked_goal} (radius={max_radius})...")
        
        for radius in range(1, max_radius + 1):
            # Check perimeter at this radius
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    # Only check perimeter (not interior)
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    
                    row = center_row + dr
                    col = center_col + dc
                    
                    # Check bounds
                    if not (0 <= row < height and 0 <= col < width):
                        continue
                    
                    # Check if free
                    if map_builder.is_free((row, col)):
                        print(f"    ✓ Found free cell at {(row, col)} (distance={radius})")
                        return (row, col)
        
        print(f"    ❌ No free cell found within radius {max_radius}")
        return None
    
    def _calculate_velocities(self, current_pos, current_yaw, target_pos):
        """PID velocity control"""
        error_vec = target_pos - current_pos
        distance = np.linalg.norm(error_vec)
        
        desired_yaw = np.arctan2(error_vec[1], error_vec[0])
        angle_error = np.arctan2(np.sin(desired_yaw - current_yaw), 
                                np.cos(desired_yaw - current_yaw))
        
        # Linear velocity - slow down if not facing target
        angle_factor = max(0, np.cos(angle_error))
        linear_vel = self.kp_linear * distance * angle_factor
        linear_vel = np.clip(linear_vel, 0, self.max_linear_velocity)
        
        # Angular velocity
        angular_vel = self.kp_angular * angle_error
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, 
                             self.max_angular_velocity)
        
        return linear_vel, angular_vel
    
    def _set_wheel_velocities(self, linear_vel, angular_vel):
        """Convert to wheel velocities and send to motors"""
        wheelbase = 0.5  # Husky wheelbase
        wheel_radius = 0.165  # Husky wheel radius
        
        v_left = linear_vel - (angular_vel * wheelbase / 2)
        v_right = linear_vel + (angular_vel * wheelbase / 2)
        
        omega_left = v_left / wheel_radius
        omega_right = v_right / wheel_radius
        
        # Set all wheels
        for side, omega in [('left', omega_left), ('right', omega_right)]:
            for position in ['front', 'rear']:
                joint_key = f'{position}_{side}'
                if joint_key in self.wheel_joints:
                    p.setJointMotorControl2(
                        self.robot_id, self.wheel_joints[joint_key],
                        p.VELOCITY_CONTROL, 
                        targetVelocity=omega,
                        force=20
                    )
    
    def _update_visualization(self, ax, map_builder, waypoints,
                             current_pos, target_pos, trajectory,
                             waypoint_idx, total_waypoints):
        """Update live visualization with SLAM map"""
        ax.clear()
        
        # Show SLAM map with colors
        if hasattr(map_builder, 'grid'):
            colored_grid = np.zeros((*map_builder.grid.shape, 3))
            
            # Unknown = gray
            unknown_mask = map_builder.grid < 0
            colored_grid[unknown_mask] = [0.5, 0.5, 0.5]
            
            # Free = white
            free_mask = (map_builder.grid >= 0) & (map_builder.grid < 50)
            colored_grid[free_mask] = [1, 1, 1]
            
            # Occupied = black
            occupied_mask = map_builder.grid >= 50
            colored_grid[occupied_mask] = [0, 0, 0]
            
            ax.imshow(colored_grid, origin='lower')
        else:
            ax.imshow(map_builder.grid, cmap='gray_r', origin='lower')
        
        # Planned path
        if waypoints:
            path_array = np.array(waypoints)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-',
                   linewidth=2, alpha=0.3, label='Plan')
        
        # Actual trajectory
        if len(trajectory) > 1:
            traj_grid = [map_builder.world_to_grid(p) for p in trajectory]
            traj_array = np.array(traj_grid)
            ax.plot(traj_array[:, 1], traj_array[:, 0], 'g-',
                   linewidth=2, alpha=0.8, label='Actual')
        
        # Current position
        current_grid = map_builder.world_to_grid(current_pos)
        ax.plot(current_grid[1], current_grid[0], 'go',
               markersize=15, label='Robot', zorder=10)
        
        # Goal
        if waypoints:
            goal = waypoints[-1]
            ax.plot(goal[1], goal[0], 'r*',
                   markersize=25, label='Goal', zorder=5)
        
        progress = waypoint_idx / (total_waypoints - 1) * 100 if total_waypoints > 1 else 100
        
        title = f'Navigation: {progress:.1f}% | WP {waypoint_idx}/{total_waypoints-1}'
        if hasattr(map_builder, '_get_explored_percentage'):
            explored = map_builder._get_explored_percentage()
            title += f'\nMap Explored: {explored:.1f}%'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.pause(0.001)


# Aliases for backward compatibility
NavigationController = NavigationControllerRealTime
NavigationControllerSmooth = NavigationControllerRealTime