#!/usr/bin/env python3
"""Robot navigation controllers - Multiple modes"""

import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt

class NavigationControllerRealTime:
    """
    Real-time velocity control for Husky robot
    Actually drives the robot using wheel velocities
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
        
        # Get wheel joints
        self.wheel_joints = self._find_wheel_joints()
        
        print("✓ Real-time navigation controller initialized")
    
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
            # Fallback: assume standard Husky joint order
            joints = {'front_left': 2, 'front_right': 3, 'rear_left': 4, 'rear_right': 5}
        
        print(f"  Wheel joints: {joints}")
        return joints
    
    def execute_path(self, waypoints_grid, map_builder, visualize=True):
        """Execute path using real-time velocity control"""
        if not waypoints_grid:
            print("No waypoints!")
            return False
        
        waypoints_world = [map_builder.grid_to_world(wp) for wp in waypoints_grid]
        
        print(f"\n{'='*60}")
        print(f"EXECUTING PATH: {len(waypoints_world)} waypoints")
        print(f"Mode: Real-time Velocity Control")
        print(f"{'='*60}\n")
        
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 10))
        
        trajectory = []
        current_waypoint_idx = 1
        start_time = time.time()
        
        while current_waypoint_idx < len(waypoints_world):
            # Get current state
            robot_state = self.env.get_robot_state()
            current_pos = np.array(robot_state['position'][:2])
            current_yaw = robot_state['yaw']
            
            trajectory.append(current_pos.copy())
            
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
        
        # Stop
        self._set_wheel_velocities(0, 0)
        
        elapsed = time.time() - start_time
        distance = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                      for i in range(len(trajectory)-1))
        
        if visualize:
            plt.ioff()
            print("\n✓ Path complete! (matplotlib window still open)")
            time.sleep(2)
            plt.close()
        
        print(f"\n✓ Navigation complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Distance: {distance:.2f}m")
        print(f"  Avg speed: {distance/elapsed:.2f}m/s\n")
        
        return True
    
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
        """Convert to wheel velocities"""
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
        """Update visualization"""
        ax.clear()
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
        
        progress = waypoint_idx / (total_waypoints - 1) * 100
        ax.set_title(f'Navigation: {progress:.1f}% | WP {waypoint_idx}/{total_waypoints-1}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.pause(0.001)


# Aliases for backward compatibility
NavigationController = NavigationControllerRealTime
NavigationControllerSmooth = NavigationControllerRealTime