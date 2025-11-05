import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Vector3
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from typing import Dict, Tuple
import math
from functools import partial

# importing helper functions
from project1.utils.helper_functions import *


class ExtendedRules(Node):

    def __init__(self):
        super().__init__('extended_rules')

        # Declaring parameters to ros
        self.declare_parameter('mass', 1.0)
        self.declare_parameter('max_acc', 3.0)
        self.declare_parameter('max_speed', 0.9)
        self.declare_parameter('min_dist', 0.3)
        self.declare_parameter('min_dist_to_obst', 2.0)
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('neigh_radius', 5.0)
        self.declare_parameter('fov_deg', 360)
        self.declare_parameter('w_separation', 0.5)
        self.declare_parameter('w_alignment', 0.6)
        self.declare_parameter('w_cohesion', 1.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('target_x', -4.0)
        self.declare_parameter('target_y', 4.0)
        self.declare_parameter('slowing_distance', 1.0)
        self.declare_parameter('w_flocking', 0.5)
        self.declare_parameter('w_navigation', 0.8)
        self.declare_parameter('w_obst_avoid', 1.0)
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('robot_radius', 0.2)
        self.declare_parameter('order', "123")

        # Load Parameters
        self.mass = self.get_parameter('mass').value
        self.max_acc = self.get_parameter('max_acc').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_dist = self.get_parameter('min_dist').value
        self.min_dist_to_obst = self.get_parameter('min_dist_to_obst').value
        self.max_force = self.mass * self.max_acc       # Computing the maximum force with mass and acc parameters
        self.num_robots = self.get_parameter('num_robots').value
        self.neigh_radius = self.get_parameter('neigh_radius').value
        self.fov = math.radians(self.get_parameter('fov_deg').value)
        self.w_separation = self.get_parameter('w_separation').value
        self.w_alignment = self.get_parameter('w_alignment').value
        self.w_cohesion = self.get_parameter('w_cohesion').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.target = np.array([[self.get_parameter('target_x').value],
                                [self.get_parameter('target_y').value]])
        self.slowing_distance = self.get_parameter('slowing_distance').value
        self.w_flocking = self.get_parameter('w_flocking').value
        self.w_navigation = self.get_parameter('w_navigation').value
        self.w_obst_avoid = self.get_parameter('w_obst_avoid').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.order = self.get_parameter('order').value

        # ATTRIBUTES
        self.gridmap = None
        self.time_step = (1.0/self.publish_rate)
        self.priority: Dict[int, np.ndarray] = {}   # Priority order for behaviours: flocking, navigation, obstacle avoidance
        for i in range(len(self.order)): # initializing desired velocity vectors for all behaviors
            self.priority[i + 1] = np.zeros((2,1))
        self.robots: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}  # Store robot states: {i: (pos[2x1], vel[2x1])}
        for i in range(self.num_robots):
            self.robots[i] = (np.zeros((2, 1)), np.zeros((2, 1)), 0.0)

        # PUBLISHERS & SUBSCRIBERS 
        self.robot_publishers = {}
        for i in range(self.num_robots):
            topic_cmd = f'/robot_{i}/cmd_vel'
            topic_odom = f'/robot_{i}/odom'
            self.robot_publishers[i] = self.create_publisher(Twist, topic_cmd, 10)
            self.create_subscription(Odometry, topic_odom, partial(self.odom_cb, i), 10)
        qos = QoSProfile( # because the map is published latched
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1
            )
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, qos)

        # TIMERS
        self.connection_timer = self.create_timer(5, self.connect_wait) # wait to allow publisher/subscriber connection
    
    # Startup timer callback function
    def connect_wait(self):
        self.destroy_timer(self.connection_timer) # stopping the 2-second wait
        self.timer = self.create_timer(1.0 / self.publish_rate, self.control_loop)
        self.get_logger().info(f"Basic Rules node started with {self.num_robots} robots")

    # Robots pose callback 
    def odom_cb(self, idx: int, msg: Odometry):
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        pos = np.array([[px], [py]])
        vel = np.array([[vx], [vy]])
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.robots[idx] = (pos, vel, yaw)

    # Occupancy gridmap callback
    def map_cb(self, gridmap: OccupancyGrid):
        self.map_width = gridmap.info.width
        self.map_height = gridmap.info.height
        self.resolution = gridmap.info.resolution
        self.origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
        self.gridmap = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T

    # ------------------------------
    # Utility: check if neighbor j is within radius and FOV of robot i
    # ------------------------------
    def in_fov(self, pos_i: np.ndarray, vel_i: np.ndarray, pos_j: np.ndarray) -> bool:
        delta = vec_sub(pos_j, pos_i)
        dist = vec_len(delta)
        if (dist > self.neigh_radius) or (dist == 0.0):
            return False

        # --- If robot i has zero (or near-zero) velocity ---
        vlen = vec_len(vel_i)
        if vlen < 1e-6:
            return True # If robot i is stationary, we assume it can "see" in all directions (full 360° FOV)

        heading = vec_norm(vel_i) # Unit vector in direction of i's motion
        delta_dir = vec_norm(delta) # Unit vector from i to j

        cos_angle = float(np.dot(heading.T, delta_dir)) # Compute cosine of the angle between heading and delta
        cos_angle = max(-1.0, min(1.0, cos_angle))  # clip

        angle = math.acos(cos_angle) # computing the angle between robot i and j (result in radiance, range [0, π])
        # self.get_logger().info(f"{pos_i} has angle {angle} with {pos_j}")
        # self.get_logger().info(self.fov)

        return abs(angle) <= self.fov / 2.0 # Check if neighbor is within half of the FOV angle to be visible

    # Main flocking (+ navigation and obstacle avoidance) control loop
    def control_loop(self):
        states = self.robots.copy()
        for i in range(self.num_robots):
            # self.get_logger().info(f"current robot: {i}")
            #--------------
            # BASIC RULES COMPONENT
            #--------------
            pos_i, vel_i, yaw = states[i]
            neighbors = []

            for j in range(self.num_robots):
                if i == j:
                    continue
                pos_j, _, _ = states[j]
                if self.in_fov(pos_i, vel_i, pos_j):
                    neighbors.append(j)

            # Default: zero behavior vectors
            sep = np.zeros((2, 1))
            align = np.zeros((2, 1))
            coh = np.zeros((2, 1))

            if len(neighbors) > 0:
                # self.get_logger().info("Neighbours detected!!")
                # --- Separation ---
                for j in neighbors:
                    pos_j, _, _ = states[j]
                    diff = vec_sub(pos_i, pos_j)
                    dist = max(vec_len(diff), 1e-6) # to prevent division by zero
                    if dist < self.min_dist:  # only compute separation velocity if dist is less than specified minimum distance 
                        sep += vec_scale(vec_norm(diff), 1.0 / dist)  #scale down difference based on distance
                sep /= len(neighbors)
                sep = sep / (1.0/self.publish_rate)  # convert to velocity unit

                # --- Alignment ---
                avg_v = np.zeros((2, 1))
                for j in neighbors:
                    _, vel_j, _ = states[j]
                    avg_v += vel_j
                avg_v /= len(neighbors)
                align = vec_sub(avg_v, vel_i)

                # --- Cohesion ---
                com = np.zeros((2, 1))
                for j in neighbors:
                    pos_j, _, _ = states[j]
                    com += pos_j
                com /= len(neighbors)
                coh = vec_sub(com, pos_i)
                coh = coh / (1.0/self.publish_rate)  # convert to velocity unit

            # Weighted average of behaviors
            steer_basic = ((self.w_separation * sep) + (self.w_alignment * align) + (self.w_cohesion * coh)) / 3

            # Compute desired velocity and scale by weight
            desired_basic = vec_add(vel_i, steer_basic)
            desired_basic = (vec_norm(desired_basic)) * self.w_flocking
            if i != 0: # 'leader' isn't flocking (let the first robot be the 'leader')
                self.priority[3] = desired_basic      # stored as the third by default

            #--------------
            # NAVIGATION RULE COMPONENT
            #--------------
            target_offset = vec_sub(self.target, pos_i)
            target_dist = vec_len(target_offset)

            # Arrival model (Reynold's): Scale speed linearly with distance until within slowing zone
            ramped_speed = (self.max_speed * self.w_navigation) * (target_dist / self.slowing_distance)
            clipped_speed = min(ramped_speed, self.max_speed)

            if target_dist > 1e-2:
                desired_nav = vec_scale((target_offset), clipped_speed/target_dist)
            else: 
                desired_nav = np.zeros((2,1))
                
            desired_nav = (vec_norm(desired_nav)) * self.w_navigation
            if i == 0: # let the first robot be the 'leader'
                self.priority[2] = desired_nav      # stored as the second behavior in the dict by default


            #--------------
            # OBSTACLE AVOIDANCE RULE COMPONENT
            #--------------
            vlen = vec_len(vel_i)
            if vlen < 1e-6:
                desired_obst_avoid = np.zeros((2,1))    # skip stationary robots
            else:
                curr_heading = vec_norm(vel_i) # using vectorial heading here, to keep everything 'clean'
                # curr_heading = np.array([[math.cos(yaw)], [math.sin(yaw)]])
                # lookahead_vec = vec_scale(curr_heading, self.lookahead_distance)
                # lookahead_point = vec_add(pos_i, lookahead_vec)

                # Find obstacles near the lookahead path
                steer_avoid = self.find_avoidance_vector(pos_i, curr_heading)

                # compute resulting obstacle avoidance steering vector
                desired_obst_avoid = vec_norm(steer_avoid / (1.0/self.publish_rate)) * self.w_obst_avoid

            self.priority[1] = desired_obst_avoid       # stored as the first behavior in the dict by default
            # self.get_logger().info(f"Robot {i} with obstacle avoidance: {desired_obst_avoid}")

            #--------------
            # TOTAL CONTROL VELOCITY (PAA for velocity)
            #--------------
            total_vel = np.zeros((2,1))
            acc = np.copy(total_vel)    # accumulator

            for y in range(len(self.order)):
                acc += self.priority[int(self.order[y])]

                # Check if max speed has been reached
                if vec_len(acc) <= self.max_speed:
                    total_vel = acc
                    continue
                else:
                    total_vel = scale_to_limit(total_vel, self.priority[int(self.order[y])], self.max_speed)
                    break   # stop; no room left for lower-priority behaviors
            
            # self.get_logger().info(f"Robot {i} velocity: {total_vel}")

            # Ensure the next pose will be within the bounds of the environment
            next_i = pos_i + (total_vel * (1.0 / self.publish_rate))
            if is_within_bounds(next_i, self.resolution, self.origin, self.gridmap):                
                # Publish desired velocity (x,y velocities)
                self.pub_rob_vel(i, total_vel)
            else:
                # Publish negative desired velocity (x,y velocities)
                # NOTE: SENDING NEGATIVE VELOCITIES FOR NOW!!!! - although this approach is naive
                self.get_logger().warn("Robot on the boundary. Moving back")
                self.pub_rob_vel(i, -total_vel)

    def find_avoidance_vector(self, pos_i: np.ndarray, heading:np.ndarray):
        """
            Implements Reynolds' obstacle avoidance logic (simplified to grid map):
            1. Create a 'lookahead cylinder' (rectangle in 2D).
            2. Find the closest occupied cell within that region.
            3. Steer laterally away from that obstacle.

            Parameters:
                pos_i (np.ndarray): current position of robot i
                heading (np.ndarray): unit vector representing heading of the robot i (gotten from the robot velocity)

            Returns a steering vector (2x1) in world coords (zero if no avoidance needed).
        """
        if self.gridmap is None:
            return np.zeros((2,1))
        
        inflated_gridmap = inflate_obstacles(self.gridmap, self.resolution)
        
        mx, my = world_to_map(pos_i, self.origin, self.resolution)

        if not (0 <= mx < self.map_width and 0 <= my < self.map_height):
            return np.zeros((2,1))
        
        # Sampling resolution: check points ahead along heading direction
        steps = int(self.lookahead_distance / self.resolution)
        most_threatening = None
        # min_dist = float('inf')

        for s in range(1, steps):
            # position of the point along the lookahead path
            p_world = pos_i + (heading * (s * self.resolution))
            px, py = float(p_world[0,0]), float(p_world[1,0])
            # convert to map indices
            mx, my = world_to_map(p_world, self.origin, self.resolution)
            if not (0 <= mx < self.map_width and 0 <= my < self.map_height):
                continue
 
            if inflated_gridmap[my, mx] == 100:   # occupied cell (threshold)
                # compute distance of the obstacle to robot center
                obstacle_pos = np.array([[px], [py]])
                dist = vec_len(vec_sub(obstacle_pos, pos_i))
                if dist <= self.min_dist_to_obst:
                    # min_dist = dist
                    most_threatening = np.copy(obstacle_pos)

        # if no obstacle ahead, no steering correction
        if most_threatening is None:
            self.get_logger().info("no threatening obstacle ahead!!!")
            return np.zeros((2,1))
        
        # Compute lateral (sideways) steering vector
        to_obstacle = vec_sub(most_threatening, pos_i)
        # perpendicular to heading (rotate by ±90°)
        perp_left = np.array([[-heading[0,0]], [heading[1,0]]])
        perp_right = -perp_left

        # decide which direction to steer (away from the obstacle)
        steer_dir = perp_right if np.dot(perp_left.reshape(2,), to_obstacle.reshape(2,)) > 0 else perp_left
        # steer_strength = max(self.lookahead_distance - min_dist, 0.0)
        return vec_scale(vec_norm(steer_dir), self.min_dist_to_obst)

    # Publisher function to publish control velocity to each robot in flock
    def pub_rob_vel(self, idx:int, desired_vel: np.ndarray):
        twist = Twist()
        twist.linear = Vector3(x=float(desired_vel[0, 0]), y=float(desired_vel[1, 0]), z=0.0)
        twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.robot_publishers[idx].publish(twist)



def main(args=None):
    rclpy.init(args=args)

    extended_rules = ExtendedRules()
    # rclpy.spin(basic_rules)

    try:
        rclpy.spin(extended_rules)
    except KeyboardInterrupt:
        pass
    extended_rules.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
