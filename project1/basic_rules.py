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


class BasicRules(Node):

    def __init__(self):
        super().__init__('basic_rules')

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

        # ATTRIBUTES
        self.robots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # Store robot states: {i: (pos[2x1], vel[2x1])}
        for i in range(self.num_robots):
            self.robots[i] = (np.zeros((2, 1)), np.zeros((2, 1)))

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
        self.connection_timer = self.create_timer(1, self.connect_wait) # wait to allow publisher/subscriber connection
    
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
        self.robots[idx] = (pos, vel)

    # Occupancy gridmap callback
    def map_cb(self, gridmap: OccupancyGrid):
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
        self.get_logger().info(f"{pos_i} has angle {angle} with {pos_j}")
        # self.get_logger().info(self.fov)

        return abs(angle) <= self.fov / 2.0 # Check if neighbor is within half of the FOV angle to be visible

    # Main flocking control loop
    def control_loop(self):
        states = self.robots.copy()
        for i in range(self.num_robots):
            pos_i, vel_i = states[i]
            neighbors = []

            for j in range(self.num_robots):
                if i == j:
                    continue
                pos_j, _ = states[j]
                if self.in_fov(pos_i, vel_i, pos_j):
                    neighbors.append(j)

            # Default: zero behavior vectors
            sep = np.zeros((2, 1))
            align = np.zeros((2, 1))
            coh = np.zeros((2, 1))

            if len(neighbors) > 0:
                self.get_logger().info("Neighbours detected!!")
                # --- Separation ---
                for j in neighbors:
                    pos_j, _ = states[j]
                    diff = vec_sub(pos_i, pos_j)
                    dist = max(vec_len(diff), 1e-6) # to prevent division by zero
                    if dist < self.min_dist:  # only compute separation velocity if dist is less than specified minimum distance 
                        sep += vec_scale(vec_norm(diff), 1.0 / dist)  #scale down difference based on distance
                sep /= len(neighbors)
                sep = sep / (1.0/self.publish_rate)  # convert to velocity unit

                # --- Alignment ---
                avg_v = np.zeros((2, 1))
                for j in neighbors:
                    _, vel_j = states[j]
                    avg_v += vel_j
                avg_v /= len(neighbors)
                align = vec_sub(avg_v, vel_i)

                # --- Cohesion ---
                com = np.zeros((2, 1))
                for j in neighbors:
                    pos_j, _ = states[j]
                    com += pos_j
                com /= len(neighbors)
                coh = vec_sub(com, pos_i)
                coh = coh / (1.0/self.publish_rate)  # convert to velocity unit

            # Weighted average of behaviors
            steer = ((self.w_separation * sep) + (self.w_alignment * align) + (self.w_cohesion * coh)) / 3

            # Compute desired velocity and clamp
            desired = vec_add(vel_i, steer)
            desired = clamp_vec(desired, self.max_speed)

            # Ensure the next pose will be within the bounds of the environment
            next_i = pos_i + (desired * (1.0 / self.publish_rate))
            if is_within_bounds(next_i, self.resolution, self.origin, self.gridmap):                
                # Publish desired velocity (x,y velocities)
                self.pub_rob_vel(i, desired)
            else:
                # Publish negative desired velocity (x,y velocities)
                # NOTE: SENDING NEGATIVE VELOCITIES FOR NOW!!!! - although this approach is naive
                self.pub_rob_vel(i, (desired * -1.0))

    # Publisher function to publish control velocity to each robot in flock
    def pub_rob_vel(self, idx:int, desired_vel: np.ndarray):
        twist = Twist()
        twist.linear = Vector3(x=float(desired_vel[0, 0]), y=float(desired_vel[1, 0]), z=0.0)
        twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.robot_publishers[idx].publish(twist)



def main(args=None):
    rclpy.init(args=args)

    basic_rules = BasicRules()
    # rclpy.spin(basic_rules)

    try:
        rclpy.spin(basic_rules)
    except KeyboardInterrupt:
        pass
    basic_rules.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
