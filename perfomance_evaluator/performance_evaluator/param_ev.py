#!/usr/bin/env python3

import csv
import math
import os
import statistics
import time
from datetime import datetime

import numpy as np
import psutil
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, TwistStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

NUMERIC_FIELDS = [
    'planning_time_s', 'total_navigation_time_s', 'planned_path_length_m',
    'executed_path_length_m', 'replans_count', 'mean_cpu_percent', 'max_cpu_percent',
    'mean_ram_mib', 'max_ram_mib', 'mean_linear_vel_mps', 'max_linear_vel_mps',
    'mean_angular_vel_radps', 'max_angular_vel_radps',
]

def path_length(points) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.hypot(dx, dy)
    return total

def path_msg_to_points(path_msg: Path):
    return [(p.pose.position.x, p.pose.position.y) for p in path_msg.poses]

def quat_to_yaw(q) -> float:
    # standard yaw-from-quaternion for a planar robot
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def detect_cmd_vel_type(node: Node, topic: str, timeout_sec: float = 3.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for name, types in node.get_topic_names_and_types():
            if name == topic:
                for t in types:
                    if t.endswith('TwistStamped'):
                        return 'TwistStamped'
                    if t.endswith('/Twist'):
                        return 'Twist'
        time.sleep(0.2)
    return None

class PerformanceEvaluator(Node):

    def __init__(self):
        super().__init__('performance_evaluator')
        cb_group = ReentrantCallbackGroup()

        # ---------------- parameters ----------------
        self.declare_parameter('planner_id', '')
        self.declare_parameter('controller_id', '')
        self.declare_parameter('run_label', 'run')
        self.declare_parameter('goal_x', -4.72791)
        self.declare_parameter('goal_y', 18.3567)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel', True)
        self.declare_parameter('resource_sample_hz', 2.0)
        self.declare_parameter('nav2_process_names',
                                ['planner_server', 'controller_server'])
        self.declare_parameter('csv_path', 'nav2_benchmark_results.csv')
        self.declare_parameter('action_result_timeout_sec', 300.0)
        self.declare_parameter('enable_warmup', True)
        self.declare_parameter('warmup_every_trial', False)
        self.declare_parameter('warmup_settle_sec', 1.5)
        self.declare_parameter('warmup_planner_id', '')
        self.declare_parameter('save_paths', True)
        self.declare_parameter('paths_dir', '.')
        self.declare_parameter('odom_record_hz', 10.0)
        # multi-trial
        self.declare_parameter('num_trials', 1)
        self.declare_parameter('return_settle_sec', 1.0)

        self.planner_id = self.get_parameter('planner_id').value
        self.controller_id = self.get_parameter('controller_id').value
        self.run_label = self.get_parameter('run_label').value
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_yaw = self.get_parameter('goal_yaw').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.plan_topic = self.get_parameter('plan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.use_stamped = self.get_parameter('use_stamped_cmd_vel').value
        self.sample_hz = self.get_parameter('resource_sample_hz').value
        self.nav2_process_names = set(self.get_parameter('nav2_process_names').value)
        self._nav2_procs = []
        self.csv_path = os.path.expanduser(self.get_parameter('csv_path').value)
        self.result_timeout = self.get_parameter('action_result_timeout_sec').value
        self.enable_warmup = self.get_parameter('enable_warmup').value
        self.warmup_every_trial = self.get_parameter('warmup_every_trial').value
        self.warmup_settle_sec = self.get_parameter('warmup_settle_sec').value
        self.warmup_planner_id = self.get_parameter('warmup_planner_id').value
        self.save_paths = self.get_parameter('save_paths').value
        self.paths_dir = os.path.expanduser(self.get_parameter('paths_dir').value)
        self.odom_record_hz = self.get_parameter('odom_record_hz').value
        self.num_trials = max(1, int(self.get_parameter('num_trials').value))
        self.return_settle_sec = self.get_parameter('return_settle_sec').value

        # ---------------- per-leg state (reset before every leg) ----------------
        self._goal_sent_time = None
        self._first_plan_time = None
        self._planned_path_len = None
        self._planned_path_points = []
        self._nav_done = False
        self._nav_success = False
        self._final_recoveries = 0
        self._nav_result_time = None
        self._last_odom_xy = None
        self._executed_len = 0.0
        self._executed_path_points = []
        self._last_odom_record_time = 0.0
        self._lin_vels = []
        self._ang_vels = []
        self._cpu_samples = []
        self._ram_samples_mib = []

        # start pose, captured from odom at startup
        self.start_x = None
        self.start_y = None
        self.start_yaw = 0.0
        self._latest_odom_pose = None  # (x, y, yaw), updated continuously

        self._all_rows = []  # one dict per evaluated trial, for the summary

        qos_reliable = QoSProfile(depth=10)
        qos_reliable.reliability = QoSReliabilityPolicy.RELIABLE

        # ---------------- subscriptions ----------------
        self.create_subscription(Path, self.plan_topic, self._on_plan, qos_reliable,
                                  callback_group=cb_group)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20,
                                  callback_group=cb_group)

        detected = detect_cmd_vel_type(self, self.cmd_vel_topic, timeout_sec=3.0)
        if detected == 'TwistStamped':
            self.get_logger().info(f"Auto-detected {self.cmd_vel_topic} as TwistStamped.")
            self.create_subscription(TwistStamped, self.cmd_vel_topic, self._on_cmd_vel_stamped,
                                      20, callback_group=cb_group)
        elif detected == 'Twist':
            self.get_logger().info(f"Auto-detected {self.cmd_vel_topic} as Twist.")
            self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, 20,
                                      callback_group=cb_group)
        else:
            self.get_logger().warn(
                f"Could not detect message type on '{self.cmd_vel_topic}' within 3s. "
                f"Falling back to use_stamped_cmd_vel param (currently {self.use_stamped})."
            )
            if self.use_stamped:
                self.create_subscription(TwistStamped, self.cmd_vel_topic, self._on_cmd_vel_stamped,
                                          20, callback_group=cb_group)
            else:
                self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, 20,
                                          callback_group=cb_group)

        # ---------------- action clients ----------------
        self._ac = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                 callback_group=cb_group)
        self._compute_path_ac = ActionClient(self, ComputePathToPose, 'compute_path_to_pose',
                                              callback_group=cb_group)

        # ---------------- resource sampling timer ----------------
        self.create_timer(1.0 / self.sample_hz, self._sample_resources,
                           callback_group=cb_group)

        self.get_logger().info(
            f"Benchmark configured: planner_id='{self.planner_id}' "
            f"controller_id='{self.controller_id}' goal=({self.goal_x}, {self.goal_y}, "
            f"{self.goal_yaw}) num_trials={self.num_trials}"
        )

    
    # Startup: capture the robot's actual current pose as "start"
    
    def capture_start_pose(self, timeout_sec: float = 10.0):
        self.get_logger().info("Waiting for odometry to capture the start pose...")
        deadline = time.monotonic() + timeout_sec
        while self._latest_odom_pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self._latest_odom_pose is None:
            self.get_logger().error(
                f"No odometry received on '{self.odom_topic}' within {timeout_sec}s -- "
                f"cannot auto-capture start pose. Check odom_topic."
            )
            rclpy.shutdown()
            return
        self.start_x, self.start_y, self.start_yaw = self._latest_odom_pose
        self.get_logger().info(
            f"Captured start pose: ({self.start_x:.3f}, {self.start_y:.3f}, "
            f"yaw={self.start_yaw:.3f})"
        )

    
    # Warmup (discarded planner-only call, robot does not move)

    def warmup(self):
        if not self._compute_path_ac.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("compute_path_to_pose server not available -- skipping warmup.")
            return

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = self.goal_x
        goal_msg.goal.pose.position.y = self.goal_y
        qz = math.sin(self.goal_yaw / 2.0)
        qw = math.cos(self.goal_yaw / 2.0)
        goal_msg.goal.pose.orientation.z = qz
        goal_msg.goal.pose.orientation.w = qw
        goal_msg.use_start = False
        goal_msg.planner_id = self.warmup_planner_id

        self.get_logger().info("Warming up planner (discarded, robot will not move)...")
        t0 = time.monotonic()
        send_future = self._compute_path_ac.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Warmup goal rejected or timed out -- proceeding without it.")
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
        elapsed = time.monotonic() - t0
        self.get_logger().info(f"Warmup complete in {elapsed:.4f}s. "
                                f"Settling {self.warmup_settle_sec}s...")
        time.sleep(self.warmup_settle_sec)

    
    # Callbacks
    
    def _on_plan(self, msg: Path):
        if self._first_plan_time is None and self._goal_sent_time is not None:
            self._first_plan_time = time.monotonic()
            self._planned_path_points = path_msg_to_points(msg)
            self._planned_path_len = path_length(self._planned_path_points)
            planning_time = self._first_plan_time - self._goal_sent_time
            self.get_logger().info(
                f"[FIRST PLAN] planning_time={planning_time:.4f}s "
                f"planned_path_length={self._planned_path_len:.4f}m "
                f"(#poses={len(msg.poses)})"
            )

    def _on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self._latest_odom_pose = (x, y, yaw)  # always updated, used for start-pose capture

        if self._goal_sent_time is None or self._nav_done:
            return

        if self._last_odom_xy is not None:
            dx = x - self._last_odom_xy[0]
            dy = y - self._last_odom_xy[1]
            d = math.hypot(dx, dy)
            if d > 0.001:
                self._executed_len += d
        self._last_odom_xy = (x, y)

        now = time.monotonic()
        if now - self._last_odom_record_time >= 1.0 / self.odom_record_hz:
            self._executed_path_points.append((x, y))
            self._last_odom_record_time = now

    def _on_cmd_vel(self, msg: Twist):
        if self._goal_sent_time is None or self._nav_done:
            return
        self._lin_vels.append(abs(msg.linear.x))
        self._ang_vels.append(abs(msg.angular.z))

    def _on_cmd_vel_stamped(self, msg: TwistStamped):
        if self._goal_sent_time is None or self._nav_done:
            return
        self._lin_vels.append(abs(msg.twist.linear.x))
        self._ang_vels.append(abs(msg.twist.angular.z))

    def _discover_nav2_processes(self):
        found = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] in self.nav2_process_names:
                    found.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        names_found = sorted({p.info['name'] for p in found})
        missing = self.nav2_process_names - set(names_found)
        if missing:
            self.get_logger().warn(f"Nav2 processes not found: {sorted(missing)}. "
                                    f"Tracking only: {names_found}")
        for p in found:
            try:
                p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._nav2_procs = found

    def _sample_resources(self):
        if self._goal_sent_time is None or self._nav_done:
            return
        cpu_total = 0.0
        ram_total_mib = 0.0
        alive = []
        for p in self._nav2_procs:
            try:
                cpu_total += p.cpu_percent(interval=None)
                ram_total_mib += p.memory_info().rss / (1024 * 1024)
                alive.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self._nav2_procs = alive
        self._cpu_samples.append(cpu_total)
        self._ram_samples_mib.append(ram_total_mib)

    
    # One leg of navigation: send a goal, block until result, optionally evaluate
    
    def _reset_leg_state(self):
        self._goal_sent_time = None
        self._first_plan_time = None
        self._planned_path_len = None
        self._planned_path_points = []
        self._nav_done = False
        self._nav_success = False
        self._final_recoveries = 0
        self._nav_result_time = None
        self._last_odom_xy = None
        self._executed_len = 0.0
        self._executed_path_points = []
        self._last_odom_record_time = 0.0
        self._lin_vels = []
        self._ang_vels = []
        self._cpu_samples = []
        self._ram_samples_mib = []

    def run_leg(self, x, y, yaw, run_label, evaluate: bool):
        if not self._ac.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("navigate_to_pose action server not available.")
            return False

        self._reset_leg_state()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self._goal_sent_time = time.monotonic()
        self._discover_nav2_processes()

        kind = "EVAL" if evaluate else "reposition"
        self.get_logger().info(
            f"[{run_label}] ({kind}) sending goal -> ({x:.3f}, {y:.3f})"
        )

        send_future = self._ac.send_goal_async(goal_msg, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self.result_timeout)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"[{run_label}] Goal rejected or timed out.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.result_timeout)
        self._nav_result_time = time.monotonic()
        result = result_future.result()
        if result is None:
            self.get_logger().error(f"[{run_label}] No result received (timeout).")
            self._nav_success = False
        else:
            self._nav_success = (result.status == GoalStatus.STATUS_SUCCEEDED)
            if not self._nav_success:
                self.get_logger().warn(f"[{run_label}] Ended with status={result.status}.")
        self._nav_done = True

        if evaluate:
            row = self._build_row(run_label)
            self._print_table(row)
            self._write_csv(row)
            if self.save_paths:
                self._save_paths(run_label)
            self._all_rows.append(row)

        return self._nav_success

    def _on_feedback(self, feedback_msg):
        self._final_recoveries = feedback_msg.feedback.number_of_recoveries

    
    # Multi-trial driver
    
    def run_all_trials(self):
        self.capture_start_pose()
        if not rclpy.ok():
            return

        if self.enable_warmup:
            self.warmup()

        for trial in range(1, self.num_trials + 1):
            self.get_logger().info(f"===== Trial {trial}/{self.num_trials} =====")

            if self.warmup_every_trial and trial > 1:
                self.warmup()

            label = f"{self.run_label}_t{trial}"
            self.run_leg(self.goal_x, self.goal_y, self.goal_yaw, label, evaluate=True)

            if trial < self.num_trials:
                self.get_logger().info(f"Returning to start for trial {trial + 1}...")
                self.run_leg(self.start_x, self.start_y, self.start_yaw,
                             f"{label}_return", evaluate=False)
                time.sleep(self.return_settle_sec)

        self._print_summary()
        self.get_logger().info(f"All {self.num_trials} trial(s) complete. Shutting down.")

    
    # Reporting
    
    def _build_row(self, run_label):
        planning_time = (
            self._first_plan_time - self._goal_sent_time
            if self._first_plan_time else float('nan')
        )
        total_nav_time = self._nav_result_time - self._goal_sent_time
        planned_len = self._planned_path_len if self._planned_path_len else float('nan')

        def stat(vals, fn, default=0.0):
            return fn(vals) if vals else default

        mean_lin = stat(self._lin_vels, statistics.mean)
        max_lin = stat(self._lin_vels, max)
        mean_ang = stat(self._ang_vels, statistics.mean)
        max_ang = stat(self._ang_vels, max)
        if not self._lin_vels:
            self.get_logger().warn(f"[{run_label}] No /cmd_vel samples -- velocity metrics are 0.0.")

        mean_cpu = stat(self._cpu_samples, statistics.mean)
        max_cpu = stat(self._cpu_samples, max)
        mean_ram = stat(self._ram_samples_mib, statistics.mean)
        max_ram = stat(self._ram_samples_mib, max)

        return {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'run_label': run_label,
            'planner_id': self.planner_id or '(active plugin)',
            'controller_id': self.controller_id or '(active plugin)',
            'success': self._nav_success,
            'planning_time_s': round(planning_time, 4),
            'total_navigation_time_s': round(total_nav_time, 4),
            'planned_path_length_m': round(planned_len, 4),
            'executed_path_length_m': round(self._executed_len, 4),
            'replans_count': self._final_recoveries,
            'mean_cpu_percent': round(mean_cpu, 2),
            'max_cpu_percent': round(max_cpu, 2),
            'mean_ram_mib': round(mean_ram, 2),
            'max_ram_mib': round(max_ram, 2),
            'mean_linear_vel_mps': round(mean_lin, 4),
            'max_linear_vel_mps': round(max_lin, 4),
            'mean_angular_vel_radps': round(mean_ang, 4),
            'max_angular_vel_radps': round(max_ang, 4),
        }

    def _print_table(self, row):
        print("\n" + "=" * 62)
        print(f" NAV2 BENCHMARK RESULT  [{row['run_label']}]  "
              f"planner={row['planner_id']}  controller={row['controller_id']}")
        print("=" * 62)
        print(f" {'Success':<32}: {row['success']}")
        print(f" {'Planning Time (s)':<32}: {row['planning_time_s']}")
        print(f" {'Total Navigation Time (s)':<32}: {row['total_navigation_time_s']}")
        print(f" {'Planned Path Length (m)':<32}: {row['planned_path_length_m']}")
        print(f" {'Executed Path Length (m)':<32}: {row['executed_path_length_m']}")
        print(f" {'Replans Count':<32}: {row['replans_count']}")
        print(f" {'Mean / Max CPU - Nav2 procs (%)':<32}: {row['mean_cpu_percent']} / {row['max_cpu_percent']}")
        print(f" {'Mean / Max RAM - Nav2 procs (MiB)':<32}: {row['mean_ram_mib']} / {row['max_ram_mib']}")
        print(f" {'Mean / Max Linear Vel (m/s)':<32}: "
              f"{row['mean_linear_vel_mps']} / {row['max_linear_vel_mps']}")
        print(f" {'Mean / Max Angular Vel (rad/s)':<32}: "
              f"{row['mean_angular_vel_radps']} / {row['max_angular_vel_radps']}")
        print("=" * 62 + "\n")

    def _print_summary(self):
        if not self._all_rows:
            return
        print("\n" + "#" * 70)
        print(f" SUMMARY across {len(self._all_rows)} trial(s) -- "
              f"{self.run_label} ({self.planner_id or '(active)'} + "
              f"{self.controller_id or '(active)'})")
        print("#" * 70)
        n_success = sum(1 for r in self._all_rows if r['success'])
        print(f" Success rate: {n_success}/{len(self._all_rows)}")
        for field in NUMERIC_FIELDS:
            vals = [r[field] for r in self._all_rows if not (isinstance(r[field], float)
                    and math.isnan(r[field]))]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f" {field:<28}: {mean:.4f} +/- {std:.4f}")
        print("#" * 70 + "\n")

    def _write_csv(self, row):
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        self.get_logger().info(f"[{row['run_label']}] appended to {self.csv_path}")

    def _save_paths(self, run_label):
        os.makedirs(self.paths_dir, exist_ok=True)
        fname = os.path.join(self.paths_dir, f"{run_label}_paths.npz")
        planned = np.array(self._planned_path_points, dtype=float) \
            if self._planned_path_points else np.zeros((0, 2))
        executed = np.array(self._executed_path_points, dtype=float) \
            if self._executed_path_points else np.zeros((0, 2))
        np.savez(
            fname,
            planned_path=planned,
            executed_path=executed,
            run_label=run_label,
            planner_id=self.planner_id,
            controller_id=self.controller_id,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
        )
        self.get_logger().info(
            f"[{run_label}] saved paths ({len(planned)} planned / "
            f"{len(executed)} executed pts) -> {fname}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceEvaluator()
    try:
        node.run_all_trials()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()