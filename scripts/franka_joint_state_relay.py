#!/usr/bin/env python3
"""
/franka/joint_position (Float64MultiArray, 7)  →  /joint_states (JointState)

MoveIt planning scene이 현재 로봇 상태를 알기 위해 /joint_states가 필요.
SHM bridge는 Float64MultiArray로만 publish하므로 이 노드가 변환.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

JOINT_NAMES = [
    'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
    'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
]


class FrankaJointStateRelay(Node):
    def __init__(self):
        super().__init__('franka_joint_state_relay')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(
            Float64MultiArray, '/franka/joint_position', self._cb, qos)
        self.get_logger().info(
            '/franka/joint_position → /joint_states relay started')

    def _cb(self, msg: Float64MultiArray):
        if len(msg.data) < 7:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = JOINT_NAMES
        js.position = [float(msg.data[i]) for i in range(7)]
        js.velocity = [0.0] * 7
        js.effort   = [0.0] * 7
        self._pub.publish(js)


def main():
    rclpy.init()
    node = FrankaJointStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
