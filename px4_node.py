import json
import math
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

def _px4_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

class PX4Node(Node):
    def __init__(self):
        super().__init__("px4_node")

        qos = _px4_qos()

        # Подписчики
        self.cmd_sub = self.create_subscription(
            String, "/voice/command", self._command_callback, 10
        )
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._status_callback, qos
        )
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self._local_pos_callback, qos,
        )

        # Издатели
        self.cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
        )

        # Состояние
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.local_pos: VehicleLocalPosition | None = None
        self.local_pos_valid = False

        self.target_pos = [0.0, 0.0, 0.0]   # NED [x, y, z]
        self.moving_to_target = False
        self.arrival_tolerance = 0.3          # метры

        self.offboard_counter = 0             # ожидаем ≥10 heartbeat до arming
        self.initial_arm_done = False

        # Таймер 
        self.timer = self.create_timer(0.1, self._timer_callback)

        self.get_logger().info("px4_node запущен. Ожидаю /voice/command...")

    #  Подписки 

    def _status_callback(self, msg: VehicleStatus):
        self.nav_state = msg.nav_state

    def _local_pos_callback(self, msg: VehicleLocalPosition):
        self.local_pos = msg
        self.local_pos_valid = (
            getattr(msg, "xy_valid", True) and getattr(msg, "z_valid", True)
        )

    def _command_callback(self, msg: String):
        """Получить JSON-команду от llm_node и поставить в очередь выполнения."""
        try:
            parsed: Dict[str, Any] = json.loads(msg.data)
            cmd_id: int = parsed["command"]
            params: dict = parsed["params"]
            name: str = parsed.get("command_name", str(cmd_id))
        except Exception as exc:
            self.get_logger().warn(f"Некорректный формат команды: {exc}")
            return

        self.get_logger().info(f"Получена команда: {name} (CMD {cmd_id})")
        self._execute(cmd_id, params)

    #  Таймер (heartbeat + движение) 

    def _timer_callback(self):
        self._publish_offboard_heartbeat()

        # Первые 10 тиков — просто накапливаем heartbeat
        if self.offboard_counter < 10:
            self.offboard_counter += 1
            return

        # После 10 тиков — один раз переключаемся в Offboard и армируем
        if self.offboard_counter == 10 and not self.initial_arm_done:
            self._set_offboard_and_arm()
            self.initial_arm_done = True
            self.offboard_counter += 1
            return

        # Публикуем сетпойнт, если летим к цели
        if self.moving_to_target and self.local_pos_valid:
            self._publish_target_setpoint()
            self._check_arrival()

    #  Выполнение команд 

    def _execute(self, cmd_id: int, params: dict):
        if cmd_id == 400:
            arm = (params.get("param1") == 1)
            self._send_arm_disarm(1.0 if arm else 0.0)
            self.get_logger().info("Arming" if arm else "Disarming")

        elif cmd_id == 22:                               # Takeoff
            alt = abs(float(params.get("z", 5.0)))
            self.target_pos = [0.0, 0.0, -alt]          # PX4 NED: вверх = -z
            self.moving_to_target = True
            self.get_logger().info(f"Takeoff → {alt} м")

        elif cmd_id == 21:                               # Land
            self.moving_to_target = False
            self._send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("Посадка")

        elif cmd_id == 20:                               # RTL
            self.moving_to_target = False
            self._send_vehicle_cmd(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                p1=1.0, p2=4.0,
            )
            self.get_logger().info("Return to Launch")

        elif cmd_id == 176:                              # Set mode
            mode = str(params.get("param1", "")).upper()
            self.get_logger().info(f"Смена режима → {mode}")
            # Здесь можно расширить маппинг режима → param2 MAVLink

        elif cmd_id == 84:                               # Move
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
            z = float(params.get("z", 0.0))
            self.target_pos = [x, y, z]
            self.moving_to_target = True
            self.get_logger().info(f"Движение к NED [{x}, {y}, {z}]")

        else:
            self.get_logger().warn(f"Неизвестный CMD ID: {cmd_id}")

    #  PX4 хелперы 

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = self._ts()
        self.offboard_pub.publish(msg)

    def _set_offboard_and_arm(self):
        self._send_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0
        )
        self._send_arm_disarm(1.0)
        self.get_logger().info("Offboard mode + Arm отправлены")

    def _send_arm_disarm(self, value: float):
        self._send_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=value
        )

    def _send_vehicle_cmd(self, cmd_id: int, p1: float = 0.0, p2: float = 0.0):
        msg = VehicleCommand()
        msg.command = cmd_id
        msg.param1 = p1
        msg.param2 = p2
        msg.timestamp = self._ts()
        self.cmd_pub.publish(msg)

    def _publish_target_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [float(v) for v in self.target_pos]
        msg.timestamp = self._ts()
        self.traj_pub.publish(msg)

    def _check_arrival(self):
        if self.local_pos is None:
            return
        dx = self.target_pos[0] - self.local_pos.x
        dy = self.target_pos[1] - self.local_pos.y
        dz = self.target_pos[2] - self.local_pos.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < self.arrival_tolerance:
            self.moving_to_target = False
            self.get_logger().info(f"Цель достигнута! dist={dist:.2f} м")

    def _ts(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

#  Точка входа 

def main(args=None):
    rclpy.init(args=args)
    node = PX4Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
