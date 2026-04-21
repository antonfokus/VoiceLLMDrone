import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, 
    VehicleStatus, VehicleCommand, VehicleLocalPosition
)
import math
import threading
import queue
import sys
import os
from typing import Dict, Any, Tuple

# LLM и валидация
try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")

MODEL_PATH = "./models/DroneLlama.gguf"

# Глобальная загрузка модели
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=8,
    n_ctx=2048,
    n_batch=64,
    verbose=False
)

def generate_response(prompt: str) -> str:
    """Генерация компактного JSON-подобного ответа LLM"""
    prompt_text = f"Human: {prompt}\nAssistant:"
    output = llm(
        prompt_text,
        max_tokens=25,
        temperature=0.1,
        stop=["}"]
    )
    model_output = output["choices"][0]["text"].strip()
    
    response_start = model_output.find("{")
    response_end = model_output.find("}")
    
    if response_start != -1:
        if response_end == -1 or not model_output.endswith("}"):
            model_output = model_output[response_start:] + "}"
        else:
            model_output = model_output[response_start:response_end + 1]
    return model_output

def decode_response(llm_output: str) -> dict:
    """Декодирование строки вида {c=400 param1=1} в словарь"""
    cleaned = llm_output.strip().strip("{}").strip()
    if not cleaned:
        return {}
    result = {}
    for part in cleaned.split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            val = int(val)
        if key == "p":
            key = "param1"
        result[key] = val
    return result

SUPPORTED_COMMANDS = {
    400: {"name": "arm_motors", "params": ["param1"]},
    22:  {"name": "takeoff",     "params": ["z"]},
    21:  {"name": "land",        "params": []},
    20:  {"name": "rtl",         "params": []},
    176: {"name": "set_mode",    "params": ["param1"]},
    84:  {"name": "move",        "params": ["x", "y", "z"]}
}

SUPPORTED_MODES = {"AUTO", "GUIDED", "RTL", "STABILIZE", "LOITER", "POSHOLD", "ALTHOLD",
                   "FBWA", "CRUISE", "MANUAL", "FBWB", "ACRO", "STEERING", "CIRCLE",
                   "HOLD", "OFFBOARD", "MISSION"}

def validate_and_parse(llm_output: str) -> Dict[str, Any]:
    """Валидация выхода LLM и возврат структурированной команды"""
    msg = decode_response(llm_output)
    if not msg:
        raise ValueError("Empty or invalid LLM output")
    if "c" not in msg:
        raise ValueError("Missing 'command' field in LLM output")

    cmd_id = msg["c"]
    if cmd_id not in SUPPORTED_COMMANDS:
        raise ValueError(f"Unsupported command ID {cmd_id}")

    cmd_spec = SUPPORTED_COMMANDS[cmd_id]
    for p in cmd_spec["params"]:
        if p not in msg:
            raise ValueError(f"Command '{cmd_spec['name']}' requires parameter '{p}'")
    
    if cmd_id == 176:          # set_mode
        mode = msg["param1"].upper()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {msg['param1']}")
        msg["param1"] = mode
        
    if cmd_id == 400:          # arm/disarm
        param1 = msg["param1"]
        if param1 not in {0, 1}:
            raise ValueError(f"Arm/disarm param1 must be 0 or 1, got {param1}")
        cmd_spec["name"] = "arm_motors" if param1 == 1 else "disarm_motors"
        msg["param1"] = param1

    return {
        "command": cmd_id,
        "command_name": cmd_spec["name"],
        "params": msg
    }

# ROS2 Узел с интеграцией LLM
class InteractiveOffboardControl(Node):
    def __init__(self):
        super().__init__('interactive_offboard_control')
        
        # QoS для PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Подписчики
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_callback, qos_profile
        )
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._local_pos_callback, qos_profile
        )

        # Издатели
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        # Состояния
        self.offboard_counter = 0
        self.local_pos = None
        self.local_pos_valid = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # Команды и движение
        self.cmd_queue = queue.Queue()          # Очередь для команд (command_id, params)
        self.moving_to_target = False
        self.target_pos = [0.0, 0.0, 0.0]
        self.arrival_tolerance = 0.3
        self.last_log_time = self.get_clock().now()

        # Таймер 10 Гц
        self.timer = self.create_timer(0.1, self._timer_callback)

        # Поток ввода с LLM
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        self.get_logger().info("Node started. Enter natural language commands (e.g. 'take off to 5 meters')")

    def _input_loop(self):
        """Поток чтения команд пользователя → LLM → валидация → очередь"""
        print("\nType your instruction in natural language. Type 'exit' or 'quit' to stop.")
        while rclpy.ok():
            try:
                user_input = input(">> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    self.get_logger().info("Shutting down...")
                    rclpy.shutdown()
                    sys.exit(0)

                # Генерация LLM
                llm_raw = generate_response(user_input)
                self.get_logger().debug(f"LLM raw: {llm_raw}")

                # Валидация
                parsed = validate_and_parse(llm_raw)
                cmd_id = parsed["command"]
                params = parsed["params"]
                self.get_logger().info(f"Parsed command: {parsed['command_name']} {params}")

                # Помещаем в очередь (будет обработано в таймере)
                self.cmd_queue.put((cmd_id, params))

            except ValueError as e:
                self.get_logger().error(f"Validation error: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")

    # ROS2 Callbacks
    def _status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def _local_pos_callback(self, msg):
        self.local_pos = msg
        self.local_pos_valid = getattr(msg, 'xy_valid', True) and getattr(msg, 'z_valid', True)

    # Основной цикл (10 Гц)
    def _timer_callback(self):
        # 1. Heartbeat для offboard
        self._publish_offboard_heartbeat()

        # 2. Разогрев (10 тиков)
        if self.offboard_counter < 10:
            self.offboard_counter += 1
            return

        # 3. Автоматический переход в OFFBOARD + Arm после разогрева
        if self.offboard_counter == 10:
            self._set_offboard_and_arm()
            self.offboard_counter += 1
            return

        # 4. Обработка очереди команд (одна команда за тик, чтобы не блокировать)
        if not self.cmd_queue.empty():
            cmd_id, params = self.cmd_queue.get_nowait()
            self._execute_llm_command(cmd_id, params)

        # 5. Если активен таргет – публикуем setpoint и проверяем прибытие
        if self.moving_to_target and self.local_pos_valid:
            self._publish_target_setpoint()
            self._check_arrival()

    # Исполнение LLM-команд
    def _execute_llm_command(self, cmd_id: int, params: Dict[str, Any]):
        """Выполнить команду, полученную от LLM"""
        # arm/disarm
        if cmd_id == 400:
            arm = params["param1"] == 1
            self._send_arm_disarm(1.0 if arm else 0.0)
            self.get_logger().info(f"{'Arming' if arm else 'Disarming'} motors")

        # takeoff
        elif cmd_id == 22:
            alt = float(params["z"])      # высота в метрах (положительная)
            self.target_pos = [0.0, 0.0, -alt]   # NED: вверх = отрицательный Z
            self.moving_to_target = True
            self.get_logger().info(f"Takeoff to {alt} meters")

        # land
        elif cmd_id == 21:
            self.moving_to_target = False
            self._send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("Landing")

        # rtl
        elif cmd_id == 20:
            self.moving_to_target = False
            cmd = VehicleCommand()
            cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            cmd.param1 = 1.0
            cmd.param2 = 4.0          # AUTO.RTL
            self.cmd_pub.publish(cmd)
            self.get_logger().info("Return to Launch")

        # set_mode
        elif cmd_id == 176:
            mode_str = params["param1"]   # например "OFFBOARD", "AUTO.RTL"
            px4_mode = self._mode_name_to_px4(mode_str)
            if px4_mode is not None:
                cmd = VehicleCommand()
                cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
                cmd.param1 = 1.0
                cmd.param2 = float(px4_mode)
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"Switching to mode {mode_str} (code {px4_mode})")
            else:
                self.get_logger().error(f"Unknown mode: {mode_str}")

        # move (NED)
        elif cmd_id == 84:
            x = float(params["x"])
            y = float(params["y"])
            z = float(params["z"])
            self.target_pos = [x, y, z]
            self.moving_to_target = True
            self.get_logger().info(f"Moving to NED [{x}, {y}, {z}]")

        else:
            self.get_logger().warn(f"Unhandled command ID {cmd_id}")

    def _mode_name_to_px4(self, mode_name: str) -> int:
        """Преобразование строки режима в код PX4 (для VEHICLE_CMD_DO_SET_MODE)"""
        mapping = {
            "MANUAL": 1, "POSITION": 2, "ALTCTL": 3, "AUTO": 4,
            "AUTO.RTL": 4, "AUTO.LAND": 5, "OFFBOARD": 6, "MISSION": 7
        }
        # Нормализуем: убираем точки и приводим к общему виду
        normalized = mode_name.replace(".", "_").upper()
        if normalized in mapping:
            return mapping[normalized]
        # Попробуем напрямую
        if mode_name.upper() in mapping:
            return mapping[mode_name.upper()]
        return None

    # Вспомогательные методы
    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _set_offboard_and_arm(self):
        # Переключение в OFFBOARD
        cmd = VehicleCommand()
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0
        cmd.param2 = 6.0
        self.cmd_pub.publish(cmd)
        # Арминг
        arm_cmd = VehicleCommand()
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.param1 = 1.0
        self.cmd_pub.publish(arm_cmd)
        self.get_logger().info("Switched to OFFBOARD & Armed")

    def _send_arm_disarm(self, param1: float):
        cmd = VehicleCommand()
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = param1
        self.cmd_pub.publish(cmd)

    def _send_vehicle_cmd(self, cmd_id: int):
        cmd = VehicleCommand()
        cmd.command = cmd_id
        self.cmd_pub.publish(cmd)

    def _publish_target_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = self.target_pos
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def _check_arrival(self):
        dx = self.target_pos[0] - self.local_pos.x
        dy = self.target_pos[1] - self.local_pos.y
        dz = self.target_pos[2] - self.local_pos.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 > 2.0:
            self.get_logger().info(f"Distance to target: {dist:.2f}m (tol: {self.arrival_tolerance}m)")
            self.last_log_time = now

        if dist < self.arrival_tolerance:
            self.moving_to_target = False
            self.get_logger().info(f"Target reached! Final dist: {dist:.3f}m")
            self._publish_target_setpoint()   # удерживаем позицию


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveOffboardControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
