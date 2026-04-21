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
import json
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer
from typing import Dict, Any, Tuple

# LLM и валидация 
try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")

# Пути к моделям
LLM_MODEL_PATH = "./models/DroneLlama.gguf" # Путь к модели Llama
VOSK_MODEL_PATH = "./models/vosk-small-ru" # Путь к модели Vosk 
SAMPLE_RATE = 48000 # Частота дискретизации для Vosk

# Глобальная загрузка LLM
llm = Llama(
    model_path=LLM_MODEL_PATH,
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
    if not msg or "c" not in msg:
        raise ValueError("Invalid LLM output format")

    cmd_id = msg["c"]
    if cmd_id not in SUPPORTED_COMMANDS:
        raise ValueError(f"Unsupported command ID {cmd_id}")

    cmd_spec = SUPPORTED_COMMANDS[cmd_id]
    for p in cmd_spec["params"]:
        if p not in msg:
            raise ValueError(f"Command '{cmd_spec['name']}' requires parameter '{p}'")
    
    if cmd_id == 176:
        mode = msg["param1"].upper()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {msg['param1']}")
        msg["param1"] = mode
        
    if cmd_id == 400:
        param1 = msg["param1"]
        cmd_spec["name"] = "arm_motors" if param1 == 1 else "disarm_motors"

    return {
        "command": cmd_id,
        "command_name": cmd_spec["name"],
        "params": msg
    }

# ROS2 Узел с голосовым управлением
class VoiceOffboardControl(Node):
    def __init__(self):
        super().__init__('voice_offboard_control')
        
        # Настройка Vosk 
        if not os.path.exists(VOSK_MODEL_PATH):
            self.get_logger().error(f"Vosk model not found at {VOSK_MODEL_PATH}")
            sys.exit(1)
        
        self.vosk_model = VoskModel(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
        self.audio_queue = queue.Queue()
        
        # QoS для PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Подписчики и Издатели 
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_callback, qos_profile)
        self.local_pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._local_pos_callback, qos_profile)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        # Состояния 
        self.offboard_counter = 0
        self.local_pos = None
        self.local_pos_valid = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.target_pos = [0.0, 0.0, 0.0]
        self.moving_to_target = False
        self.arrival_tolerance = 0.3
        self.last_log_time = self.get_clock().now()
        self.cmd_queue = queue.Queue()

        # Таймер 10 Гц
        self.timer = self.create_timer(0.4, self._timer_callback)

        # Поток распознавания голоса
        self.input_thread = threading.Thread(target=self._voice_input_loop, daemon=True)
        self.input_thread.start()
        self.get_logger().info("Node started. Listening for voice commands...")

    def _audio_callback(self, indata, frames, time, status):
        """Захват аудио из микрофона"""
        if status:
            self.get_logger().error(f"Audio capture error: {status}")
        self.audio_queue.put(bytes(indata))

    def _voice_input_loop(self):
        """Поток: Микрофон -> Vosk -> LLM -> Команда"""
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=40000, 
                                   dtype='int16', channels=1, callback=self._audio_callback):
                while rclpy.ok():
                    data = self.audio_queue.get()
                    if self.rec.AcceptWaveform(data):
                        result = json.loads(self.rec.Result())
                        user_text = result.get("text", "").strip()
                        
                        if user_text:
                            self.get_logger().info(f"Heard: '{user_text}'")
                            self._process_natural_language(user_text)
        except Exception as e:
            self.get_logger().error(f"Voice loop error: {e}")

    
    def _process_natural_language(self, text):
        """Обработка текста через LLM"""
        try:
            llm_raw = generate_response(text)
            parsed = validate_and_parse(llm_raw)
            self.get_logger().info(f"Executing: {parsed['command_name']}")
            self.cmd_queue.put((parsed["command"], parsed["params"]))
        except Exception as e:
            self.get_logger().warn(f"Failed to parse command: {e}")

    # ROS2 Callbacks
    def _status_callback(self, msg):
        self.nav_state = msg.nav_state

    def _local_pos_callback(self, msg):
        self.local_pos = msg
        self.local_pos_valid = getattr(msg, 'xy_valid', True) and getattr(msg, 'z_valid', True)

    def _timer_callback(self):
        self._publish_offboard_heartbeat()
        if self.offboard_counter < 10:
            self.offboard_counter += 1
            return
        if self.offboard_counter == 10:
            self._set_offboard_and_arm()
            self.offboard_counter += 1
            return

        if not self.cmd_queue.empty():
            cmd_id, params = self.cmd_queue.get_nowait()
            self._execute_llm_command(cmd_id, params)

        if self.moving_to_target and self.local_pos_valid:
            self._publish_target_setpoint()
            self._check_arrival()

    # Исполнение команд
    """def _execute_llm_command(self, cmd_id: int, params: Dict[str, Any]):
        if cmd_id == 400:
            self._send_arm_disarm(1.0 if params["param1"] == 1 else 0.0)
        elif cmd_id == 22:
            alt = float(params["z"])
            self.target_pos = [0.0, 0.0, -alt]
            self.moving_to_target = True
        elif cmd_id == 21:
            self.moving_to_target = False
            self._send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        elif cmd_id == 84:
            self.target_pos = [float(params["x"]), float(params["y"]), float(params["z"])]
            self.moving_to_target = True
            """
            
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

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _set_offboard_and_arm(self):
        self._send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self._send_arm_disarm(1.0)
        self.get_logger().info("Initial Offboard & Arm sent")

    def _send_arm_disarm(self, param1: float):
        self._send_vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1)

    def _send_vehicle_cmd(self, cmd_id: int, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = cmd_id
        msg.param1 = p1
        msg.param2 = p2
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    def _publish_target_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = self.target_pos
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def _check_arrival(self):
        dx = self.target_pos[0] - self.local_pos.x
        dy = self.target_pos[1] - self.local_pos.y
        dz = self.target_pos[2] - self.local_pos.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < self.arrival_tolerance:
            self.moving_to_target = False
            self.get_logger().info(f"Target reached! Dist: {dist:.2f}m")

def main(args=None):
    rclpy.init(args=args)
    node = VoiceOffboardControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
