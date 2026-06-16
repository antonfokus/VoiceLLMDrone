import json
import sys
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("Установите llama-cpp-python: pip install llama-cpp-python")

LLM_MODEL_PATH = "../models/DroneLlama.gguf"

#  LLM

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_threads=8,
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=64,
    verbose=False,
)


def generate_response(prompt: str) -> str:
    """Генерация компактного JSON-подобного ответа LLM."""
    output = llm(
        f"Human: {prompt}\nAssistant:",
        max_tokens=25,
        temperature=0.1,
        stop=["}"],
    )
    raw = output["choices"][0]["text"].strip()
    start = raw.find("{")
    end = raw.find("}")
    if start != -1:
        raw = (raw[start:] + "}") if end == -1 or not raw.endswith("}") else raw[start : end + 1]
    return raw

#  Валидация 

SUPPORTED_COMMANDS: Dict[int, Dict[str, Any]] = {
    400: {"name": "arm_motors",  "params": ["param1"]},
    22:  {"name": "takeoff",     "params": ["z"]},
    21:  {"name": "land",        "params": []},
    20:  {"name": "rtl",         "params": []},
    176: {"name": "set_mode",    "params": ["param1"]},
    84:  {"name": "move",        "params": ["x", "y", "z"]},
}

SUPPORTED_MODES = {
    "AUTO", "GUIDED", "RTL", "STABILIZE", "LOITER", "POSHOLD",
    "ALTHOLD", "FBWA", "CRUISE", "MANUAL", "FBWB", "ACRO",
    "STEERING", "CIRCLE", "HOLD", "OFFBOARD", "MISSION",
}


def decode_response(llm_output: str) -> dict:
    """Декодирование строки вида {c=400 param1=1} в словарь."""
    cleaned = llm_output.strip().strip("{}").strip()
    result: dict = {}
    for part in cleaned.split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        if val.lstrip("-").isdigit():
            val = int(val)
        if key == "p":
            key = "param1"
        result[key] = val
    return result


def validate_and_parse(llm_output: str) -> Dict[str, Any]:
    """Валидация выхода LLM → структурированная команда."""
    msg = decode_response(llm_output)
    if not msg or "c" not in msg:
        raise ValueError("Некорректный формат LLM-ответа")

    cmd_id = msg["c"]
    if cmd_id not in SUPPORTED_COMMANDS:
        raise ValueError(f"Неизвестный CMD ID: {cmd_id}")

    spec = SUPPORTED_COMMANDS[cmd_id]
    for p in spec["params"]:
        if p not in msg:
            raise ValueError(f"Команда '{spec['name']}' требует параметр '{p}'")

    if cmd_id == 176:
        mode = str(msg["param1"]).upper()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Неизвестный режим: {msg['param1']}")
        msg["param1"] = mode

    if cmd_id == 400:
        spec["name"] = "arm_motors" if msg["param1"] == 1 else "disarm_motors"

    return {
        "command":      cmd_id,
        "command_name": spec["name"],
        "params":       msg,
    }

#  ROS2-нода

class LLMNode(Node):
    def __init__(self):
        super().__init__("llm_node")

        self.sub = self.create_subscription(
            String, "/voice/text", self._text_callback, 10
        )
        self.pub = self.create_publisher(String, "/voice/command", 10)

        self.get_logger().info("llm_node запущен. Ожидаю /voice/text...")

    def _text_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"Обрабатываю текст: '{text}'")
        try:
            llm_raw = generate_response(text)
            self.get_logger().debug(f"LLM raw: {llm_raw}")
            parsed = validate_and_parse(llm_raw)
        except Exception as exc:
            self.get_logger().warn(f"Не удалось распарсить команду: {exc}")
            return

        out = String()
        out.data = json.dumps(parsed, ensure_ascii=False)
        self.pub.publish(out)
        self.get_logger().info(f"Команда отправлена: {parsed['command_name']} → /voice/command")


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
