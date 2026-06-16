import os
import sys
import json
import queue
import threading
import select

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

VOSK_MODEL_PATH = "../models/vosk-small-ru"
SAMPLE_RATE = 48000
BLOCK_SIZE = 8000

class MicNode(Node):
    def __init__(self):
        super().__init__("mic_node")

        # Vosk
        if not os.path.exists(VOSK_MODEL_PATH):
            self.get_logger().error(f"Vosk model not found: {VOSK_MODEL_PATH}")
            sys.exit(1)
        self.vosk_model = VoskModel(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)

        # Состояние
        self.is_recording = False
        self.get_logger().info("\n>>> НАЖМИТЕ ENTER, ЧТОБЫ НАЧАТЬ/ОСТАНОВИТЬ ЗАПИСЬ <<<\n")

        # Микрофон
        self.device_id = None # По умолчанию
        self.text_pub = self.create_publisher(String, "/voice/text", 10)
        self.audio_queue = queue.Queue()

        # Потоки
        self._stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Таймер для проверки нажатия клавиш (каждые 100мс)
        self.create_timer(0.1, self._check_keyboard)

    def _check_keyboard(self):
        """Проверка ввода в терминале без блокировки программы"""
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline() # Считываем нажатие Enter
            self.is_recording = not self.is_recording
            
            if self.is_recording:
                self.get_logger().info("ЗАПИСЬ ИДЕТ...")
            else:
                self.rec.Reset()
                self.get_logger().info("ОСТАНОВЛЕНО.")

    def _audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_queue.put(bytes(indata))

    def _capture_loop(self):
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=self.device_id,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while rclpy.ok() and not self._stop_event.is_set():
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        if self.rec.AcceptWaveform(data):
                            result = json.loads(self.rec.Result())
                            text = result.get("text", "").strip()
                            if text:
                                self.get_logger().info(f"Текст: {text}")
                                self.text_pub.publish(String(data=text))
                    except queue.Empty:
                        continue
        except Exception as exc:
            self.get_logger().error(f"Ошибка аудио: {exc}")

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
