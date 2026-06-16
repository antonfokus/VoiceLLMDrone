Выпускная квалификационная работа

"Рекомендательная система с использованием больших языковых моделей"

Студента группы КТмо2-14 Щерба А.С.

## Инструкция
Данная инструкция подразумевает использвание Linux Ubuntu и PX4 SITL/Gazebo

## Установка
1. Должны быть установлены PX4 SITL/Gazebo. Для установки можно использовать <a href="https://docs.px4.io/main/en/dev_setup/dev_env.html">документацию PX4</a>. Также необходимо установить QGroundControl.
2. Клонировать этот репозиторий
```
git clone https://github.com/antonfokus/VoiceLLMDrone.git
```
3. Запустить setup.sh для загрузки библиотек и модели DroneLlama
```bash
cd VoiceLLMDrone
bash setup.sh
  ```
4.  Собрать проект
```bash
colcon build
source install/setup.bash
  ```
   
## Использование
1. Запустить PX4
```bash
cd PX4-Autopilot
make px4_sitl gz_x500
  ```
2. Запустить приложение в другом терминале
```
cd VoiceLLMDrone
python mic_node.py # узел для голосового управления
python llm_node.py # узел для обработки команд через LLM
python px4_node.py # узел для отправки обработанных команд в PX4/Gazebo
  ```
