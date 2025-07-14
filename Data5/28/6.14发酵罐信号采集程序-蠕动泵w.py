# deepseek_combined.py
import minimalmodbus
import time
import keyboard
import serial
from serial import SerialException
import threading
import queue
import csv
import os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import cProfile
import pstats
import io
import serial.tools.list_ports  # 新增：用于列出可用串口

# ------------------ 全局配置 -------------------
# MODBUS配置（泵控制）
PORT_MODBUS = 'COM9'  # 修改为独立的阀门控制串口
BAUDRATE_MODBUS = 9600
SLAVE_ADDRESS = 1

# 串口配置（双数据采集）
COM3_CONFIG = {    #电导
    'port': 'COM5',
    'baudrate': 115200,
    'timeout': 0.1  # 已优化为0.1秒
}

COM6_CONFIG = {         #荧光
    'port': 'COM37',
    'baudrate': 9600,
    'bytesize': 8,
    'parity': serial.PARITY_NONE,
    'stopbits': 1,
    'slave_address': 1
}

COM7_CONFIG = {        #浊度1
    'port': 'COM11',
    'baudrate': 9600,
    'bytesize': 8,
    'parity': serial.PARITY_NONE,
    'stopbits': 1,
    'timeout': 1
}

# 新增：COM15浊度传感器配置（浊度2）
COM15_CONFIG = {        #浊度2
    'port': 'COM8',
    'baudrate': 9600,
    'bytesize': 8,
    'parity': serial.PARITY_NONE,
    'stopbits': 1,
    'timeout': 1
}

AD_RESOLUTION = 4096
VOLTAGE_REF = 10.0

CMD_REGISTER = 0
POS_REGISTER = 1
SPD_REGISTER = 3
STATUS_REGISTER = 4
VALVE_CHANNEL_CMD = 0x08

MAX_STEPS = 384000
MAX_SPEED = 50000
VALVE_CHANNELS = [1]
#VALVE_CHANNELS = [1, 1, 1, 1]

command_queue = queue.Queue()
data_queue = queue.Queue(maxsize=1000)  # 限制队列大小
stop_event = threading.Event()


# ------------------ 新泵控制模块 -------------------
class NewPumpController:
    def __init__(self, port, pump_address):
        self.ser = None
        self.port = port
        self.pump_address = pump_address
        self.RUN_REGISTER = 0x0000
        self.DIRECTION_REGISTER = 0x0001
        self.SPEED_REGISTER = 0x0002
        self.initialize_serial()
        
    def initialize_serial(self):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=9600,
                    bytesize=serial.EIGHTBITS,
                    stopbits=serial.STOPBITS_ONE,  # 修改为1个停止位（更常见）
                    parity=serial.PARITY_NONE,
                    timeout=1
                )
                print(f"{self.port} 泵串口连接成功")
                return
            except SerialException as e:
                print(f"{self.port} 泵连接失败 (尝试 {attempt+1}/{max_attempts}): {e}")
                time.sleep(1)
        
        print(f"错误：无法连接 {self.port}，程序退出")
        exit(1)
    
    def calculate_crc(self, data):
        """计算 Modbus RTU CRC 校验码（低字节在前）"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001  # 多项式 x^16 + x^15 + x^2 + 1
                else:
                    crc >>= 1
        return crc

    def write_register(self, reg_addr, reg_value):
        """
        写入单个寄存器（功能码 0x06）
        返回 True 表示成功，False 表示失败
        """
        function_code = 0x06
        # 构建请求帧
        data = bytes([self.pump_address, function_code]) + reg_addr.to_bytes(2, 'big') + reg_value.to_bytes(2, 'big')
        crc = self.calculate_crc(data)
        crc_bytes = [(crc & 0xFF), (crc >> 8) & 0xFF]  # CRC 低字节在前
        frame = data + bytes(crc_bytes)
        
        # 发送请求
        self.ser.write(frame)
        time.sleep(0.2)  # 增加等待时间
        
        # 读取响应（最大读取长度为 100 字节）
        response = self.ser.read(100)
        if not response:
            print(f"{self.port} 泵未收到响应")
            return False
        
        # 响应长度检查（功能码0x06的响应应为8字节）
        if len(response) < 8:
            print(f"{self.port} 泵响应长度不足: {len(response)}")
            return False
        
        # 解析响应
        resp_address = response[0]
        resp_func = response[1]
        resp_reg = int.from_bytes(response[2:4], 'big')
        resp_val = int.from_bytes(response[4:6], 'big')
        received_crc = int.from_bytes(response[6:8], 'little')  # CRC 低字节在前
        
        # 重新计算 CRC
        expected_crc = self.calculate_crc(response[:6])
        expected_crc_low = expected_crc & 0xFF
        expected_crc_high = (expected_crc >> 8) & 0xFF
        
        # 验证响应内容
        if (resp_address == self.pump_address and 
            resp_func == function_code and 
            resp_reg == reg_addr and 
            resp_val == reg_value and 
            received_crc == ((expected_crc_high << 8) | expected_crc_low)):
            return True
        else:
            print(f"{self.port} 泵响应校验失败")
            return False

    def start_pump(self, direction, speed_rpm):
        """启动泵，设置方向和速度"""
        # 设置方向
        if not self.write_register(self.DIRECTION_REGISTER, direction):
            print(f"{self.port} 泵方向设置失败")
            return False
        
        # 设置速度（速度值 = RPM * 10）
        speed_value = int(speed_rpm * 10)
        if not self.write_register(self.SPEED_REGISTER, speed_value):
            print(f"{self.port} 泵速度设置失败")
            return False
        
        # 启动泵
        if self.write_register(self.RUN_REGISTER, 0x0001):
            print(f"{self.port} 泵已启动 - {'正转' if direction == 0x0001 else '反转'}，速度{speed_rpm}转/分钟")
            return True
        return False

    def stop_pump(self):
        """停止泵"""
        if self.write_register(self.RUN_REGISTER, 0x0000):
            print(f"{self.port} 泵已停止")
            return True
        print(f"{self.port} 泵停止失败")
        return False

    def safe_shutdown(self):
        """安全关闭泵"""
        stop_success = False
        for attempt in range(3):
            print(f"尝试停止 {self.port} 泵... (尝试 {attempt+1}/3)")
            if self.stop_pump():
                stop_success = True
                break
            time.sleep(0.5)
        
        if not stop_success:
            print(f"警告：未能确认 {self.port} 泵停止状态！")
        return stop_success

# ------------------ 阀门控制模块 -------------------
class ValveController:
    def __init__(self):
        self.instrument = None
        self.initialize_modbus()
        
    def initialize_modbus(self):
        try:
            #self.instrument = minimalmodbus.Instrument(PORT_MODBUS, SLAVE_ADDRESS)
            #self.instrument.serial.baudrate = BAUDRATE_MODBUS
            #self.instrument.serial.timeout = 1
            #self.instrument.mode = minimalmodbus.MODE_RTU
            #self.instrument.clear_buffers_before_each_transaction = True
            print("阀门控制器初始化成功")
        except SerialException as e:
            print(f"阀门控制器连接失败: {e}")
            exit()

    def check_status(self):
        try:
            response = self.instrument.read_registers(STATUS_REGISTER, 2, functioncode=4)
            status_low = response[0]
            status_high = response[1]
            motor_busy = (status_low >> 8) & 0x01 == 0
            valve_channel = status_high & 0x0F
            return {"motor_busy": motor_busy, "valve_channel": valve_channel}
        except Exception as e:
            print(f"阀门状态读取失败: {e}")
            return {"motor_busy": True, "valve_channel": 0}

    def send_command(self, command, parameter=0, wait=True):
        try:
            value = (command << 8) | parameter
            self.instrument.write_register(CMD_REGISTER, value, functioncode=6)
            if wait:
                while self.check_status()["motor_busy"]:
                    time.sleep(0.1)
            print(f"阀门指令 0x{command:02X} 参数 0x{parameter:02X} 发送成功")
            return True
        except Exception as e:
            print(f"阀门指令发送失败: {e}")
            return False

    def switch_valve(self, channel):
        if 1 <= channel <= 15:
            print(f"\n切换到通道 {channel}...")
            command_queue.put(0)
            success = self.send_command(VALVE_CHANNEL_CMD, channel, wait=True)
            if success:
                time.sleep(3)
            return success
        print("\n无效的通道号（1-15）")
        return False

    def initialize_valve(self):
        print("\n正在初始化阀门控制器...")
        return self.send_command(0x06, 0x01)
        
    def check_pause(self):
        if keyboard.is_pressed('shift') or keyboard.is_pressed('left shift') or keyboard.is_pressed('right shift'):
            print("\n检测到Shift键，程序暂停...（按回车继续）")
            start_pause = time.time()
            while True:
                if keyboard.is_pressed('enter'):
                    print("继续运行...")
                    return True
                if stop_event.is_set():
                    return False
                if time.time() - start_pause > 3600:
                    print("暂停超时，继续运行...")
                    return True
                time.sleep(0.05)
        return True

# ------------------ 主控制模块 -------------------
class MainController:
    def __init__(self):
        self.valve_controller = ValveController()
        self.pump1 = NewPumpController('COM34', 0x01)  # 1号泵
        self.pump2 = NewPumpController('COM41', 0x01)  # 2号泵
        
    def initialize_system(self):
        if not self.valve_controller.initialize_valve():
            return False
        return True
    
    def main_loop(self):
        try:
            #if not self.initialize_system():
            #    return
                
            while not stop_event.is_set() and not keyboard.is_pressed('esc'):
                for channel in VALVE_CHANNELS:
                    print(f"\n--- 处理通道 {channel} ---")
                    
                    #if not self.valve_controller.switch_valve(channel):
                    #    continue
                    #time.sleep(2)
                    
                    print("\n执行吸液操作...")
                    #command_queue.put(0)
                    # 使用1号泵进行吸液
                    if not self.pump1.start_pump(0x0001, 20):  # 正转，20转/分钟
                        continue
                    #time.sleep(5)  # 吸液5秒
                    #self.pump1.stop_pump()
                    #time.sleep(0.5)
                    
                    #if not self.valve_controller.switch_valve(6):
                    #    continue
                    #time.sleep(2)
                    
                    #print("\n执行排液操作...")
                    #command_queue.put(1)
                    # 使用2号泵进行排液
                    if not self.pump2.start_pump(0x0001, 10):  # 正转，10转/分钟
                        continue
                    for _ in range(20):
                        if stop_event.is_set():
                            return
                        time.sleep(1)    
                    time.sleep(600)  # 
                    #self.pump2.stop_pump()
                    if not self.pump1.start_pump(0x0001, 0):  # 正转，10转/分钟
                        continue                    
                    if not self.pump2.start_pump(0x0001, 0):  # 正转，20转/分钟
                        continue                    
                    
                    for _ in range(20):
                        if stop_event.is_set():
                            return
                        time.sleep(1)
                    time.sleep(1200)
                    
                    print(f"\n通道 {channel} 完成，等待运行...")
                    start_time = time.time()
                    while time.time() - start_time < 1:
                        if stop_event.is_set() or keyboard.is_pressed('esc'):
                            return
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("程序手动终止")
            stop_event.set()
        finally:
            # 安全关闭所有设备
            self.pump1.safe_shutdown()
            self.pump2.safe_shutdown()
            #self.valve_controller.send_command(0x04, 0x00)
            stop_event.set()

# ------------------ 数据采集模块 -------------------
class DataCollector:
    def __init__(self):
        self.ser_com3 = None
        self.instrument_com6 = None
        self.ser_com7 = None
        self.ser_com15 = None  # 新增：COM15浊度传感器
        self.latest_com3_data = [0.0, 0.0]
        self.latest_com6_data = [0.0, 0.0]
        self.latest_com7_data = [0, 0]  # 浊度1数据
        self.latest_com15_data = [0, 0]  # 新增：浊度2数据
        self.data_lock = threading.Lock()
        self.initialize_serial()
        self.last_log = ""
        self.channel6_files = {}
        self.current_normal_file = None  # 新增：单独管理普通通道文件
    def initialize_serial(self):
        # COM3初始化
        try:
            self.ser_com3 = serial.Serial(**COM3_CONFIG)
        except SerialException as e:
            print(f"COM3连接失败: {e}")
            exit()

        # COM6初始化
        try:
            self.instrument_com6 = minimalmodbus.Instrument(
                COM6_CONFIG['port'], 
                COM6_CONFIG['slave_address']
            )
            self.instrument_com6.serial.baudrate = COM6_CONFIG['baudrate']
            self.instrument_com6.serial.bytesize = COM6_CONFIG['bytesize']
            self.instrument_com6.serial.parity = COM6_CONFIG['parity']
            self.instrument_com6.serial.stopbits = COM6_CONFIG['stopbits']
            self.instrument_com6.serial.timeout = 1
            self.instrument_com6.mode = minimalmodbus.MODE_RTU
            self.instrument_com6.clear_buffers_before_each_transaction = True
        except Exception as e:
            print(f"COM6连接失败: {e}")
            exit()

        # COM7初始化
        try:
            self.ser_com7 = serial.Serial(**COM7_CONFIG)
            self.ser_com7.reset_input_buffer()
        except SerialException as e:
            print(f"COM7连接失败: {e}")
            exit()
            
        # 新增：COM15初始化
        try:
            self.ser_com15 = serial.Serial(**COM15_CONFIG)
            self.ser_com15.reset_input_buffer()
            print("COM15浊度传感器连接成功")
        except SerialException as e:
            print(f"COM15浊度传感器连接失败: {e}")
            # 即使连接失败也要继续运行，但使用默认值
            self.ser_com15 = None

    def read_com7_data(self):
        try:
            dirt_value, ad_value = None, None
            cmd_dirt = bytes([0x18, 0x05, 0x00, 0x01, 0x0D])
            self.ser_com7.write(cmd_dirt)
            time.sleep(0.05)  # 优化后的延迟
            response = self.ser_com7.read(5)
            if len(response) == 5 and response[0] == 0x18 and response[-1] == 0x0D:
                dirt_value = response[3]

            cmd_ad = bytes([0x18, 0x05, 0x00, 0x02, 0x0D])
            self.ser_com7.write(cmd_ad)
            time.sleep(0.05)  # 优化后的延迟
            response_ad = self.ser_com7.read(6)
            if len(response_ad) == 6 and response_ad[0] == 0x18 and response_ad[-1] == 0x0D:
                ad_high = response_ad[3]
                ad_low = response_ad[4]
                ad_value = (ad_high << 8) | ad_low

            return (dirt_value if dirt_value is not None else self.latest_com7_data[0], 
                    ad_value if ad_value is not None else self.latest_com7_data[1])
        except Exception as e:
            print(f"COM7数据读取失败: {e}")
            return (self.latest_com7_data[0], self.latest_com7_data[1])
            
    # 新增：读取COM15浊度传感器数据（与COM7相同）
    def read_com15_data(self):
        try:
            if not self.ser_com15:
                return (self.latest_com15_data[0], self.latest_com15_data[1])
                
            dirt_value, ad_value = None, None
            cmd_dirt = bytes([0x18, 0x05, 0x00, 0x01, 0x0D])
            self.ser_com15.write(cmd_dirt)
            time.sleep(0.05)
            response = self.ser_com15.read(5)
            if len(response) == 5 and response[0] == 0x18 and response[-1] == 0x0D:
                dirt_value = response[3]

            cmd_ad = bytes([0x18, 0x05, 0x00, 0x02, 0x0D])
            self.ser_com15.write(cmd_ad)
            time.sleep(0.05)
            response_ad = self.ser_com15.read(6)
            if len(response_ad) == 6 and response_ad[0] == 0x18 and response_ad[-1] == 0x0D:
                ad_high = response_ad[3]
                ad_low = response_ad[4]
                ad_value = (ad_high << 8) | ad_low

            return (dirt_value if dirt_value is not None else self.latest_com15_data[0], 
                    ad_value if ad_value is not None else self.latest_com15_data[1])
        except Exception as e:
            print(f"COM15数据读取失败: {e}")
            return (self.latest_com15_data[0], self.latest_com15_data[1])

    def read_com3_data(self):
        try:
            data = self.ser_com3.read_until(b'$').decode('utf-8', 'ignore').strip()
            if not data.startswith('#') or '$' not in data:
                return None
            content = data[len('#'):data.index('$')]
            parts = content.split(',')
            num_list = []
            for part in parts[:2]:
                try:
                    num = float(part)
                    num_list.append(num)
                except ValueError:
                    num_list.append(-32768)
            return num_list
        except Exception as e:
            print(f"COM3数据解析失败: {e}")
            return None

    def read_com6_data(self):
        try:
            raw_values = self.instrument_com6.read_registers(
                registeraddress=0x0000,
                number_of_registers=2,
                functioncode=3
            )
            voltage1 = (raw_values[0] / AD_RESOLUTION) * VOLTAGE_REF
            return [raw_values[0], round(voltage1, 3)]
        except Exception as e:
            print(f"COM6数据读取失败: {e}")
            return None

    def serial_to_csv(self):
        pr = cProfile.Profile()
        pr.enable()
        
        # 备份逻辑保持不变
        backup_files = [f'sensor_data{channel}.csv' for channel in range(1,5)] + \
                      ['sensor_data.csv'] + \
                      [f'sensor_data6_{c}.csv' for c in range(1,5)]
        for filename in backup_files:
            if os.path.exists(filename):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{filename.split('.')[0]}_{timestamp}.csv"
                os.rename(filename, backup_name)
        
        csv_head = ['Timestamp', 'OD', 'electrical conductivity', 'fluorescence', 'Hum%RH', 'Dirt1', 'AD Value1', 'Dirt2', 'AD Value2']
        
        with open('sensor_data.csv', 'w', newline='') as f_merged:
            csv.writer(f_merged).writerow(csv_head)
        
        current_channel = None
        buffer = []
        merged_buffer = []
        n = 0
        last_save = time.time()
        
        print("\n初始化完成，等待10秒后开始采集数据...", flush=True)
        for i in range(10, 0, -1):
            if stop_event.is_set():
                print("\n检测到终止信号，程序退出。")
                return
            print(f"\r{i} 秒后开始采集...（按Ctrl+C可退出采集）", end='', flush=True)
            time.sleep(1)
        
        try:
            with open('sensor_data.csv', 'a', newline='') as f_merged:
                merged_writer = csv.writer(f_merged)
                
                while not stop_event.is_set():
                    if not command_queue.empty():
                        cmd = command_queue.get()
                        if isinstance(cmd, tuple):
                            if cmd[0] == 'channel':
                                new_channel = cmd[1]
                                if new_channel != current_channel:
                                    # 仅处理普通通道文件
                                    if self.current_normal_file:
                                        if len(buffer) > 0:
                                            csv.writer(self.current_normal_file).writerows(buffer)
                                            buffer.clear()
                                        self.current_normal_file.close()
                                    
                                    # 创建新通道文件（仅限普通通道）
                                    file_path = f'sensor_data{new_channel}.csv'
                                    if not os.path.exists(file_path):
                                        with open(file_path, 'w', newline='') as f_new:
                                            csv.writer(f_new).writerow(csv_head[:-1])
                                    self.current_normal_file = open(file_path, 'a', newline='')
                                    current_channel = new_channel
                                    print(f"\n切换到通道 {new_channel}，数据将保存至 {file_path}")

                            elif cmd[0] == 'channel6':
                                parent_channel = cmd[1]
                                file_path = f'sensor_data6_{parent_channel}.csv'
                                # 通道6文件单独管理
                                if file_path not in self.channel6_files:
                                    if not os.path.exists(file_path):
                                        with open(file_path, 'w', newline='') as f_new:
                                            csv.writer(f_new).writerow(csv_head[:-1])
                                    self.channel6_files[file_path] = open(file_path, 'a', newline='')
                                current_file = self.channel6_files[file_path]
                                current_writer = csv.writer(current_file)
                                print(f"\n切换到通道6（来自通道{parent_channel}），数据将保存至 {file_path}")
                                
                            elif cmd[0] == 'log':
                                self.last_log = cmd[1]
                        else:
                            self.ser_com3.write(str(cmd).encode())
                    
                    # 数据采集与写入逻辑保持不变
                    com3_data = self.read_com3_data()
                    com6_data = self.read_com6_data()
                    com7_data = self.read_com7_data()
                    com15_data = self.read_com15_data()  # 新增：读取COM15数据
                    with self.data_lock:
                        if com3_data and len(com3_data) == 2:
                            self.latest_com3_data = com3_data
                        if com6_data:
                            self.latest_com6_data = com6_data
                        self.latest_com7_data = com7_data
                        self.latest_com15_data = com15_data  # 新增：更新COM15数据
                    
                    merged_data = [
                        *self.latest_com3_data,
                        self.latest_com6_data[0],
                        self.latest_com6_data[1],
                        self.latest_com7_data[0],
                        self.latest_com7_data[1],
                        self.latest_com15_data[0],  # 新增：COM15的Dirt值
                        self.latest_com15_data[1]   # 新增：COM15的AD值
                    ]
                    
                    if len(merged_data) == 8:  # 更新为8个数据点
                        timestamp = datetime.now().isoformat(timespec='milliseconds')
                        if current_channel is not None:
                            row_channel = [timestamp] + merged_data
                            buffer.append(row_channel)
                        row_merged = [timestamp] + merged_data + [self.last_log]
                        merged_buffer.append(row_merged)
                        self.last_log = ""
                        
                        try:
                            data_queue.put(row_merged.copy(), block=False, timeout=0.1)
                        except queue.Full:
                            pass
                        n += 1
                        
                        # 写入逻辑优化
                        if len(buffer) >= 10:
                            if current_channel in VALVE_CHANNELS and self.current_normal_file:
                                csv.writer(self.current_normal_file).writerows(buffer)
                                buffer.clear()
                            elif current_channel == 6 and current_file:
                                current_writer.writerows(buffer)
                                buffer.clear()
                        
                        if len(merged_buffer) >= 10:
                            merged_writer.writerows(merged_buffer)
                            merged_buffer.clear()
                            print(f'\r已保存: {row_merged} (总计: {n})', end='', flush=True)
                            
                    # 定期刷新缓冲区
                    if time.time() - last_save > 10:
                        if self.current_normal_file:
                            self.current_normal_file.flush()
                        for f in self.channel6_files.values():
                            f.flush()
                        f_merged.flush()
                        last_save = time.time()
                    
                    if n >= 144000:
                        break
            
            pr.disable()
            s = io.StringIO()
            stats = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            stats.print_stats(20)
            print("\n性能分析结果:\n", s.getvalue())
            
        finally:
            # 关闭所有文件句柄
            if self.current_normal_file:
                if len(buffer) > 0:
                    csv.writer(self.current_normal_file).writerows(buffer)
                self.current_normal_file.close()
            for f in self.channel6_files.values():
                if not f.closed:
                    f.close()
            self.ser_com3.close()
            self.instrument_com6.serial.close()
            self.ser_com7.close()
            if self.ser_com15:
                self.ser_com15.close()

# ------------------ 绘图模块 -------------------
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    lines = [
        ax1.plot([], [], 'g-', label='OD')[0],
        ax2.plot([], [], 'b-', label='Fluorescence')[0],
        ax2.plot([], [], 'r-', label='Humidity')[0]
    ]
    timestamps, od_values, fluo_values, hum_values = [], [], [], []
    
    def init():
        for ax in [ax1, ax2]:
            ax.set_xlim(datetime.now(), datetime.now() + timedelta(minutes=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.set_ylabel('OD Value')
        ax2.set_ylabel('Fluorescence/Humidity')
        ax2.legend(loc='upper right')
        return lines
    
    def update(frame):
        nonlocal od_values, fluo_values, hum_values
        if stop_event.is_set():
            plt.close('all')
            return
        while not data_queue.empty():
            data = data_queue.get()
            try:
                ts = datetime.fromisoformat(data[0])
                timestamps.append(ts)
                od_values.append(data[2])
                fluo_values.append(data[3])
                hum_values.append(data[5])
            except Exception as e:
                print(f"绘图数据错误: {e}")
        
        lines[0].set_data(timestamps, od_values)
        lines[1].set_data(timestamps, fluo_values)
        lines[2].set_data(timestamps, hum_values)
        
        for line in lines:
            line.axes.relim()
            line.axes.autoscale_view()
        
        return lines
    
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=500)
    plt.tight_layout()
    plt.show()

# ------------------ 主程序 -------------------
if __name__ == "__main__":
    # 打印可用串口列表
    print("可用串口列表:")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f" - {port.device}")
    
    def main():
        controller = MainController()
        collector = DataCollector()
        
        controller_thread = threading.Thread(target=controller.main_loop)
        data_thread = threading.Thread(target=collector.serial_to_csv)
        
        try:
            controller_thread.start()
            data_thread.start()
            
            plot_data()
            
            while not stop_event.is_set():
                if keyboard.is_pressed('esc'):
                    print("\n检测到ESC按键，正在停止...")
                    stop_event.set()
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n检测到Ctrl+C，正在停止...")
            stop_event.set()
        finally:
            controller_thread.join(timeout=2)
            data_thread.join(timeout=2)
            NewPumpController.pump1.safe_shutdown()
            NewPumpController.pump2.safe_shutdown()
            print("\n程序已安全退出")

    cProfile.runctx('main()', globals(), locals(), filename='performance.prof')