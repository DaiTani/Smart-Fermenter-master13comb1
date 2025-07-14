import serial
import time

def calculate_crc(data):
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

def write_register(ser, address, reg_addr, reg_value):
    """
    写入单个寄存器（功能码 0x06）
    返回 True 表示成功，False 表示失败
    """
    function_code = 0x06
    # 构建请求帧
    data = bytes([address, function_code]) + reg_addr.to_bytes(2, 'big') + reg_value.to_bytes(2, 'big')
    crc = calculate_crc(data)
    crc_bytes = [(crc & 0xFF), (crc >> 8) & 0xFF]  # CRC 低字节在前
    frame = data + bytes(crc_bytes)
    
    # 发送请求
    ser.write(frame)
    time.sleep(0.2)  # 增加等待时间
    
    # 读取响应（最大读取长度为 100 字节）
    response = ser.read(100)
    if not response:
        print("未收到响应")
        return False
    
    print(f"原始响应数据: {response.hex()}")  # 调试信息
    
    # 响应长度检查（功能码0x06的响应应为8字节）
    if len(response) < 8:
        print(f"响应长度不足: {len(response)}")
        return False
    
    # 解析响应
    resp_address = response[0]
    resp_func = response[1]
    resp_reg = int.from_bytes(response[2:4], 'big')
    resp_val = int.from_bytes(response[4:6], 'big')
    received_crc = int.from_bytes(response[6:8], 'little')  # CRC 低字节在前
    
    # 重新计算 CRC
    expected_crc = calculate_crc(response[:6])
    expected_crc_low = expected_crc & 0xFF
    expected_crc_high = (expected_crc >> 8) & 0xFF
    
    # 验证响应内容
    if (resp_address == address and 
        resp_func == function_code and 
        resp_reg == reg_addr and 
        resp_val == reg_value and 
        received_crc == ((expected_crc_high << 8) | expected_crc_low)):
        return True
    else:
        print("响应校验失败")
        return False

def main():
    try:
        # 串口配置（关键修复：停止位改为2位）
        ser = serial.Serial(
            port='COM13',          # 修改为实际串口号
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_TWO,  # 修复：2位停止位
            parity=serial.PARITY_NONE,
            timeout=1
        )
    except serial.SerialException as e:
        print(f"串口打开失败: {e}")
        return

    PUMP_ADDRESS = 0x01  # 从机地址（拨码开关设置为1）
    RUN_REGISTER = 0x0000
    DIRECTION_REGISTER = 0x0001
    SPEED_REGISTER = 0x0002

    current_state = 0  # 0:停止，1:运行

    print("按下回车键控制蠕动泵（Ctrl+C 退出）")
    try:
        while True:
            input()
            current_state ^= 1
            if current_state == 1:
                # 设置正转 + 速度1转/分钟（0x000A对应10*0.1转）
                success = write_register(ser, PUMP_ADDRESS, DIRECTION_REGISTER, 0x0001)
                if not success:
                    print("方向设置失败")
                    continue
                success = write_register(ser, PUMP_ADDRESS, SPEED_REGISTER, int(2 * 10))
                if not success:
                    print("速度设置失败")
                    continue
                success = write_register(ser, PUMP_ADDRESS, RUN_REGISTER, 0x0001)
                if success:
                    print("电机已启动 - 正转，速度1转/分钟")
                else:
                    print("启动失败")
            else:
                success = write_register(ser, PUMP_ADDRESS, RUN_REGISTER, 0x0000)
                if success:
                    print("电机已停止")
                else:
                    print("停止失败")
    except KeyboardInterrupt:
        print("\n程序终止")
        ser.close()

if __name__ == '__main__':
    main()