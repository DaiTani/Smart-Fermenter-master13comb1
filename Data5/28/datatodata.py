import pandas as pd
import time
import os

def append_next_row():
    original_file = "data.xlsx"
    copy_file = "data - 副本.xlsx"
    sheet_name = "Data"
    
    # 读取目标文件获取最后时间戳
    if os.path.exists(original_file):
        try:
            df_original = pd.read_excel(original_file, sheet_name=sheet_name)
            last_timestamp = df_original["Timestamp"].iloc[-1]
        except Exception as e:
            print(f"读取目标文件出错: {e}")
            return False
    else:
        # 文件不存在时初始化
        #df_original = pd.DataFrame()
        #last_timestamp = None
        print(f"读取目标文件出错")
    # 读取副本文件
    try:
        df_copy = pd.read_excel(copy_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"读取副本文件出错: {e}")
        return False

    # 查找下一行数据
    if last_timestamp is None:
        # 目标文件为空时取第一行
        next_row = df_copy.iloc[0]
    else:
        # 找到副本中比最后时间戳大的第一行
        next_rows = df_copy[df_copy["Timestamp"] > last_timestamp]
        if next_rows.empty:
            print("没有更多数据可复制")
            return False
        next_row = next_rows.iloc[0]

    # 追加数据到目标文件
    new_row_df = pd.DataFrame([next_row])
    # 写入重试机制
    max_retries = 100  # 最大重试次数（实际可能无限重试）
    retry_count = 0
    
    while retry_count < max_retries:
        try:
                with pd.ExcelWriter(
                    original_file, 
                    engine="openpyxl", 
                    mode="a" if os.path.exists(original_file) else "w",
                    if_sheet_exists="overlay"
                ) as writer:
                    new_row_df.to_excel(
                        writer, 
                        sheet_name=sheet_name, 
                        index=False,
                        header=False,
                        startrow=writer.sheets[sheet_name].max_row
                    )
                print(f"已添加时间戳: {next_row['Timestamp']}")
                return True
        except PermissionError as e:
            print(f"文件被占用，等待1秒后重试... (尝试次数: {retry_count+1})")
            time.sleep(1)  # 等待1秒
            retry_count += 1
        except Exception as e:
            print(f"写入文件出错: {e}")
            return False
    print(f"达到最大重试次数({max_retries})，写入失败")
    return False
# 主循环
while True:
    try:
        result = append_next_row()
        if not result:
            print("复制过程终止")
            break
    except Exception as e:
        print(f"发生错误: {e}")
    
    # 等待10秒
    time.sleep(10)