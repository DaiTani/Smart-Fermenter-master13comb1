"""
程序设定向导模块
多步骤的配置向导界面
"""
from PyQt5.QtWidgets import (QMainWindow, QStackedWidget, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QMessageBox, QInputDialog)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from window_selector import WindowSelector
from region_selector import RegionSelector
from region_selector2 import RegionSelector2
from operation_editor import OperationEditor
from data_source_selector import DataSourceSelector
from config_manager import save_operation_config
from window_manager import WindowManager
from execution_window import ExecutionWindow
import pygetwindow as gw
import threading
import time

class SettingsWizard(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("程序设定向导")
        self.setMinimumSize(1000, 700)
        self.setup_ui()
        
        # 新增成员变量，用于控制窗口置顶
        self._keep_window_on_top = False
        self._on_top_thread = None
    
    def setup_ui(self):
        """初始化向导界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("程序设定向导")
        title_label.setFont(QFont("微软雅黑", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ecf0f1; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 步骤指示器
        self.step_indicator = QLabel("步骤 1/7: 选择目标窗口")
        self.step_indicator.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.step_indicator.setAlignment(Qt.AlignCenter)
        # 步骤指示器美化
        self.step_indicator.setStyleSheet("background:#eaf6fb; color:#2980b9; border-radius:8px; padding:8px 0; margin-bottom:10px; font-size:16px;")
        main_layout.addWidget(self.step_indicator)
        
        # 堆叠窗口用于不同步骤
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 创建步骤 - 按照新顺序创建
        self.create_steps()
        
        # 导航按钮
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(18)

        self.prev_btn = QPushButton("← 上一步")
        self.prev_btn.setIcon(QIcon("icons/prev.png"))
        self.prev_btn.setMinimumHeight(38)
        self.prev_btn.setToolTip("返回上一步")
        self.prev_btn.clicked.connect(self.prev_step)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一步 →")
        self.next_btn.setIcon(QIcon("icons/next.png"))
        self.next_btn.setMinimumHeight(38)
        self.next_btn.setToolTip("进入下一步")
        self.next_btn.clicked.connect(self.next_step)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setIcon(QIcon("icons/cancel.png"))
        self.cancel_btn.setMinimumHeight(38)
        self.cancel_btn.setStyleSheet("background-color: #e74c3c;")
        self.cancel_btn.setToolTip("取消并返回主界面")
        self.cancel_btn.clicked.connect(self.cancel_wizard)

        self.test_operations_btn = QPushButton("测试操作逻辑")
        self.test_operations_btn.setIcon(QIcon("icons/test.png"))
        self.test_operations_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.test_operations_btn.clicked.connect(self.test_operations)
        self.test_operations_btn.setVisible(False)

        self.skip_data_source_btn = QPushButton("跳过数据源配置")
        self.skip_data_source_btn.setIcon(QIcon("icons/skip.png"))
        self.skip_data_source_btn.setStyleSheet("background-color: #8e44ad; color: white;")
        self.skip_data_source_btn.clicked.connect(self.skip_data_source)
        self.skip_data_source_btn.setVisible(False)
        
        self.test_operations_btn2 = QPushButton("测试操作逻辑2")
        self.test_operations_btn2.setIcon(QIcon("icons/test.png"))
        self.test_operations_btn2.setStyleSheet("background-color: #f39c12; color: white;")
        self.test_operations_btn2.clicked.connect(self.test_operations2)
        self.test_operations_btn2.setVisible(False)

        self.skip_data_source_btn2 = QPushButton("跳过数据源配置2")
        self.skip_data_source_btn2.setIcon(QIcon("icons/skip.png"))
        self.skip_data_source_btn2.setStyleSheet("background-color: #8e44ad; color: white;")
        self.skip_data_source_btn2.clicked.connect(self.skip_data_source2)
        self.skip_data_source_btn2.setVisible(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        nav_layout.addWidget(self.test_operations_btn)
        nav_layout.addWidget(self.skip_data_source_btn)
        nav_layout.addWidget(self.test_operations_btn2)
        nav_layout.addWidget(self.skip_data_source_btn2)
        nav_layout.addWidget(self.next_btn)
        
        main_layout.addWidget(nav_widget)
        
        # 当前步骤索引
        self.current_step = 0
    
    def create_steps(self):
        """创建所有设定步骤 - 按照新顺序"""
        # 步骤1: 窗口选择
        self.window_selector = WindowSelector()
        self.stacked_widget.addWidget(self.window_selector)

        # 步骤2: 数据源选择
        self.data_source_selector = DataSourceSelector()
        self.stacked_widget.addWidget(self.data_source_selector)

        # 步骤3: 操作编辑
        self.operation_editor = OperationEditor()
        self.stacked_widget.addWidget(self.operation_editor)

        # 步骤4: 框选界面元素
        self.region_selector = RegionSelector()
        self.stacked_widget.addWidget(self.region_selector)
        
        # 步骤5: 数据源选择2
        self.data_source_selector2 = DataSourceSelector()
        self.stacked_widget.addWidget(self.data_source_selector2)
        
        # 步骤6: 操作编辑2
        self.operation_editor2 = OperationEditor()
        self.stacked_widget.addWidget(self.operation_editor2)
        
        # 步骤7: 框选界面元素2
        self.region_selector2 = RegionSelector2()
        self.stacked_widget.addWidget(self.region_selector2)
    
    def prev_step(self):
        """返回到上一步"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step()
    
    def get_regions_from_operations(self, operation_editor):
        """从操作序列中提取所有唯一的区域"""
        regions = []
        for operation in operation_editor.get_operations():
            region = operation.get("region")
            if region and region not in regions:
                regions.append(region)
        return regions
    
    def next_step(self):
        """前进到下一步"""
        if self.validate_current_step():
            if self.current_step == 0:
                # 步骤1: 选择目标窗口后进入步骤2
                pass
            elif self.current_step == 1:
                # 数据源选择后进入操作编辑
                self.operation_editor.set_data_source(self.data_source_selector.get_data_source())
            elif self.current_step == 2:
                # 操作编辑后进入框选界面元素
                self.region_selector.set_imported_regions(
                    self.get_regions_from_operations(self.operation_editor)
                )
                # 设置窗口信息并激活目标窗口
                window_title = self.window_selector.get_selected_window()
                if window_title:
                    self.region_selector.set_window_info(window_title)
                    try:
                        win = gw.getWindowsWithTitle(window_title)
                        if win:
                            win[0].activate()
                            # 启动置顶线程（只在第4步持续置顶）
                            self._start_keep_on_top_thread(window_title)
                    except Exception:
                        pass
            elif self.current_step == 3:
                # 框选界面元素后进入数据源选择2
                pass
            elif self.current_step == 4:
                # 数据源选择2后进入操作编辑2
                self.operation_editor2.set_data_source(self.data_source_selector2.get_data_source())
            elif self.current_step == 5:
                # 操作编辑2后进入框选界面元素2
                self.region_selector2.set_imported_regions(
                    self.get_regions_from_operations(self.operation_editor2)
                )
                # 设置窗口信息并激活目标窗口
                window_title = self.window_selector.get_selected_window()
                if window_title:
                    self.region_selector2.set_window_info(window_title)
                    try:
                        win = gw.getWindowsWithTitle(window_title)
                        if win:
                            win[0].activate()
                    except Exception:
                        pass
            elif self.current_step == 6:
                self.complete_setup()
                return

            self.current_step += 1
            self.update_step()

    def validate_current_step(self):
        """验证当前步骤是否完成"""
        if self.current_step == 0:
            return self.window_selector.is_valid()
        elif self.current_step == 1:
            return self.data_source_selector.is_valid()
        elif self.current_step == 2:
            return self.operation_editor.is_valid()
        elif self.current_step == 3:
            return self.region_selector.is_valid()
        elif self.current_step == 4:
            return self.data_source_selector2.is_valid()
        elif self.current_step == 5:
            return self.operation_editor2.is_valid()
        elif self.current_step == 6:
            return self.region_selector2.is_valid()
        return True
    
    def update_step(self):
        """更新UI以显示当前步骤"""
        self.stacked_widget.setCurrentIndex(self.current_step)

        # 更新步骤指示器 - 按照新顺序
        step_names = [
            "步骤 1/7: 选择目标窗口",
            "步骤 2/7: 选择数据源",
            "步骤 3/7: 编辑操作逻辑",
            "步骤 4/7: 框选界面元素",
            "步骤 5/7: 选择数据源2",
            "步骤 6/7: 编辑操作逻辑2",
            "步骤 7/7: 框选界面元素2"
        ]
        self.step_indicator.setText(step_names[self.current_step])

        # 更新按钮状态
        self.prev_btn.setEnabled(self.current_step > 0)
        self.next_btn.setText("完成" if self.current_step == 6 else "下一步")
        self.test_operations_btn.setVisible(self.current_step == 2)
        self.skip_data_source_btn.setVisible(self.current_step == 1)
        self.test_operations_btn2.setVisible(self.current_step == 5)
        self.skip_data_source_btn2.setVisible(self.current_step == 4)
        
        # 每次进入第四步和第七步都刷新区域选择器的窗口截图
        if self.current_step == 3:  # 步骤4: 框选界面元素
            window_title = self.window_selector.get_selected_window()
            if window_title:
                self.region_selector.set_window_info(window_title)
        elif self.current_step == 6:  # 步骤7: 框选界面元素2
            window_title = self.window_selector.get_selected_window()
            if window_title:
                self.region_selector2.set_window_info(window_title)
        
        # 只在第四步（框选）时保持目标窗口置顶
        if self.current_step != 3:
            self._keep_window_on_top = False
    
    def complete_setup(self):
        """完成设置并保存配置"""
        # 收集配置数据
        window_title = self.window_selector.get_selected_window()
        
        # 创建第一个配置（步骤1-4）
        config1 = {
            "window_title": window_title,
            "regions": self.region_selector.get_regions(),
            "data_source": self.data_source_selector.get_data_source(),
            "operations": self.operation_editor.get_operations()
        }
        
        # 区域匹配更新
        region_map1 = {r["name"]: r for r in config1["regions"]}
        
        for op in config1["operations"]:
            region_name = op["region"].get("name")
            if region_name in region_map1:
                # 用实际区域替换虚拟区域
                op["region"] = region_map1[region_name]
        
        # 创建第二个配置（步骤1,5-7）
        config2 = {
            "window_title": window_title,
            "regions": self.region_selector2.get_regions(),
            "data_source": self.data_source_selector2.get_data_source(),
            "operations": self.operation_editor2.get_operations()
        }
        
        # 区域匹配更新
        region_map2 = {r["name"]: r for r in config2["regions"]}
        
        for op in config2["operations"]:
            region_name = op["region"].get("name")
            if region_name in region_map2:
                # 用实际区域替换虚拟区域
                op["region"] = region_map2[region_name]
        
        # 保存第一个配置
        default_name1 = f"配置_{len(config1['regions'])}元素"
        config_name1, ok1 = QInputDialog.getText(self, "保存配置", "请输入配置名称：", text=default_name1)
        if not ok1 or not config_name1.strip():
            QMessageBox.warning(self, "未保存", "未输入配置名称，配置未保存。")
            return
        
        config_name1 = config_name1.strip()
        
        if save_operation_config(config_name1, config1):
            QMessageBox.information(self, "设置完成", f"配置已保存: {config_name1}")
        else:
            QMessageBox.warning(self, "保存失败", "配置保存失败，请重试")
            return
        
        # 保存第二个配置
        default_name2 = f"{config_name1}_2"
        config_name2, ok2 = QInputDialog.getText(
            self, 
            "保存第二个配置", 
            "请输入第二个配置的名称：", 
            text=default_name2
        )
        
        config_name2 = config_name2.strip() if ok2 and config_name2.strip() else default_name2
        
        if save_operation_config(config_name2, config2):
            QMessageBox.information(self, "设置完成", f"第二个配置已保存: {config_name2}")
        else:
            QMessageBox.warning(self, "保存失败", "第二个配置保存失败，请重试")
        
        # 将第二个配置文件复制为ocr_config.json
        try:
            import shutil
            import os
            import time
            from datetime import datetime
            
            # 配置文件路径
            config_dir = "configs"
            config2_path = os.path.join(config_dir, f"{config_name2}.json")
            
            # 目标文件路径
            ocr_config_path = "ocr_config.json"
            
            # 检查源文件是否存在
            if not os.path.exists(config2_path):
                QMessageBox.warning(self, "文件不存在", f"无法找到配置文件: {config2_path}")
                return
                
            # 检查目标文件是否存在
            if os.path.exists(ocr_config_path):
                # 创建备份文件名（带时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"ocr_config_{timestamp}.json"
                os.rename(ocr_config_path, backup_path)
                QMessageBox.information(self, "备份成功", f"已将原ocr_config.json备份为: {backup_path}")
            
            # 复制文件
            shutil.copyfile(config2_path, ocr_config_path)
            QMessageBox.information(self, "复制成功", f"已将 {config_name2}.json 复制为 ocr_config.json")
        except Exception as e:
            QMessageBox.warning(self, "复制出错", f"复制OCR配置文件时出错: {str(e)}\n请手动将文件复制到根目录。")
        
        WindowManager().return_to_main()
    
    def cancel_wizard(self):
        """取消向导并返回主窗口"""
        WindowManager().return_to_main()
    
    def test_operations(self):
        """无需数据源直接测试操作逻辑"""
        config = {
            "window_title": self.window_selector.get_selected_window(),
            "regions": self.region_selector.get_regions() if hasattr(self.region_selector, 'get_regions') else [],
            "data_source": None,
            "operations": self.operation_editor.get_operations()
        }
        config_name = f"测试_{len(config.get('operations', []))}步骤"
        exec_win = ExecutionWindow(config_name)
        # 直接传递当前配置并覆盖load_config方法，使其不加载默认配置
        exec_win.config = config
        def fake_load_config():
            info_text = (
                f"窗口标题: {config.get('window_title', '无')}\n"
                f"操作步骤数量: {len(config.get('operations', []))}\n"
                f"循环次数: 1\n循环间隔: 0秒"
            )
            exec_win.config_info.setText(info_text)
            exec_win.progress_bar.setMaximum(len(config.get('operations', [])))
        exec_win.load_config = fake_load_config
        exec_win.load_config()
        exec_win.show()
    
    def test_operations2(self):
        """无需数据源直接测试操作逻辑2"""
        config = {
            "window_title": self.window_selector.get_selected_window(),
            "regions": self.region_selector2.get_regions() if hasattr(self.region_selector2, 'get_regions') else [],
            "data_source": None,
            "operations": self.operation_editor2.get_operations()
        }
        config_name = f"测试2_{len(config.get('operations', []))}步骤"
        exec_win = ExecutionWindow(config_name)
        # 直接传递当前配置并覆盖load_config方法，使其不加载默认配置
        exec_win.config = config
        def fake_load_config():
            info_text = (
                f"窗口标题: {config.get('window_title', '无')}\n"
                f"操作步骤数量: {len(config.get('operations', []))}\n"
                f"循环次数: 1\n循环间隔: 0秒"
            )
            exec_win.config_info.setText(info_text)
            exec_win.progress_bar.setMaximum(len(config.get('operations', [])))
        exec_win.load_config = fake_load_config
        exec_win.load_config()
        exec_win.show()
    
    def skip_data_source(self):
        """跳过数据源配置，直接进入操作逻辑编辑"""
        # 跳过时，data_source 设为 None，直接进入下一步
        self.current_step = 2  # 跳过数据源后进入操作编辑
        self.update_step()
        
    def skip_data_source2(self):
        """跳过数据源配置2，直接进入操作逻辑编辑2"""
        # 跳过时，data_source2 设为 None，直接进入下一步
        self.current_step = 5  # 跳过数据源2后进入操作编辑2
        self.update_step()

    def _start_keep_on_top_thread(self, window_title):
        """
        后台线程定时将目标窗口置顶，仅在框选阶段（第四步）才置顶
        """
        self._keep_window_on_top = True

        def keep_on_top():
            while self._keep_window_on_top and self.current_step == 3:  # 只在步骤4置顶
                try:
                    win = gw.getWindowsWithTitle(window_title)
                    if win:
                        try:
                            win[0].activate()
                        except Exception:
                            pass
                        try:
                            win[0].bringToFront()
                        except Exception:
                            pass
                        try:
                            win[0].set_foreground()
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.5)
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._keep_window_on_top = False
            self._on_top_thread.join()
        self._on_top_thread = threading.Thread(target=keep_on_top, daemon=True)
        self._on_top_thread.start()

    def closeEvent(self, event):
        """关闭时停止置顶线程"""
        self._keep_window_on_top = False
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._on_top_thread.join(timeout=1)
        super().closeEvent(event)