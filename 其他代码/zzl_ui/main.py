from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QFormLayout
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QPixmap, QImage
import io
from psd_function_ui import calculate_snr, gaussian_psd


# 决策参数设置界面
class MyWindow1(QMainWindow):
    def __init__(self):
        super(MyWindow1, self).__init__()
        loadUi('control_Ui.ui', self)


# 决策参数展示界面
class MyWindow2(QWidget):
    def __init__(self):
        super(MyWindow2, self).__init__()
        self.value = 0
        self.frequency_range = np.linspace(140, 260, 12000)
        self.sinr = 0
        self.sinr_before = 0
        loadUi('display_Ui_v4.ui', self)

        # 创建 FigureCanvas 对象
        self.canvas = FigureCanvas(Figure(figsize=(600 / 80, 280 / 80), dpi=80))

        # 将 FigureCanvas 的图形设置给 QLabel
        self.label_10.setPixmap(self.canvas_to_pixmap())

        # 创建一个定时器，每秒触发一次更新数值的方法
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_value)
        self.timer.timeout.connect(self.update_plot)

        # 将按钮的点击事件连接到 click 方法
        self.pushButton.clicked.connect(self.click)

    def click(self):
        # 点击按钮时启动或停止定时器
        if not self.timer.isActive():
            self.timer.start(1000)  # 1000毫秒（1秒）触发一次
        else:
            self.timer.stop()

    def update_value(self):
        # 这个方法会在定时器触发时调用，更新 QLabel 显示的数值
        self.value += 1
        vs = self.value % 5
        # 隐真框 分别是频点、带宽、功率
        self.label_13.setText(str(180 + (self.value * 0.8)))
        self.label_14.setText(str(6 + vs))
        self.label_15.setText(str(20))
        # 示假框 分别是频点、带宽、功率
        # self.label_16.setText(str(160 + (self.value * 0.8)))
        # self.label_17.setText(str(6 + vs))
        # self.label_18.setText(str(20))

        # 干扰框 分别是频点、带宽、功率
        self.label_19.setText(str(179 + (self.value * 0.8)))
        self.label_20.setText(str(5 + vs))
        self.label_21.setText(str(30))
        # 信噪比设置 分别是当前时刻和上一时刻
        self.label_24.setText(str(self.sinr))
        self.label_25.setText(str(self.sinr_before))

    def update_plot(self):
        # 在这里编写每一帧的 matplotlib 绘图逻辑
        x = self.frequency_range
        vs = self.value % 5
        out = calculate_snr(180 + (self.value * 0.8), 6 + vs, 20,
                            179 + (self.value * 0.8), 5 + vs, 30,
                            160 + (self.value * 0.8), 6 + vs, 20, self.frequency_range)

        # 单独计算每种信号的功率谱数值
        # y_signal = gaussian_psd(self.frequency_range, 180 + (self.value * 0.8), 6 + vs, 20)
        # y_jamming = gaussian_psd(self.frequency_range, 179 + (self.value * 0.8), 5 + vs, 30)
        # y_fake = gaussian_psd(self.frequency_range, 160 + (self.value * 0.8), 6 + vs, 20)

        self.sinr_before = self.sinr
        self.sinr = round(out[0][0], 2)
        y_signal = out[1][0]
        y_jamming = out[2][0]
        y_fake = out[3][0]

        # 清除旧图形，绘制新图形
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(x, y_signal, label='Signal Power Spectrum', color='blue')
        ax.plot(x, y_jamming, label='Jamming Power Spectrum', color='orange')
        ax.plot(x, y_fake, label='fake Power Spectrum', color='green')

        # 设置 x 坐标范围
        ax.set_xlim([140, 260])

        # 显示图例
        ax.legend()

        # 更新画布
        self.canvas.draw()

        # 将 FigureCanvas 的图形设置给 QLabel
        self.label_10.setPixmap(self.canvas_to_pixmap())

    def canvas_to_pixmap(self):
        # 将 FigureCanvas 的图形转为 QPixmap
        buf = io.BytesIO()
        self.canvas.print_png(buf)
        data = buf.getvalue()
        qimage = QImage.fromData(data)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def decision_algorithm(self):



        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置缩放因子，没改好
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    # app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    main_window1 = MyWindow1()
    main_window1.show()

    main_window2 = MyWindow2()
    main_window2.show()
    sys.exit(app.exec_())
