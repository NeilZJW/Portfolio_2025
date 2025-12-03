# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/3/15 20:20

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QVBoxLayout, QWidget, QHBoxLayout, QScrollArea)
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt
import sys
import main
import time


class HairStyleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_path = {"style": "", "color": ""}
        self.selectedColor = QColor('black')  # 默认颜色
        self.selectedHairStyle = None
        self.selectedHairStyleButton = None
        self.selectedColorButton = None
        self.userImagePath = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AI-Barbershop')
        self.setGeometry(500, 100, 1000, 800)

        global mainLayout
        mainLayout = QVBoxLayout()

        # Create a horizontal layout for the two image labels
        imageLayout = QHBoxLayout()

        self.userImageLabel1 = QLabel(self)
        self.userImageLabel1.setFixedSize(400, 400)
        imageLayout.addWidget(self.userImageLabel1)

        self.userImageLabel2 = QLabel(self)
        self.userImageLabel2.setFixedSize(400, 400)
        imageLayout.addWidget(self.userImageLabel2)

        mainLayout.addLayout(imageLayout)

        self.uploadUserImageButton = QPushButton('Upload now!', self)
        self.uploadUserImageButton.clicked.connect(lambda: self.uploadUserImage(self.userImageLabel1))
        mainLayout.addWidget(self.uploadUserImageButton)

        self.anotherUploadButton = QPushButton('Second upload! What would you want to be?', self)  # 按钮文本可以根据您的需要修改
        self.anotherUploadButton.clicked.connect(lambda: self.uploadUserImage(self.userImageLabel2))  # 连接到同一个槽函数
        mainLayout.addWidget(self.anotherUploadButton)

        self.hairStyleScrollArea = QScrollArea(self)
        self.hairStyleScrollArea.setWidgetResizable(True)
        self.hairStyleScrollAreaWidgetContents = QWidget()
        self.hairStyleLayout = QHBoxLayout(self.hairStyleScrollAreaWidgetContents)
        self.hairStyleScrollArea.setWidget(self.hairStyleScrollAreaWidgetContents)
        mainLayout.addWidget(self.hairStyleScrollArea)

        hairStylePaths = ["./input/face/style/s1.png", "./input/face/style/s2.png",
                          "./input/face/style/s3.png", "./input/face/style/s4.png",
                          "./input/face/style/s5.png", "./input/face/style/s6.png"]
        for path in hairStylePaths:
            self.addHairStyleImage(path)

        colorPaths = ["./input/face/color/c1", "./input/face/color/c2",
                      "./input/face/color/c3", "./input/face/color/c4",
                      "./input/face/color/c5"]
        colorOptionsLayout = QHBoxLayout()
        self.colors = {'1': QColor('#1A171E'), '2': QColor('#B0815E'), '3': QColor('#A07544'), '4': QColor('#A5555A'),
                       '5': QColor('#AF643C')}
        for colorName, colorValue in self.colors.items():
            colorButton = QPushButton()
            colorButton.setStyleSheet(f"background-color: {colorValue.name()}; border: 1px solid {colorValue.name()};")
            colorButton.colorValue = colorValue  # 保存颜色值到按钮
            colorButton.clicked.connect(lambda _, col=colorValue: self.selectColor(col))
            colorOptionsLayout.addWidget(colorButton)
            colorButton.setFixedSize(40, 40)
        mainLayout.addLayout(colorOptionsLayout)


        self.applyButton = QPushButton('APPLY', self)
        self.applyButton.clicked.connect(self.applyHairStyle)
        mainLayout.addWidget(self.applyButton)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def addHairStyleImage(self, imagePath):
        hairStyleButton = QPushButton(self.hairStyleScrollAreaWidgetContents)
        pixmap = QPixmap(imagePath).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon = QIcon(pixmap)
        hairStyleButton.setIcon(icon)
        hairStyleButton.setIconSize(pixmap.rect().size())
        hairStyleButton.clicked.connect(self.selectHairStyle)
        hairStyleName = imagePath.split('/')[-1].split('.')[0]
        hairStyleButton.hairStyleName = hairStyleName
        self.hairStyleLayout.addWidget(hairStyleButton)

    def selectHairStyle(self):
        sender = self.sender()
        if self.selectedHairStyleButton:
            self.selectedHairStyleButton.setStyleSheet("")
        self.selectedHairStyleButton = sender
        self.selectedHairStyle = sender.hairStyleName
        sender.setStyleSheet("border: 2px solid blue")
        self.updateSelection()

    def uploadUserImage(self, imageLabel):
        fname, _ = QFileDialog.getOpenFileName(self, 'Choose a image', '/', 'Format of image (*.png *.jpg *.jpeg)')
        if fname:
            self.userImagePath.append(fname)
            imageLabel.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(self.userImagePath[-1])
            imageLabel.setPixmap(
                pixmap.scaled(imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def selectColor(self, color):
        sender = self.sender()
        if self.selectedColorButton:
            # 将之前选中的按钮样式重置
            prevColor = self.selectedColorButton.colorValue
            self.selectedColorButton.setStyleSheet(
                f"background-color: {prevColor.name()}; border: 1px solid {prevColor.name()};")
        self.selectedColorButton = sender
        self.selectedColor = color
        # 更新当前选中的按钮样式以反映选中状态
        sender.setStyleSheet(f"background-color: {color.name()}; border: 2px solid blue; padding: 2px;")

        self.updateSelection()

    def applyHairStyle(self):
        # 清空当前界面
        self.clearLayout(mainLayout)
        print("OK")
        # 显示指定路径的图片
        if len(self.userImagePath) == 1:
            output_name = self.userImagePath[0].split("/")[-1].split(".")[0]
            user_input = self.userImagePath[0].split("/")[-1]
            structure_input = self.input_path["style"] + ".png"
            color_input = self.input_path["color"] + ".png"
            print(output_name, user_input, structure_input, color_input)
            start = time.clock()
            args = main.add_parser(
                user_input, structure_input, color_input, output_name
            ).parse_args()
            main.main(args)
            end = time.clock()
            print(f" Total time - {float(end - start)} ".center(100, "*"))
            result_path = './output/{}/{}_{}_{}_realistic.png'.format(
                output_name, output_name,
                self.input_path["style"].split("/")[-1],
                self.input_path["color"].split("/")[-1]
            )
            self.displayImage(result_path)
        else:
            output_name = self.userImagePath[0].split("/")[-1].split(".")[0]
            user_input = self.userImagePath[0].split("/")[-1]
            structure_input = self.userImagePath[1].split("/")[-1]
            color_input = self.userImagePath[1].split("/")[-1]
            print(output_name, user_input, structure_input, color_input)
            start = time.clock()
            args = main.add_parser(
                user_input, structure_input, color_input, output_name
            ).parse_args()
            main.main(args)
            end = time.clock()
            print(f" Total time - {float(end - start)} ".center(100, "*"))
            result_path = './output/{}/{}_{}_{}_realistic.png'.format(
                output_name, output_name,
                structure_input.split(".")[0],
                color_input.split(".")[0]
            )
            print(result_path)
            self.displayImage(result_path)


    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clearLayout(child.layout())

    def displayImage(self, image_path):
        # 创建一个标签并设置图片
        imageLabel = QLabel(self)
        pixmap = QPixmap(image_path)
        imageLabel.setPixmap(pixmap.scaled(self.userImageLabel1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 将图片标签添加到布局
        mainLayout.addWidget(imageLabel)

    def updateSelection(self):
        # print(c_panel)
        if self.selectedColor and self.selectedHairStyle:
            self.input_path["style"] = f"style/{self.selectedHairStyle}"
            if self.selectedColor.name() == "#000000":
                self.input_path["color"] = self.input_path["style"]
            else:
                self.input_path["color"] = f"color/{c_panel[self.selectedColor.name().upper()]}"
            print(f"Selected Path: {self.input_path}")
            # 在这里更新界面或处理路径等


if __name__ == '__main__':
    user_image = []
    c_panel = {
        '#1A171E': "c1",
        '#B0815E': 'c2',
        '#A07544': 'c3',
        '#A5555A': 'c4',
        '#AF643C': 'c5'
    }
    app = QApplication(sys.argv)
    ex = HairStyleApp()
    ex.show()
    sys.exit(app.exec_())
