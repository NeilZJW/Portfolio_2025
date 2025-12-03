#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QStackedWidget, QWidget, QVBoxLayout, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QLabel, QFileDialog,
    QProgressBar
)

from data_loader import data_get
from rule import filter_by_gender, filter_by_age, filter_by_education, filter_by_major
from train_eval import data_preprocess, eval_in_app


# =======================
# 主菜单页面 (Index = 0)
# =======================
class MainMenuPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        title_label = QLabel("Welcome to the HR system, please select the function:")
        layout.addWidget(title_label)

        # 按钮：访问分类器
        self.classifier_btn = QPushButton("Accessing the filter")
        self.classifier_btn.clicked.connect(self.go_classifier)
        layout.addWidget(self.classifier_btn)

        # 按钮：仅测试模型
        self.test_btn = QPushButton("Upload and launch the classifier")
        self.test_btn.clicked.connect(self.go_test)
        layout.addWidget(self.test_btn)

        self.setLayout(layout)

    def go_classifier(self):
        """切换到分类器页面"""
        self.parent().show_classifier()

    def go_test(self):
        """切换到测试页面"""
        self.parent().show_test()


# =======================
# 分类器页面 (Index = 1)
# =======================
class ClassifierPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HR Filter")

        # 初始 data 为 None 或空 DataFrame
        self.data = pd.DataFrame()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 性别筛选
        self.genderComboBox = QComboBox()
        self.genderComboBox.addItems(["Не ограничено", "Муж", "Жен"])
        self.genderComboBox.currentIndexChanged.connect(self.update_table)
        layout.addWidget(self.genderComboBox)

        # 年龄筛选
        self.ageComboBox = QComboBox()
        self.ageComboBox.addItems(["Не ограничено", "18-", "18-25", "26-35", "36-45", "46+"])
        self.ageComboBox.currentIndexChanged.connect(self.update_table)
        layout.addWidget(self.ageComboBox)

        # 学历筛选
        self.eduComboBox = QComboBox()
        self.eduComboBox.addItems(["Не ограничено", "бакалавр", "магистр", "кандидат", "доктор"])
        self.eduComboBox.currentIndexChanged.connect(self.update_table)
        layout.addWidget(self.eduComboBox)

        # 专业筛选
        self.majComboBox = QComboBox()
        self.majComboBox.addItems([
            "Не ограничено",
            "ComputerScience",
            "Engineering",
            "Finance",
            "Journalism",
            "Management",
            "TechnicalWorker"
        ])
        self.majComboBox.currentIndexChanged.connect(self.update_table)
        layout.addWidget(self.majComboBox)

        # 数据表格
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # 返回按钮
        self.back_btn = QPushButton("← Back to menu")
        self.back_btn.clicked.connect(self.go_menu)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

        # 注意：此处不再调用 self.load_data()，留待 MainWindow.show_classifier() 中调用
        self.update_table()

    def load_data(self, path="test_with_predictions.csv"):
        """
        在该方法中，判断文件是否存在：
          - 若存在，读取 csv
          - 若不存在，加载空 DataFrame or 提示
        """
        if os.path.exists(path):
            self.data = data_get(path)
            print(f"[ClassifierPage] Successfully loaded data from {path}")
        else:
            self.data = pd.DataFrame()
            print(f"[ClassifierPage] {path} not found, load empty DataFrame.")

        # 重新刷新表格
        self.update_table()

    def update_table(self):
        """根据用户选择的筛选条件更新表格"""
        if self.data.empty:
            # 如果 data 为空，则清空表格并提示
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.table.setHorizontalHeaderLabels([])
            return

        filtered_data = self.data.copy()

        # 筛选性别
        selected_gender = self.genderComboBox.currentText()
        if selected_gender != "Не ограничено":
            filtered_data = filter_by_gender(filtered_data, selected_gender)

        # 筛选年龄
        selected_age = self.ageComboBox.currentText()
        if selected_age != "Не ограничено":
            if selected_age == "18-":
                filtered_data = filter_by_age(filtered_data, "T")
            elif selected_age == "18-25":
                filtered_data = filter_by_age(filtered_data, "A")
            elif selected_age == "26-35":
                filtered_data = filter_by_age(filtered_data, "B")
            elif selected_age == "36-45":
                filtered_data = filter_by_age(filtered_data, "C")
            elif selected_age == "46+":
                filtered_data = filter_by_age(filtered_data, "D")

        # 筛选学历
        selected_edu = self.eduComboBox.currentText()
        if selected_edu != "Не ограничено":
            filtered_data = filter_by_education(filtered_data, selected_edu)

        # 筛选专业
        selected_maj = self.majComboBox.currentText()
        if selected_maj != "Не ограничено":
            filtered_data = filter_by_major(filtered_data, selected_maj)

        # 更新表格
        self.table.setRowCount(len(filtered_data))
        self.table.setColumnCount(len(filtered_data.columns))
        self.table.setHorizontalHeaderLabels(filtered_data.columns)

        for row_index, (_, row_data) in enumerate(filtered_data.iterrows()):
            for col_index, value in enumerate(row_data):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))

    def go_menu(self):
        """返回主菜单"""
        self.parent().show_menu()


# =======================
# 测试页面 (Index = 2)
# =======================
class TestPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classifier")
        self.test_df = None  # 用户上传的测试集 DataFrame

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label_title = QLabel("Please upload the test set and start performing the classification function.")
        layout.addWidget(self.label_title)

        # 上传测试集
        self.btn_upload_test = QPushButton("upload dataset (CSV)")
        self.btn_upload_test.clicked.connect(self.upload_test_csv)
        layout.addWidget(self.btn_upload_test)

        # 显示已选择文件路径
        self.label_file_path = QLabel("The test set file is not selected")
        layout.addWidget(self.label_file_path)

        # 进度条
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        layout.addWidget(self.progressBar)

        # 开始测试按钮
        self.btn_start_test = QPushButton("Start testing")
        self.btn_start_test.clicked.connect(self.start_testing)
        layout.addWidget(self.btn_start_test)

        # 显示测试状态
        self.label_status = QLabel("Status: Not started")
        layout.addWidget(self.label_status)

        # 返回按钮
        self.back_btn = QPushButton("← Back to menu")
        self.back_btn.clicked.connect(self.go_menu)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

    def upload_test_csv(self):
        """让用户选择测试集"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecting dataset (CSV)", "", "CSV Files (*.csv)")
        if file_path:
            self.test_df = data_get(file_path)
            self.label_file_path.setText(f"The dataset has been selected: {file_path}")
        else:
            self.label_file_path.setText("No dataset was selected")

    def start_testing(self):
        """执行测试：只调用 eval_in_app，不显示准确率"""
        if self.test_df is None:
            self.label_file_path.setText("Please upload the dataset first!")
            return

        # 进度模拟
        self.progressBar.setValue(20)
        self.label_status.setText("Status: Data preprocessing...")

        # 生成 fasttext_test.txt
        test_file = data_preprocess(self.test_df, "Описание", train=False)
        self.progressBar.setValue(60)
        self.label_status.setText("Status: Start prediction...")

        # 直接调用 eval_in_app
        msg = eval_in_app(test_file, self.test_df, model_path="resume_classifier.bin")
        self.progressBar.setValue(100)

        # 显示状态
        self.label_status.setText(f"Status：{msg}")

    def go_menu(self):
        """返回主菜单"""
        self.parent().show_menu()


# =======================
# 主窗口：QStackedWidget
# =======================
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.menu_page = MainMenuPage(self)     # 索引0
        self.classifier_page = ClassifierPage(self)  # 索引1
        self.test_page = TestPage(self)         # 索引2

        self.addWidget(self.menu_page)
        self.addWidget(self.classifier_page)
        self.addWidget(self.test_page)

        self.setCurrentIndex(0)
        self.setWindowTitle("HR System")
        self.resize(1200, 800)

    def show_menu(self):
        self.setCurrentIndex(0)

    def show_classifier(self):
        """
        仅在切换到分类器页面时，才尝试加载 test_with_predictions.csv
        """
        self.classifier_page.load_data("test_with_predictions.csv")
        self.setCurrentIndex(1)

    def show_test(self):
        self.setCurrentIndex(2)


# =======================
# 程序入口
# =======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
