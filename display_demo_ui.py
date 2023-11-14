# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'display_demo.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(2129, 849)
        Form.setStyleSheet(u"")
        self.graphics_monkey = QGraphicsView(Form)
        self.graphics_monkey.setObjectName(u"graphics_monkey")
        self.graphics_monkey.setGeometry(QRect(0, 40, 231, 241))
        self.graphics_monkey.setStyleSheet(u"border: none")
        self.graphics_trail = QGraphicsView(Form)
        self.graphics_trail.setObjectName(u"graphics_trail")
        self.graphics_trail.setGeometry(QRect(20, 350, 201, 191))
        self.graphics_trail.setStyleSheet(u"")
        self.button_select_data = QPushButton(Form)
        self.button_select_data.setObjectName(u"button_select_data")
        self.button_select_data.setGeometry(QRect(50, 610, 131, 81))
        font = QFont()
        font.setPointSize(14)
        self.button_select_data.setFont(font)
        self.graphics_trail_inputspikes = QGraphicsView(Form)
        self.graphics_trail_inputspikes.setObjectName(u"graphics_trail_inputspikes")
        self.graphics_trail_inputspikes.setGeometry(QRect(240, 70, 491, 761))
        self.graphics_trail_inputspikes.setStyleSheet(u"")
        self.graphics_trail_ouputspikes = QGraphicsView(Form)
        self.graphics_trail_ouputspikes.setObjectName(u"graphics_trail_ouputspikes")
        self.graphics_trail_ouputspikes.setGeometry(QRect(750, 70, 1331, 431))
        self.graphics_trail_ouputspikes.setStyleSheet(u"")
        self.textEdit = QTextEdit(Form)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(30, 300, 191, 41))
        self.textEdit.setTabletTracking(False)
        self.textEdit.setAutoFillBackground(False)
        self.textEdit.setStyleSheet(u"border: none")
        self.textEdit.setTabChangesFocus(True)
        self.graphics_pre_trail = QGraphicsView(Form)
        self.graphics_pre_trail.setObjectName(u"graphics_pre_trail")
        self.graphics_pre_trail.setGeometry(QRect(760, 580, 211, 191))
        self.graphics_pre_trail.setStyleSheet(u"border: none")
        self.textEdit_2 = QTextEdit(Form)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(750, 520, 251, 51))
        self.textEdit_2.setStyleSheet(u"border: none")
        self.textEdit_3 = QTextEdit(Form)
        self.textEdit_3.setObjectName(u"textEdit_3")
        self.textEdit_3.setEnabled(True)
        self.textEdit_3.setGeometry(QRect(1290, 510, 261, 41))
        self.textEdit_3.setTabletTracking(False)
        self.textEdit_3.setStyleSheet(u"border: none")
        self.listView = QListView(Form)
        self.listView.setObjectName(u"listView")
        self.listView.setGeometry(QRect(-50, 0, 2201, 861))
        self.graphics_power = QGraphicsView(Form)
        self.graphics_power.setObjectName(u"graphics_power")
        self.graphics_power.setGeometry(QRect(1050, 560, 841, 241))
        self.power = QLabel(Form)
        self.power.setObjectName(u"power")
        self.power.setGeometry(QRect(1900, 590, 211, 161))
        self.power.setFont(font)
        self.textEdit_4 = QTextEdit(Form)
        self.textEdit_4.setObjectName(u"textEdit_4")
        self.textEdit_4.setGeometry(QRect(360, 10, 251, 51))
        self.textEdit_4.setStyleSheet(u"border: none")
        self.textEdit_5 = QTextEdit(Form)
        self.textEdit_5.setObjectName(u"textEdit_5")
        self.textEdit_5.setGeometry(QRect(1330, 10, 251, 51))
        self.textEdit_5.setStyleSheet(u"border: none")
        self.listView.raise_()
        self.graphics_monkey.raise_()
        self.graphics_trail.raise_()
        self.button_select_data.raise_()
        self.graphics_trail_inputspikes.raise_()
        self.graphics_trail_ouputspikes.raise_()
        self.textEdit.raise_()
        self.graphics_pre_trail.raise_()
        self.textEdit_2.raise_()
        self.textEdit_3.raise_()
        self.graphics_power.raise_()
        self.power.raise_()
        self.textEdit_4.raise_()
        self.textEdit_5.raise_()

        self.retranslateUi(Form)

        self.button_select_data.setDefault(True)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Demo", None))
        self.button_select_data.setText(QCoreApplication.translate("Form", u"select_data", None))
        self.textEdit.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt;\">Original track</span></p></body></html>", None))
        self.textEdit_2.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt;\">Predicted track</span></p></body></html>", None))
        self.textEdit_3.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt;\">Power consumption</span></p></body></html>", None))
        self.power.setText("")
        self.textEdit_4.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt;\">Input Spikes</span></p></body></html>", None))
        self.textEdit_5.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt;\">Ouput Spikes</span></p></body></html>", None))
    # retranslateUi

