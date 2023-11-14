import sys
import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path 
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from synnet_rebuild import synnet_mix_rebuild
from networks import *
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.transform import quantize_methods as q
from tqdm.auto import tqdm
from rockpool.devices.xylo.syns61201 import xa2_devkit_utils as hdu


model = synnet_mix_rebuild
model.load('/home/ruixing/workspace/bc_interface/models/LIFTorch_lack_of_Linear_91channel_modelmix_spike_624_0.9020618556701031.pth')

new_model = new_synnet_mix_rebuild 

def add_linear_layer(new_model, original_model):
    for i, m in enumerate(new_model.seq):
        if isinstance(m, LinearTorch):
            if i == 0:
                pass
            else: new_model.seq[i].weight.data = original_model.seq[i-1].weight.data
        if isinstance(m, LIFTorch):
            new_model.seq[i].tau_syn.data = original_model.seq[i-1].tau_syn.data
    return new_model
complete_model = add_linear_layer(new_model, model)


# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks
connected_hdks, support_modules, chip_versions = find_xylo_hdks()
found_xylo = len(connected_hdks) > 0
if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
spec = x.mapper(complete_model.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))



# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = 0.001)
    print(modSamna)

print(f'Clock freq. set to {hdu.set_xylo_core_clock_freq(modSamna._device, 6.25)} MHz')

dt = 0

triggle = False
select_data_triggle = False
data_array = np.zeros((8, 1)).astype(np.int16)
colors = ['red', 'cyan', 'blue', 'green', 'orange', 'brown', 'pink', 'purple']
x = 0
io_power_list = []
core_power_list = []



class FigureCanvasDemo3(FigureCanvas):
    def __init__(self, function_type=None, data=None, width=10, height=10,control=None):
        fig = Figure(figsize=(width, height), tight_layout=True)
        FigureCanvas.__init__(self, fig)
        if function_type == 'inputspikes':   
            num_samples,num_channels = data.shape
            for channel in range(num_channels):
                ax = fig.add_subplot(num_channels, 1, channel + 1)
                ax.plot(data[:,channel])
                ax.axis('off')
            print('faw')
        elif function_type == 'outputspikes':
            global data_array
            global colors
            global x
            num_channels,_ = data.shape
            data_array = np.concatenate((data_array,data), axis=1)
            for channel in range(num_channels):
                ax = fig.add_subplot(num_channels, 1, channel + 1)
                # ax.plot(data_array[channel],marker='o',markersize=1,linestyle='None',color=colors[channel])
                ax.plot(data_array[channel],color=colors[channel])
                ax.axis('off')
            for ax in fig.get_axes():
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 1)
            x += 1
            out = np.sum(data_array,axis=1) 
            pred = out.argmax(0)
            print(pred)
            out_label_path = f'/home/ruixing/workspace/bc_interface/label/label{pred}.png'
            scene_trail = QGraphicsScene()
            pixmap_trail = QPixmap(out_label_path)
            item_trail = QGraphicsPixmapItem(pixmap_trail)
            scene_trail.addItem(item_trail)
            control.graphics_pre_trail.setScene(scene_trail)
        elif function_type == 'power':
            num_channels,_ = data.shape
            for channel in range(num_channels):
                ax = fig.add_subplot(num_channels, 1, channel + 1)
                ax.plot(data[channel],marker='o',markersize=4,linestyle='None',color=colors[channel])
                if channel == 0:
                    ax.set_ylim(70, 95)
                elif channel == 1:
                    ax.set_ylim(500, 535)
                ax.set_xlim(0, 100)

        self.draw()
        


        
        



class MyMainWindow(QMainWindow):
    def __init__(self):
        global triggle
        super().__init__()

        loadUi("/home/ruixing/workspace/bc_interface/display_demo.ui", self)
        scene_monkey = QGraphicsScene()
        self.graphicsScene1 = QGraphicsScene()
        pixmap_monkey = QPixmap("/home/ruixing/workspace/bc_interface/smart_monkey.png")
        item_monkey = QGraphicsPixmapItem(pixmap_monkey)
        scene_monkey.addItem(item_monkey)
        self.graphics_monkey.setScene(scene_monkey)
        self.button_select_data.clicked.connect(self.select_data)
        self.button_begin = QPushButton("", self)
        self.button_begin.clicked.connect(self.forword_data)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.button_begin.click)
        self.timer.start(5)                      

    def select_data(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        global data
        global select_data_triggle
        data_path, _ = QFileDialog.getOpenFileName(self, "选择npy文件", "", "文件 (*.npy);;所有文件 (*)", options=options)
        label_path = f'/home/ruixing/workspace/bc_interface/label/label{Path(data_path).parts[-2]}.png'
        scene_trail = QGraphicsScene()
        pixmap_trail = QPixmap(label_path)
        item_trail = QGraphicsPixmapItem(pixmap_trail)
        scene_trail.addItem(item_trail)
        self.graphics_trail.setScene(scene_trail)
    

        data = (np.load(data_path))
        self.plot1 = FigureCanvasDemo3(function_type='inputspikes', data = data, width=self.graphics_monkey.width() / 49,
                                       height=self.graphics_monkey.height() / 20)
        self.graphicsScene1 = QGraphicsScene()
        self.graphicsScene1.addWidget(self.plot1)
        self.graphics_trail_inputspikes.setScene(self.graphicsScene1)
        select_data_triggle = True
    
    def forword_data(self):
        global dt
        global select_data_triggle
        global io_power_list
        global core_power_list
        
        
        if x == 500:
            select_data_triggle = False
        if select_data_triggle == True:

            out_model, _, rec = modSamna(data.astype(np.int16)[dt].reshape(-1,91), record=False, record_power=True)
            self.plot2 = FigureCanvasDemo3(function_type='outputspikes', data = out_model.T, width=self.graphics_monkey.width() / 20,
                                        height=self.graphics_monkey.height() / 60,control=self)
            self.graphicsScene2 = QGraphicsScene()
            self.graphicsScene2.addWidget(self.plot2)
            self.graphics_trail_ouputspikes.setScene(self.graphicsScene2)
            if rec['snn_core_power'].size != 0 and rec['io_power'].size != 0:
                self.power.setText(f"Io_Power: {round(rec['io_power'][0][0] * 1000000,2)} μW\nCore_Power: {round(rec['snn_core_power'][0][0] * 1000000,2)} μW")
                io_power_list.append(round(rec['io_power'][0][0] * 1000000,2))
                core_power_list.append(round(rec['snn_core_power'][0][0] * 1000000,2))
            
                power=np.vstack((np.array(io_power_list),np.array(core_power_list)))
                
                self.plot3 = FigureCanvasDemo3(function_type='power', data = power, width=self.graphics_monkey.width() / 30,
                                            height=self.graphics_monkey.height() / 100,control=self)
                self.graphicsScene3 = QGraphicsScene()
                self.graphicsScene3.addWidget(self.plot3)
                self.graphics_power.setScene(self.graphicsScene3)
            
            
            dt += 1
    

            
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
