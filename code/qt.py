import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.Qt import QLineEdit
from HeatmapGenerator import gen_heatmap
import cv2


heatmap = 'test/heatmap.png'
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Media Project'
        self.left = 100
        self.top = 100
        self.width = 960
        self.height = 800
        self.res = list(range(14))
        self.initUI()
    
    def initUI(self):    
        self.pixmap = QPixmap("White.png")
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.setFont(QFont("Roman times",12))  
        openFile.triggered.connect(self.msg)

        menubar = self.menuBar()
        menubar.setFont(QFont("Roman times",12))  
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)     

        lb1 = QLabel("Origin Picture", self)
        lb1.move(200, 50)
        lb1.setFont(QFont("Roman times",14))  
        lb1.resize(150, 30)

        lb2 = QLabel("Heatmap", self)
        lb2.move(200, 390)
        lb2.setFont(QFont("Roman times",14))  
        lb2.resize(150, 30)

        self.l1 = QLabel(self)
        self.l1.resize(224, 224)
        self.l1.move(150, 90)
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        lbl1 = QLabel(CLASS_NAMES[0], self)
        lbl1.move(550, 50)
        lbl1.setFont(QFont("Roman times",12))  
        lbl1.resize(150, 30)
        self.t1 = QLineEdit(self)
        self.t1.move(700, 50)
        self.t1.setFont(QFont("Roman times",12))  
        self.t1.resize(120, 30)

        lbl2 = QLabel(CLASS_NAMES[1], self)
        lbl2.move(550, 100)
        lbl2.setFont(QFont("Roman times",12))  
        lbl2.resize(150, 30)
        self.t2 = QLineEdit(self)
        self.t2.move(700, 100)
        self.t2.setFont(QFont("Roman times",12))  
        self.t2.resize(120, 30)

        lbl3 = QLabel(CLASS_NAMES[2], self)
        lbl3.move(550, 150)
        lbl3.setFont(QFont("Roman times",12))
        lbl3.resize(150, 30)  
        self.t3 = QLineEdit(self)
        self.t3.move(700, 150)
        self.t3.setFont(QFont("Roman times",12))  
        self.t3.resize(120, 30)

        lbl4 = QLabel(CLASS_NAMES[3], self)
        lbl4.move(550, 200)
        lbl4.setFont(QFont("Roman times",12))  
        lbl4.resize(150, 30)
        self.t4 = QLineEdit(self)
        self.t4.move(700, 200)
        self.t4.setFont(QFont("Roman times",12))  
        self.t4.resize(120, 30)

        lbl5 = QLabel(CLASS_NAMES[4], self)
        lbl5.move(550, 250)
        lbl5.setFont(QFont("Roman times",12))  
        lbl5.resize(150, 30)
        self.t5 = QLineEdit(self)
        self.t5.move(700, 250)
        self.t5.setFont(QFont("Roman times",12))  
        self.t5.resize(120, 30)

        lbl6 = QLabel(CLASS_NAMES[5], self)
        lbl6.move(550, 300)
        lbl6.setFont(QFont("Roman times",12))  
        lbl6.resize(150, 30)
        self.t6 = QLineEdit(self)
        self.t6.move(700, 300)
        self.t6.setFont(QFont("Roman times",12))  
        self.t6.resize(120, 30)

        lbl7 = QLabel(CLASS_NAMES[6], self)
        lbl7.move(550, 350)
        lbl7.setFont(QFont("Roman times",12))  
        lbl7.resize(150, 30)
        self.t7 = QLineEdit(self)
        self.t7.move(700, 350)
        self.t7.setFont(QFont("Roman times",12))  
        self.t7.resize(120, 30)

        lbl8 = QLabel(CLASS_NAMES[7], self)
        lbl8.move(550, 400)
        lbl8.setFont(QFont("Roman times",12))  
        lbl8.resize(150, 30)
        self.t8 = QLineEdit(self)
        self.t8.move(700, 400)
        self.t8.setFont(QFont("Roman times",12))  
        self.t8.resize(120, 30)

        lbl9 = QLabel(CLASS_NAMES[8], self)
        lbl9.move(550, 450)
        lbl9.setFont(QFont("Roman times",12))  
        lbl9.resize(150, 30)
        self.t9 = QLineEdit(self)
        self.t9.move(700, 450)
        self.t9.setFont(QFont("Roman times",12))  
        self.t9.resize(120, 30)

        lbl10 = QLabel(CLASS_NAMES[9], self)
        lbl10.move(550, 500)
        lbl10.setFont(QFont("Roman times",12))  
        lbl10.resize(150, 30)
        self.t10 = QLineEdit(self)
        self.t10.move(700, 500)
        self.t10.setFont(QFont("Roman times",12))  
        self.t10.resize(120, 30)

        lbl11 = QLabel(CLASS_NAMES[10], self)
        lbl11.move(550, 550)
        lbl11.setFont(QFont("Roman times",12))  
        lbl11.resize(150, 30)
        self.t11 = QLineEdit(self)
        self.t11.move(700, 550)
        self.t11.setFont(QFont("Roman times",12))  
        self.t11.resize(120, 30)

        lbl12 = QLabel(CLASS_NAMES[11], self)
        lbl12.move(550, 600)
        lbl12.setFont(QFont("Roman times",12))  
        lbl12.resize(150, 30)
        self.t12 = QLineEdit(self)
        self.t12.move(700, 600)
        self.t12.setFont(QFont("Roman times",12))  
        self.t12.resize(120, 30)

        lbl13 = QLabel(CLASS_NAMES[12], self)
        lbl13.move(550, 650)
        lbl13.setFont(QFont("Roman times",12))  
        lbl13.resize(150, 30)
        self.t13 = QLineEdit(self)
        self.t13.move(700, 650)
        self.t13.setFont(QFont("Roman times",12))  
        self.t13.resize(120, 30)

        lbl14 = QLabel(CLASS_NAMES[13], self)
        lbl14.move(550, 700)
        lbl14.setFont(QFont("Roman times",12))  
        lbl14.resize(150, 30)
        self.t14 = QLineEdit(self)
        self.t14.move(700, 700)
        self.t14.setFont(QFont("Roman times",12))  
        self.t14.resize(120, 30)

        self.l2 = QLabel(self)
        self.l2.resize(224, 224)
        self.l2.move(150, 430)
       
        self.clear()

        button1 = QPushButton('Diagnosis', self)
        button1.move(120, 340) 
        button1.setFont(QFont("Roman times",12))  
        button1.clicked.connect(self.diag)
        
        button2 = QPushButton('clear', self)
        button2.move(280, 340) 
        button2.setFont(QFont("Roman times",12))  
        button2.clicked.connect(self.clear)

        self.show()
        
    def diag(self):   
        try:
            res = gen_heatmap(self.file1, heatmap)
            pixmap2 = QPixmap(heatmap)
            self.l2.setPixmap(pixmap2) 
        
            self.t1.setText(str(res[0]))  
            self.t2.setText(str(res[1]))  
            self.t3.setText(str(res[2]))  
            self.t4.setText(str(res[3]))  
            self.t5.setText(str(res[4]))  
            self.t6.setText(str(res[5]))  
            self.t7.setText(str(res[6]))  
            self.t8.setText(str(res[7]))  
            self.t9.setText(str(res[8]))  
            self.t10.setText(str(res[9]))  
            self.t11.setText(str(res[10]))  
            self.t12.setText(str(res[11]))  
            self.t13.setText(str(res[12]))  
            self.t14.setText(str(res[13]))
        except:
             print("Please choose a picture as input")

    def msg(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    "./",
                                    "All Files (*);;Picture Files (*.jpg)")  
        self.file1 = fileName1
        img = cv2.imread(fileName1)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(fileName1, img)
        pixmap1 = QPixmap(fileName1)
        self.l1.setPixmap(pixmap1)

    def clear(self):
        self.t1.setText("")  
        self.t2.setText("")  
        self.t3.setText("")  
        self.t4.setText("")  
        self.t5.setText("")  
        self.t6.setText("")  
        self.t7.setText("")  
        self.t8.setText("")  
        self.t9.setText("")  
        self.t10.setText("")  
        self.t11.setText("")  
        self.t12.setText("")  
        self.t13.setText("")  
        self.t14.setText("")  
        self.l1.setPixmap(self.pixmap)
        self.l2.setPixmap(self.pixmap)
        self.file1 = ""


if __name__=="__main__":  

    app=QApplication(sys.argv)  
    myshow=MyWindow()
    sys.exit(app.exec_())  
