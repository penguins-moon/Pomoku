from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QComboBox
from PySide6.QtGui import QPainter, QColor, QPixmap, QCursor, QPen, QFont, QRadialGradient
from PySide6.QtCore import Qt, Slot, QRect
import qtmodern.styles
import qtmodern.windows
import sys
# import os #vscode
# import gobang_v2 as gobang # 2.0
import gobang # 1.0

class mywindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setWindowTitle("五子棋")
        self.setGeometry(100, 100, 1050, 760)
        self.AI = False
        self.Afirsthand = False
        self.setup_ui()

    def setup_ui(self):
        """set up the user interface"""
        self.Layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # 左侧棋盘
        self.left_layout.addWidget(chessboard)
        self.Layout.addLayout(self.left_layout)

        # 右方控件
        banner = QHBoxLayout()
        title1 = QLabel()
        title1.setPixmap(QPixmap("icons\Blackpiece.png"))
        title2 = QLabel()
        title2.setPixmap(QPixmap("icons\whitepiece.png"))
        title3 = QLabel()
        title3.setPixmap(QPixmap("icons\Blackpiece.png"))
        banner.addWidget(title1)
        banner.addWidget(title2)
        banner.addWidget(title3)
        self.right_layout.addLayout(banner)

        line1 = QHBoxLayout()
        label1 = QLabel("Human")
        label1.setStyleSheet("background-color: brown;font-size: 20px;")
        label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line1.addWidget(label1)
        color1 = QComboBox()
        color1.addItems(['黑棋', '白棋'])
        line1.addWidget(color1)
        self.right_layout.addLayout(line1)

        self.Pass1 = QPushButton("PASS")
        self.Pass1.setMaximumWidth(300)
        self.Pass1.clicked.connect(chessboard.Pass)

        line2 = QHBoxLayout()
        label2 = QLabel("Player B")
        label2.setStyleSheet("background-color: brown;font-size: 20px;")
        label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line2.addWidget(label2)
        self.color2 = QComboBox()
        self.color2.addItems(['白棋', '黑棋'])
        line2.addWidget(self.color2)
        self.right_layout.addLayout(line2)

        color1.currentIndexChanged.connect(lambda: self.color2.setCurrentIndex(color1.currentIndex()))
        self.color2.currentIndexChanged.connect(lambda: color1.setCurrentIndex(self.color2.currentIndex()))
        self.color2.currentTextChanged.connect(self.switch_mode)

        depthlabel = QLabel("设置搜索深度:")
        depthlabel.setMaximumWidth(100)
        self.set_depth = QComboBox()
        self.set_depth.addItems(['4', '5', '6', '7', '8', '9', '10', '12', '16', '20', '1000'])
        self.set_depth.setMaximumWidth(200)
        self.set_depth.setEnabled(False)
        line3 = QHBoxLayout()
        line3.addWidget(depthlabel)
        line3.addWidget(self.set_depth)

        num_label = QLabel("设置节点数目:")
        num_label.setMaximumWidth(100)
        self.set_num = QComboBox()
        self.set_num.addItems(['4', '5', '6', '7', '8','10', '12', '16', '25', '100'])
        self.set_num.setMaximumWidth(200)
        self.set_num.setEnabled(False)
        line4 = QHBoxLayout()
        line4.addWidget(num_label)
        line4.addWidget(self.set_num)

        self.analysis = QLabel()
        self.analysis.setWordWrap(True)
        self.analysis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis.setFont(QFont("方正姚体", 15, QFont.Bold, False))
        self.analysis.setStyleSheet("background-color: grey; font-size: 15px;")
        self.analysis.setMaximumWidth(300)

        self.Pass2 = QPushButton("NEXT STEP")
        self.Pass2.setMaximumWidth(300)
        self.Pass2.setEnabled(False)
        self.Pass2.clicked.connect(chessboard.Next)

        self.label0 = QLabel("第 0 回合")
        self.label0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label0.setMaximumWidth(300)
        self.label0.setFont(QFont("华文行楷", 15, QFont.Bold, False))
        self.label0.setStyleSheet("background-color: grey;")

        self.mode = QComboBox()
        self.mode.addItems(['Human', 'AI-basic', 'AI-alphabeta', 'AI-MCTS'])
        self.mode.currentTextChanged.connect(lambda: label2.setText(self.mode.currentText()))
        self.mode.currentTextChanged.connect(self.switch_mode)

        self.button1 = QPushButton("START")
        self.button1.setMaximumWidth(300)
        self.button1.clicked.connect(chessboard.start_game)
        self.button1.clicked.connect(lambda: self.button1.setEnabled(False))

        rollout_label = QLabel("设置模拟次数:")
        rollout_label.setMaximumWidth(100)
        self.set_nrollout = QComboBox()
        self.set_nrollout.addItems(['100', '200', '400', '800', '1600'])
        self.set_nrollout.setMaximumWidth(200)
        self.set_nrollout.setEnabled(False)
        line5 = QHBoxLayout()
        line5.addWidget(rollout_label)
        line5.addWidget(self.set_nrollout) 

        button2 = QPushButton("refresh")
        button2.setMaximumWidth(300)
        button2.clicked.connect(chessboard.clear)
        button2.clicked.connect(self.reset)
        button2.clicked.connect(lambda: self.button1.setEnabled(True))

        self.diary = QLabel("information box")
        self.diary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diary.setMaximumWidth(300)
        self.diary.setFont(QFont("方正姚体", 15, QFont.Bold,False))
        self.diary.setStyleSheet("background-color: grey;")

        self.regret = QPushButton('悔棋')
        self.regret.clicked.connect(chessboard.regret)
        self.regret.setEnabled(False)
        self.regret.setMaximumWidth(300)

        sizelabel = QLabel("棋盘大小:")
        sizelabel.setMaximumWidth(100)
        self.swithch_size = QComboBox()
        self.swithch_size.addItems(['15*15','7*7','9*9', '13*13','19*19'])
        self.swithch_size.setMaximumWidth(200)
        self.swithch_size.currentTextChanged.connect(
            lambda: chessboard.set_cell_num(int(self.swithch_size.currentText().split('*')[0])))
        line6 = QHBoxLayout()
        line6.addWidget(sizelabel)
        line6.addWidget(self.swithch_size)
        
        self.right_layout.addWidget(self.label0)
        self.right_layout.addWidget(self.mode)
        self.right_layout.addLayout(line3)
        self.right_layout.addLayout(line4)
        self.right_layout.addLayout(line5) 
        self.right_layout.addLayout(line6)
        self.right_layout.addWidget(self.Pass1)
        self.right_layout.addWidget(self.Pass2)
        self.right_layout.addWidget(self.regret)
        self.right_layout.addWidget(self.button1)
        self.right_layout.addWidget(button2)
        self.right_layout.addWidget(self.analysis)
        self.right_layout.addWidget(self.diary)
        self.Layout.addLayout(self.right_layout)

        self.setLayout(self.Layout)

    @Slot()
    def switch_mode(self):
        """choose AI type"""
        if self.mode.currentText() in ['AI-basic', 'AI-alphabeta', 'AI-MCTS']:
            self.AI = True
            self.Pass2.setEnabled(True)
            if self.mode.currentText() == 'AI-alphabeta':
                self.set_depth.setEnabled(True)
                self.set_num.setEnabled(True)
                self.set_nrollout.setEnabled(False)
            elif self.mode.currentText() == 'AI-MCTS':
                self.set_depth.setEnabled(True)
                self.set_num.setEnabled(True)
                self.set_nrollout.setEnabled(True)
            else:
                self.set_depth.setEnabled(False)
                self.set_num.setEnabled(False)
                self.set_nrollout.setEnabled(False)
                
        else:
            self.AI = False
        if self.color2.currentText() == '白棋':
            self.Afirsthand = True
        else:
            self.Afirsthand = False

    @Slot()
    def reset(self):
        """reset the buttons"""
        self.analysis.setText("")
        self.diary.setText('information box')
        self.regret.setEnabled(False)
        self.Pass2.setEnabled(False)
        self.swithch_size.setEnabled(True)
        self.switch_mode()

    @Slot()
    def hault(self):
        """game over"""
        #self.regret.setEnabled(False)
        self.Pass1.setEnabled(False)
        self.Pass2.setEnabled(False)
        self.set_depth.setEnabled(False)
        self.set_num.setEnabled(False)
        self.set_nrollout.setEnabled(False)


class Chessboard(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cell_num = 15
        self.cell_size = 720//self.cell_num
        self.round = 0
        self.Cursor = []
        self.Cursor.append(
            QCursor(QPixmap("icons/Blackpiece.png").scaled(self.cell_size, self.cell_size), 5, 5))
        self.Cursor.append(
            QCursor(QPixmap("icons/whitepiece.png").scaled(self.cell_size, self.cell_size), 5 ,5))
        #self.scene = QGraphicsScene(self)
        self.r_pos=self.cell_num//2
        self.c_pos=self.cell_num//2 #鼠标位置
        self.setMouseTracking(True) #跟踪鼠标位置
        self.setCursor(self.Cursor[0])
        self.resize(800, 800)

    @Slot()
    def start_game(self):
        """initialize the game"""
        window.diary.setText('Game Start!')
        window.switch_mode()
        window.swithch_size.setEnabled(False)
        window.set_depth.setEnabled(False)
        window.set_num.setEnabled(False)
        self.round = 0
        mode=[window.mode.currentText() == "AI-basic",window.mode.currentText() == 'AI-alphabeta',window.mode.currentText() == 'AI-MCTS']
        game.switch_mode(window.AI,mode,(1 if window.Afirsthand else 2))
        game.choose_policy()
        if window.mode.currentText() == 'AI-alphabeta':
            game.master.switch_mode(int(window.set_depth.currentText()),int(window.set_num.currentText()))
        elif window.mode.currentText() =='AI-MCTS':
            game.master.switch_mode(int(window.set_depth.currentText()),int(window.set_num.currentText()),int(window.set_nrollout.currentText()))
        game.start_game()
        self.setCursor(self.Cursor[game.current_player - 1])
        self.update()

    @Slot()
    def Pass(self):
        """pass the current round"""
        if game.ingame:
            window.diary.setText('{}:PASS'.format('黑' if game.current_player == 1 else '白'))
            game.Pass()
            self.round=len(game.past)//2
            window.label0.setText(f'第{self.round}回合')
            self.setCursor(self.Cursor[game.current_player - 1])
            self.update()

    @Slot()
    def Next(self):
        """AI takes the next move"""
        game.Next()
        self.round=len(game.past)//2
        window.label0.setText(f'第{self.round}回合')
        self.setCursor(self.Cursor[game.current_player - 1])
        self.update()

    @Slot()
    def clear(self):
        """reset the game"""
        self.round = 0
        window.label0.setText(f'第{self.round}回合')
        self.setCursor(self.Cursor[0])
        game.clear()
        game.master.clear()
        self.update()

    @Slot()
    def regret(self):
        """regret before it's too late"""
        game.regret()
        self.update()
        if len(game.past) <= 3:
            window.regret.setEnabled(False)

    @Slot()
    def set_cell_num(self, num):
        """reset the board size"""
        self.cell_num = num
        game.set_cell_num(num)
        game.master.set_cell_num(num)
        self.cell_size = 720 // num
        self.adjust_cursor()
        self.update()
    
    def adjust_cursor(self):
        """change cursor size and color"""
        self.Cursor[0]=QCursor(QPixmap("icons/Blackpiece.png").scaled(self.cell_size, self.cell_size), self.cell_size//2, self.cell_size//2)
        self.Cursor[1]=QCursor(QPixmap("icons/whitepiece.png").scaled(self.cell_size, self.cell_size), self.cell_size//2, self.cell_size//2) 
        self.setCursor(self.Cursor[game.current_player - 1])

    def paintEvent(self, event):
        """paint the chessboard"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, self.cell_num * self.cell_size, self.cell_num * self.cell_size, QColor(175, 160, 125))
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(QColor(0, 0, 0))
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0))
        #棋盘绘制
        if self.cell_num==15: #标准棋盘
            painter.drawEllipse((self.cell_num - 1) / 4 * self.cell_size - 4, (self.cell_num - 1) / 4 * self.cell_size - 4,
                                8, 8)
            painter.drawEllipse((self.cell_num - 1) / 4 * self.cell_size - 4,
                                (self.cell_num - (self.cell_num - 1) / 4) * self.cell_size - 4, 8, 8)
            painter.drawEllipse((self.cell_num - (self.cell_num - 1) / 4) * self.cell_size - 4,
                                (self.cell_num - 1) / 4 * self.cell_size - 4, 8, 8)
            painter.drawEllipse((self.cell_num - (self.cell_num - 1) / 4) * self.cell_size - 4,
                                (self.cell_num - (self.cell_num - 1) / 4) * self.cell_size - 4, 8, 8)
            painter.drawEllipse((self.cell_num / 2) * self.cell_size - 4, (self.cell_num / 2) * self.cell_size - 4, 8, 8)
        for i in range(self.cell_num):
            painter.drawLine(self.cell_size * (2 * i + 1) / 2, self.cell_size / 2, self.cell_size * (2 * i + 1) / 2,
                             self.cell_num * self.cell_size - self.cell_size / 2)
            painter.drawLine(self.cell_size / 2, self.cell_size * (2 * i + 1) / 2,
                             self.cell_size * self.cell_num - self.cell_size / 2, self.cell_size * (2 * i + 1) / 2)
        
        painter.setPen(QColor(Qt.GlobalColor.black))
        painter.setFont(QFont("方正姚体", self.cell_size//5, QFont.Weight.Bold, False))  # 设置字体为Arial，大小为12
        for i in range(self.cell_num):
            painter.drawText(self.cell_size * (2*i + 1)/2 - self.cell_size//10, self.cell_size//3, str(i))
            painter.drawText(self.cell_size//8, self.cell_size * (2*i + 1)/2 + self.cell_size//8, str(i))
        #光圈跟踪
        if game.past:
            painter.setPen(QColor(0,0,0,0))
            x,y=game.past[-1].get_move()
            glow_color = QColor(255, 255, 0, 150)  # 设置发光颜色和透明度
            gradient = QRadialGradient(10, 10, 100)
            gradient.setColorAt(0, QColor(0, 0, 0, 0))
            gradient.setColorAt(1, glow_color)
            painter.setBrush(gradient)
            painter.drawEllipse(y * self.cell_size-5, x * self.cell_size-5, self.cell_size+10, self.cell_size+10)
            painter.setPen(QColor(Qt.GlobalColor.black))
        
        #显示棋子
        for r in range(self.cell_num):
            for c in range(self.cell_num):
                if game.board[r][c] == 1:
                    painter.setBrush(QColor(Qt.GlobalColor.black))
                    painter.drawEllipse(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                elif game.board[r][c] == 2:
                    painter.setBrush(QColor(Qt.GlobalColor.white))
                    painter.drawEllipse(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)

        #标记棋子
        if game.past:
            painter.setFont(QFont("Arial", self.cell_size//3, QFont.Weight.Bold, True))  # 设置字体为Arial，大小为12
            for i in range(len(game.past)-1):
                if game.past[i].get_player() == 2: #黑棋完成落子
                    painter.setPen(QColor(Qt.GlobalColor.white))
                else:
                    painter.setPen(QColor(Qt.GlobalColor.black))
                x,y=game.past[i].get_move()
                rect=QRect(y*self.cell_size, x*self.cell_size, self.cell_size, self.cell_size)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(i+1))
            painter.setPen(QColor(Qt.GlobalColor.red))
            x,y=game.past[-1].get_move()
            rect=QRect(y*self.cell_size, x*self.cell_size, self.cell_size, self.cell_size)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(len(game.past)))
        
        #落子位置预览
        c = self.c_pos
        r = self.r_pos
        if game.ingame and gobang.Game.is_valid(r, c, game.board, self.cell_num):
            if game.current_player == 1:
                painter.setBrush(QColor(Qt.GlobalColor.black))
            else:
                painter.setBrush(QColor(Qt.GlobalColor.white))
            painter.setOpacity(0.5)
            painter.drawEllipse(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            painter.setOpacity(1)
        
    
    def mouseMoveEvent(self, event):
        """get mouse coordinate"""
        self.c_pos = event.x() // self.cell_size
        self.r_pos = event.y() // self.cell_size
        self.update()


    def mousePressEvent(self, event):
        """mouse click"""
        c = event.x() // self.cell_size
        r = event.y() // self.cell_size
        end=False
        if game.ingame and gobang.Game.is_valid(r, c, game.board, self.cell_num):
            window.diary.setText('{}子落 ({},{})'.format('黑' if game.current_player == 1 else '白', r, c))
            end=game.put_piece(r,c)
            self.repaint()
            if end:
                window.analysis.setText('Game over\n{}子获胜'.format('黑' if game.current_player == 1 else '白'))
                window.hault()
            else:
                r,c,win_rate,end=game.get_move(r,c)
                if win_rate >= 0: window.analysis.setText('AI 胜率： {:.2f}%'.format(100*win_rate))
                self.repaint()
        if end:
            window.analysis.setText('Game over\n{}子获胜'.format('黑' if game.current_player == 1 else '白'))
            window.hault()
        if game.ingame:
            window.diary.setText('{}子落 ({},{})'.format('黑' if game.current_player == 2 else '白',r,c))  
            self.round=len(game.past)//2
            window.label0.setText(f'第{self.round}回合')
            if len(game.past) >= 2:
                window.regret.setEnabled(True)
            self.setCursor(self.Cursor[game.current_player - 1])
        self.update()   



if __name__ == "__main__":
    # os.chdir(sys.path[0])
    app = QApplication([])
    qtmodern.styles.dark(app)
    game=gobang.Game()
    chessboard = Chessboard()
    window = mywindow()
    prettier = qtmodern.windows.ModernWindow(window)
    prettier.show()
    sys.exit(app.exec())
