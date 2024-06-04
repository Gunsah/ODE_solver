import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QLabel, QLineEdit, QPushButton, QGridLayout, QDesktopWidget
from PyQt5.QtGui import QIcon
from random import randint
import numpy as np
from numpy import exp as e
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def add_arrow(line, position=None, direction='right', size=15, color=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        try:
            position = xdata[randint(0, len(xdata))]
        except IndexError:
            position = 0
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


class ODESolver(QWidget):
    def __init__(self):
        super().__init__()
        self.warning = warning_window()
        self.init_ui()


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def init_ui(self):
        self.setWindowIcon(QIcon('МФТИ.svg'))
        self.setWindowTitle('ОДУ - solver')
        self.center()

        self.label_eq1 = QLabel('dx/dt = ')
        self.edit_eq1 = QLineEdit(self)
        self.edit_eq1.setText('x - 4 * y')

        self.label_eq2 = QLabel('dy/dt =')
        self.edit_eq2 = QLineEdit(self)
        self.edit_eq2.setText('2 * x - y')

        self.label_t_span = QLabel('временной промежуток:')
        self.edit_t_span = QLineEdit(self)
        self.edit_t_span.setText('0, 30')
        
        self.label_SC = QLabel('Начальные условия задачи Коши (t = t_start):')
        self.edit_SC = QLineEdit(self)
        self.edit_SC.setText('1, 1')

        self.label_t_steps = QLabel('Количество узлов сетки:')
        self.edit_t_steps = QLineEdit(self)
        self.edit_t_steps.setText('1000')

        self.btn_plot = QPushButton('Построить график численного решения')
        self.btn_plot.clicked.connect(lambda: self.plot_solution(method="precise"))

        self.phase_plot = QPushButton('Построить фазовые траектории')
        self.phase_plot.clicked.connect(lambda: self.plot_solution(method='group'))
        
        layout = QGridLayout()
        layout.setSpacing(10) 
        layout.addWidget(self.label_eq1, 1, 0)
        layout.addWidget(self.edit_eq1, 1, 1, 1, 3)
        layout.addWidget(self.label_eq2, 2, 0)
        layout.addWidget(self.edit_eq2, 2, 1, 2, 3)
        layout.addWidget(self.label_SC)
        layout.addWidget(self.edit_SC)
        layout.addWidget(self.label_t_span)
        layout.addWidget(self.edit_t_span)
        layout.addWidget(self.label_t_steps)
        layout.addWidget(self.edit_t_steps)
        layout.addWidget(self.btn_plot)
        layout.addWidget(self.phase_plot)
        self.setLayout(layout)


    def trans(self):
        
        eq1 = self.edit_eq1.text()
        i = 0
        while i < len(eq1):
            if eq1[i] == 'x':
                eq1 = eq1[:i] + 'y[0]' + eq1[i + 1:]
                i += 2
            elif eq1[i] == 'y':
                eq1 = eq1[:i] + 'y[1]' + eq1[i + 1:] 
                i += 2
            i += 1
            
        eq2 = self.edit_eq2.text()
        i = 0
        while i < len(eq2):
            if eq2[i] == 'x':
                eq2 = eq2[:i] + 'y[0]' + eq2[i + 1:]
                i += 2
            elif eq2[i] == 'y':
                eq2 = eq2[:i] + 'y[1]' + eq2[i + 1:]
                i += 2
            i += 1
        return eq1, eq2


    def plot_solution(self, method):
        plt.close()
        eq1, eq2 = ODESolver.trans(self)
        t_span = tuple(map(float, self.edit_t_span.text().split(',')))
        t_steps = int(self.edit_t_steps.text())
        sc = list(map(float, self.edit_SC.text().split(',')))


        def ode_system(t, y):
            return [eval(eq1), eval(eq2)]


        def precise_graph():
            solution = solve_ivp(ode_system, t_span, sc, t_eval=np.linspace(t_span[0], t_span[1], t_steps))
            plt.plot(solution.y[0], solution.y[1], label='y (x)')
            plt.title('Численное решение')
            plt.xlabel('x(t)')
            plt.ylabel('y(t)')
            plt.legend()
            plt.grid()
            plt.show()
        
        
        def group_graph(key):
            def f(y):
                return np.array([eval(eq1), eval(eq2)])
            
            root = fsolve(f, [1, 1]) if key else np.array([0, 0])
            for x in range(-3, 4):
                for y in range(-3, 4):
                    sc_curr = root + np.array([x, y])
                    solution = solve_ivp(ode_system, t_span, sc_curr, t_eval=np.linspace(t_span[0], t_span[1], t_steps))
                    line = plt.plot(solution.y[0], solution.y[1], alpha = 0.2)[0]
                    add_arrow(line)
                
            plt.title('Фазовые траектории')
            plt.xlabel('x(t)')
            plt.ylabel('y(x(t))')
            plt.grid()
            plt.show()
        
        if method == 'precise':
            precise_graph()
        elif method == 'group':
            if 't' in eq1 or 't' in eq2:
                if self.warning.answer() == True:
                    group_graph(0)
                else:
                    self.warning.close()
            else:
                group_graph(1)


class warning_window(QWidget):
    def __init__(self):
        super().__init__()
        self.msgBox = QMessageBox()
        self.msgBox.setText("Вы хотите построить фазовые траектории, но в системе присутствует время t в правой части в явном виде!")
        self.msgBox.setWindowTitle("Предупреждение!")
        self.msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    
    
    def answer(self):
        returnValue = self.msgBox.exec()
        if returnValue == QMessageBox.Ok:
            return 1
        else:
            return 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ODESolver().show()
    sys.exit(app.exec_())
    
