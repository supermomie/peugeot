import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
#from generate_xml import write_xml

# global constants
img = None
tl_list = []
br_list = []
object_list = []

# constants
image_folder = '../../data/image/GH040022/1589131931.0567613.jpg'
savedir = 'annotations'
obj = 'fidget_spinner'


def line_select_callback(clk, rls):
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)


def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        print(br_list)
        print(tl_list)
        #write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        tl_list = []
        br_list = []
        object_list = []
        img = None
        plt.close()


def toggle_selector(event):
    toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    #for n, image_file in enumerate(os.scandir(image_folder)):
    img = image_folder
    fig, ax = plt.subplots(1)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry = (250, 120, 1280, 1024)
    img = cv2.imread(img)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    toggle_selector.RS = RectangleSelector(
        ax, line_select_callback,
        drawtype='box', useblit=True,
        button=[1], minspanx=1, minspany=1,
        spancoords='pixels', interactive=True
    )
    bbox = plt.connect('key_press_event', toggle_selector)
    key = plt.connect('key_press_event', onkeypress)
    plt.show()

"""
from matplotlib.widgets import MultiCursor
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
t = np.arange(0.0, 2.0, 0.01)
ax1.plot(t, np.sin(2*np.pi*t))
ax2.plot(t, np.sin(4*np.pi*t))

multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1,
                    horizOn=True, vertOn=True)
plt.show()
"""

