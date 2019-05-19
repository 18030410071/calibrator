import cv2
import datetime
import os
PATH = "./data/"
image_writer = None


def init():
    global image_writer
    path = PATH
    image_writer = ImagesWrite(path)


class ImagesWrite:
    def __init__(self, path, index=0):
        dir_str = datetime.datetime.now().strftime('%Y-%m-%d %Hh %Mm %Ss')

        self.root_path = os.path.join(path, dir_str)
        os.mkdir(self.root_path)
        self.index = index

    def new_hit(self):
        dir_str = datetime.datetime.now().strftime('%Y-%m-%d %Hh %Mm %Ss')
        self.path = os.path.join(self.root_path, dir_str)
        os.mkdir(self.path)
        self.index = 0

    def write(self, frame):
        cv2.imwrite(self.path + '/%.4d.jpg' % self.index, frame)
        word = '____Save %.4d.jpg in %s____' % (self.index, self.path)
        print(word)
        self.index += 1


if __name__ == '__main__':
    init()
    img = cv2.imread("./demo")
    image_writer.write(img)
