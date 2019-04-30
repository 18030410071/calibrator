import cv2
import os
import shelve
import numpy
import math

from .transform import PointP
from . import geometry as ge
from . import file as utl


def get_random_number(low, high):
    """
    get a random in [low, high]
    :param low: low boundary
    :param high: high boundary
    :return: random number
    """
    return numpy.random.randint(int(low), int(high))


class Corner(object):

    def __init__(self):
        self.p1 = PointP(0, 0)
        self.p2 = PointP(100, 0)
        self.p3 = PointP(0, 100)
        self.p4 = PointP(100, 100)

    def randomize(self, width, height):
        self.p1.update(get_random_number(0, width * 0.25), get_random_number(0, height * 0.25))
        self.p2.update(get_random_number(width * 0.75, width), get_random_number(0, height * 0.25))
        self.p3.update(get_random_number(0, width * 0.25), get_random_number(height * 0.75, height))
        self.p4.update(get_random_number(width * 0.75, width), get_random_number(height * 0.75, height))

    def __str__(self):
        return '%s, %s, %s, %s' % (self.p1, self.p2, self.p3, self.p4)

    def numpy(self):
        return numpy.array([self.p1.tuple(), self.p2.tuple(), self.p3.tuple(), self.p4.tuple()], numpy.float32)


class DragCorner:

    def __init__(self, img, corners, function, win_name, size=(900, 500), pos=(0, 0), line_thick=1):
        self.img = img
        self.corners = corners
        self.function = function
        self.win_name = win_name
        self.size = size
        self.pos = pos
        self.res = []
        self._point = None
        self.line_thick = line_thick

    def _get_nearest_corner(self):
        """
        get nearest corner
        :return: nearest corner
        """
        nearest = self.corners.p1
        distance = self._point.distance_to_point(nearest)

        dist = self._point.distance_to_point(self.corners.p2)
        if dist < distance:
            distance = dist
            nearest = self.corners.p2

        dist = self._point.distance_to_point(self.corners.p3)
        if dist < distance:
            distance = dist
            nearest = self.corners.p3

        dist = self._point.distance_to_point(self.corners.p4)
        if dist < distance:
            nearest = self.corners.p4

        return nearest

    def _update(self, event, x, y, flag, para):
        """
        mouse event
        """
        image = self.img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._point = ge.Point(x, y)
            self._point = self._get_nearest_corner()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._point is not None:
                self._point.update(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._point = None

        cv2.line(image, self.corners.p1.tuple(), self.corners.p2.tuple(), (255, 255, 255), self.line_thick)
        cv2.line(image, self.corners.p1.tuple(), self.corners.p3.tuple(), (255, 255, 255), self.line_thick)
        cv2.line(image, self.corners.p2.tuple(), self.corners.p4.tuple(), (255, 255, 255), self.line_thick)
        cv2.line(image, self.corners.p3.tuple(), self.corners.p4.tuple(), (255, 255, 255), self.line_thick)

        cv2.circle(image, self.corners.p1.tuple(), 5, [255, 255, 255], 1)
        cv2.circle(image, self.corners.p2.tuple(), 10, [255, 255, 255], 2)
        cv2.circle(image, self.corners.p3.tuple(), 15, [255, 255, 255], 3)
        cv2.circle(image, self.corners.p4.tuple(), 20, [255, 255, 255], 4)

        cv2.imshow(self.win_name, image)

    def run(self, delay):
        """
        start tuning
        :param delay:
            if True, waitKey(0)
        """
        if self.size == 0:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.win_name, self.pos[0], self.pos[1])
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.win_name)
            cv2.resizeWindow(self.win_name, self.size[0], self.size[1])

        cv2.setMouseCallback(self.win_name, self._update, [])
        cv2.moveWindow(self.win_name, self.pos[0], self.pos[1])
        cv2.imshow(self.win_name, self.img)

        if delay:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def trasform_remap(image, map_x, map_y):
    """
    transform input image into warped one according to the a nonlinear coordinate map f(x)
    :return: output warped image
    """
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


if __name__ == '__main__':
    from model.court import Court
    # load image
    _path = "data/tennis"
    _img_path = os.path.join(_path, "test.jpg")
    if not os.path.exists(_img_path):
        raise IOError('cannot open video： %s' % _img_path)
    _image = cv2.imread(_img_path)

    # load param
    _dat_path = os.path.join(_path, "param.dat")
    try:
        _camera_corner = utl.load_dat(_dat_path, "corner")
    except (IOError, KeyError):
        print("Can't find db, create one.")
        _camera_corner = Corner()

    # process corner
    tuner = DragCorner(_image, _camera_corner, [], 'tuning corner', pos=(200, 100))
    # cv2.moveWindow('tuning corner', 0, 0)
    tuner.run(delay=True)

    utl.save_dat(_dat_path, "corner", _camera_corner)

    size = _image.shape
    _figure_width = 200
    _figure_height = 400
    _court = Court(_figure_width, _figure_height)

    _figure_corner = numpy.array([(0, 0), (_figure_width, 0), (0, _figure_height), (_figure_width, _figure_height)], numpy.float32)
    _camera2figure_matrix = cv2.getPerspectiveTransform(_camera_corner.numpy(), _figure_corner)

    _court.set_view2figure_matrix(_camera2figure_matrix)

    utl.save_dat(_dat_path, "court", _court)

    _figure_path = os.path.join(_path, "court.jpeg")
    _figure = cv2.imread(_figure_path)
    court = cv2.resize(_figure, (_figure_width, _figure_height))

    _player_pos = ge.Point(866, 230)
    _player_pos2 = ge.Point(1037, 583)

    _player_figure_pos = _court.view2figure(_player_pos)
    _player_figure_pos2 = _court.view2figure(_player_pos2)
    # _player_figure_pos = _player_pos.perspective(_camera2figure_matrix)

    print(_player_figure_pos)
    cv2.circle(court, _player_figure_pos.int().tuple(), 3, (0, 0, 0), 10)
    cv2.circle(court, _player_figure_pos2.int().tuple(), 3, (0, 0, 0), 10)
    cv2.imshow('court', court)
    cv2.waitKey()
    cv2.destroyAllWindows()
