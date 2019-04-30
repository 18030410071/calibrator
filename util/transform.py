import cv2
import numpy

from . import geometry as ge


def get_remap_from_matrix(trans_matrix, width, height):
    matrix = numpy.linalg.inv(trans_matrix)
    map_x = numpy.array([height, width], dtype=float)
    map_y = numpy.array([height, width], dtype=float)
    for x in range(0, height):
        for y in range(0, width):
            map_x[x, y] = (matrix[0, 0] * y + matrix[0, 1] * x + matrix[0, 2]) / \
                          (matrix[2, 0] * y + matrix[2, 1] * x + matrix[2, 2])
            map_y[x, y] = (matrix[1, 0] * y + matrix[1, 1] * x + matrix[1, 2]) / \
                          (matrix[2, 0] * y + matrix[2, 1] * x + matrix[2, 2])
    return map_x, map_y



def warp_image_t2c(image):
    """
    Warp a image from table coordination system to image.
    :param image: the image in table coordinate system.
    :return: image in camera coordinate system.
    """
    return cv2.warpPerspective(image, _table_to_view, (glb.camera_width, glb.camera_height))


def un_distort_image(image):
    """
    :param image: The image you want to undistorted.
    :return: The undistorted image
    """
    image = clb.un_distort_image(image)
    # warp_image_c2v(image)
    # image = clb.un_distort_image_old(image)
    return image


class PointP(ge.Point):
    def __init__(self, x, y):
        if isinstance(x, int):
            self.x = x
        else:
            self.x = int(round(x))

        if isinstance(y, int):
            self.y = y
        else:
            self.y = int(round(y))

    def __str__(self):
        return 'PointP: (%d, %d)' % (self.x, self.y)

    def __add__(self, other):
        return PointP(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return PointP(self.x - other.x, self.y - other.y)

    def table_to_view(self):
        return self.perspective(_table_to_view).int()

    def view_to_table(self):
        return self.perspective(_view_to_table).int()

    def table_to_canvas(self):
        return self.perspective(_table_to_canvas).int()

    def canvas_to_table(self):
        return self.perspective(_canvas_to_table).int()

    def distort(self):
        point = clb.distort_point(self).int()
        return PointP(point.x, point.y)

    def un_distort(self):
        point = clb.un_distort_point(self)
        return PointP(point.x, point.y)

    def perspective(self, mat):
        pt = ge.Point.perspective(self, mat)
        return PointP(pt.x, pt.y)


class SegmentP(PointP, ge.Segment):
    def __init__(self, p1, p2):
        self.A = p2.y - p1.y
        self.B = p1.x - p2.x
        self.C = p2.x * p1.y - p1.x * p2.y
        self.p1 = p1.int()
        self.p2 = p2.int()

    def __str__(self):
        return 'SegmentP is (%d, %d), (%d, %d)' % (self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    __repr__ = __str__

    def table_to_view(self):
        p1 = self.p1.table_to_view()
        p2 = self.p2.table_to_view()
        return SegmentP(p1, p2)

    def view_to_table(self):
        p1 = self.p1.view_to_table()
        p2 = self.p2.view_to_table()
        return SegmentP(p1, p2)

    def table_to_canvas(self):
        p1 = self.p1.table_to_canvas()
        p2 = self.p2.table_to_canvas()
        return SegmentP(p1, p2)

    def distort(self):
        p1 = clb.distort_point(self.p1).int()
        p2 = clb.distort_point(self.p2).int()
        return SegmentP(p1, p2)

    def un_distort(self):
        p1 = clb.un_distort_point(self.p1).int()
        p2 = clb.un_distort_point(self.p2).int()
        return SegmentP(p1, p2)


class VectorP(ge.Vector):
    def __init__(self, x, y):
        ge.Vector.__init__(self, x, y)

    def table_to_view(self):
        return self.perspective(_table_to_view)

    def view_to_table(self):
        return self.perspective(_view_to_table)

    def table_to_canvas(self):
        return self.perspective(_table_to_canvas)


class Vector2P(ge.Vector2, VectorP):
    def __init__(self, start, end):
        self.x = float(end.x - start.x)
        self.y = float(end.y - start.y)


class RayP(ge.Ray):
    def __init__(self, source, vector):
        ge.Ray.__init__(self, source, vector)
        self.source = PointP(self.source.x, self.source.y)

    def __str__(self):
        return "RayP object: the source is (%d, %d), vector is (%f, %f)" \
               % (self.source.x, self.source.y, self.vector.x, self.vector.y)

    def table_to_view(self):
        ray = self.perspective(_table_to_view)
        ray.source = ray.source.int()
        return ray

    def view_to_table(self):
        ray = self.perspective(_view_to_table)
        ray.source = ray.source.int()
        return ray

    def table_to_canvas(self):
        ray = self.perspective(_table_to_canvas)
        ray.source = ray.source.int()
        return ray


class Ray2P(ge.Line2, RayP):
    def __init__(self, start, end):
        ge.Line2.__init__(self, start, end)
        self.vector = ge.Vector(end.x - start.x, end.y - start.y)
        self.source = start.int()


class Block(object):
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def __str__(self):
        return '%s - %s' % (self.point1, self.point2)

    def center(self):
        return PointP((self.point1.x + self.point2.x) / 2, (self.point1.y + self.point2.y) / 2)
