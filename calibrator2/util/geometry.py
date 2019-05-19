import numpy


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'Point: (%f, %f)' % (self.x, self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False

    def __mul__(self, times):
        return Point(self.x * times, self.y * times)

    def __rmul__(self, times):
        return Point(self.x * times, self.y * times)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __tuple__(self):
        return tuple([self.x, self.y])

    def update(self, x, y):
        self.x = x
        self.y = y

    def distance_to_line(self, line):
        return abs(line.A * self.x + line.B * self.y + line.C) / ((line.A ** 2 + line.B ** 2) ** 0.5)

    def distance_to_point(self, p):
        return ((self.x - p.x) ** 2 + (self.y - p.y) ** 2) ** 0.5

    def get_perpendicular(self, line):
        A = line.B
        B = -line.A
        C = -A * self.x - B*self.y
        return Line(A, B, C)

    def int(self):
        self.x = int(round(self.x))
        self.y = int(round(self.y))
        return self

    def tuple(self):
        return tuple([self.x, self.y])

    def list(self):
        return [self.x, self.y]

    def project_to_line(self, line):
        perp = self.get_perpendicular(line)
        pt = perp.intersection(line)
        return pt

    def perspective(self, mat):
        x = self.x
        y = self.y
        rx = (mat[0][0] * x + mat[0][1] * y + mat[0][2]) / (mat[2][0] * x + mat[2][1] * y + mat[2][2])
        ry = (mat[1][0] * x + mat[1][1] * y + mat[1][2]) / (mat[2][0] * x + mat[2][1] * y + mat[2][2])
        return Point(rx, ry)

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def angle(self, p):
        value = (self.x * p.x + self.y * p.y) / (self.norm() * p.norm())
        return int(round(numpy.arccos(value) * 180.0 / numpy.pi))


class Line(object):

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def __str__(self):
        return 'A = %d, B = %d, C = %d' % (self.A, self.B, self.C)

    def intersection(self, line):
        x = (line.B * self.C - self.B * line.C) / (line.A * self.B - self.A * line.B)
        y = (line.A * self.C - self.A * line.C) / (self.A * line.B - line.A * self.B)
        return Point(x, y)


class Line2(Line):

    def __init__(self, p1, p2):
        self.A = p2.y - p1.y
        self.B = p1.x - p2.x
        self.C = p2.x * p1.y - p1.x * p2.y


class Segment(Line2):

    def __init__(self, p1, p2):
        Line2.__init__(self, p1, p2)
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return 'Segment is (%f, %f), (%f, %f)' % (self.p1.x, self.p1.y, self.p2.x, self.p2.y)


class Vector(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vector(float(self.x + other.x), float(self.y + other.y))

    def __sub__(self, other):
        return Vector(float(self.x - other.x), float(self.y - other.y))

    def __radd__(self, other):
        return Vector(float(self.x + other.x), float(self.y + other.y))

    def __rsub__(self, other):
        return Vector(float(other.x - self.x), float(other.y - self.y))

    def __mul__(self, times):
        return Vector(self.x * times, self.y * times)

    def __rmul__(self, times):
        return Vector(self.x * times, self.y * times)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __str__(self):
        return '(%f, %f)' % (self.x, self.y)

    def update(self, x, y):
        self.x = x
        self.y = y

    def normalize(self):
        norm = (self.x ** 2 + self.y ** 2) ** 0.5
        self.x /= norm
        self.y /= norm
        return Vector(self.x, self.y)

    def dis(self):
        distance = (self.x ** 2 + self.y ** 2) ** 0.5
        return distance

    def tuple(self):
        return tuple([self.x, self.y])

    def list(self):
        return [self.x, self.y]

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def perspective(self, mat):
        x = self.x
        y = self.y
        rx = (mat[0][0] * x + mat[0][1] * y + mat[0][2]) / (mat[2][0] * x + mat[2][1] * y + mat[2][2])
        ry = (mat[1][0] * x + mat[1][1] * y + mat[1][2]) / (mat[2][0] * x + mat[2][1] * y + mat[2][2])
        return Point(rx, ry)


class Vector2(Vector):
    def __init__(self, start, end):
        self.x = float(end.x - start.x)
        self.y = float(end.y - start.y)


def compute_cos(Vecter_a, Vecter_b):
    m = Vecter_a.x * Vecter_b.x + Vecter_a.y * Vecter_b.y
    n = ((Vecter_a.x ** 2 + Vecter_a.y ** 2) * (Vecter_b.x ** 2 + Vecter_b.y ** 2)) ** 0.5
    return m / n


class Ray(Line):
    def __init__(self, source, vector):
        assert isinstance(source, Point)
        assert isinstance(vector, Vector)
        self.source = source
        self.vector = vector
        self.__get_abc()

    def __str__(self):
        return "the source is (%d, %d), vector is (%f, %f)" % (self.source.x, self.source.y, self.vector.x, self.vector.y)

    def __get_abc(self):
        self.A = self.vector.y
        self.B = -self.vector.x
        self.C = self.source.y*self.vector.x - self.source.x * self.vector.y

    def perspective(self, trans_matrix):
        vector = self.vector.normalize()
        start_point = self.source.perspective(trans_matrix)
        temp_point = Point(self.source.x + vector.x * 8000, self.source.y + vector.y * 8000)
        end_point = temp_point.perspective(trans_matrix)
        shoot_ray = Ray2(start_point, end_point)
        return shoot_ray

    def cross_test_segment(self, segment):
        end = self.source + 100*self.vector
        return cross_test(segment.p1, segment.p2, self.source, end)


class Ray2(Line2, Ray):
    def __init__(self, start, end):
        Line2.__init__(self, start, end)
        self.source = start
        self.vector = Vector(end.x - start.x, end.y - start.y)


def list_to_points(lists):
    balls = []
    for pos in lists:
        ball = Point(pos[0], pos[1])
        balls.append(ball)
    return balls


def points_to_list(lists):
    balls = []
    for pos in lists:
        ball = [pos.x, pos.y]
        balls.append(ball)
    return balls


def dot_product(p, q):
    return p.x*q.x + p.y*q.y


def cross_product(p, q):
    return p.x*q.y - p.y*q.x


def cross_test(p1, p2, l1, l2):
    # Test segment and line intersection.
    r1 = cross_product(p1 - l1, l2 - l1)
    r2 = cross_product(l2 - l1, p2 - l2)
    r = r1*r2
    if r >= 0:
        return True
    else:
        return False
