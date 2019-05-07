from util.file import load_dat, save_dat
from util.calibrator import Corner

_calibrator_path = "./data/paramleft"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 1024

TABLE_WIDTH = 152
TABLE_HEIGHT = 137

CANVAS_TABLE_RATIO = 8

CANVAS_WIDTH = TABLE_WIDTH * CANVAS_TABLE_RATIO
CANVAS_HEIGHT = TABLE_HEIGHT * CANVAS_TABLE_RATIO

# key name
_TABLE_CORNER_VIEW_KN = "table_corner_view"

_DRAW_CORNER_PROJECTION_KN = 'draw_corner_projection'
_DRAW_CORNER_VIEW_KN = 'draw_corner_view'

_VIEW2TABLE_MATRIX_KN = "view2table_matrix"
_TABLE2CANVAS_MATRIX_KN = "table2canvas_matrix"
_CANVAS2PROJECTION_MATRIX_KN = "canvas2projection_matrix"


_CAMERA_MATRIX = 'camera_matrix'
_CAMERA_TUNED_MATRIX = 'camera_tuned_matrix'
_CAMERA_DISTORTION = 'camera_distortion'
# table corner


def load_table_corner_view():
    """
    load last tuning table corner data
    or new corner from memory
    """
    corner = load_cali_dat(_TABLE_CORNER_VIEW_KN)
    if corner is None:
        corner = Corner()
        corner.randomize(CAMERA_WIDTH, CAMERA_HEIGHT)
    return corner


def save_table_corner_view(corner):
    save_cali_dat(_TABLE_CORNER_VIEW_KN, corner)


# projector corner


def load_draw_corner_projection():
    """
    load last tuning projector corner data
    or new corner from memory
    """
    corner = load_cali_dat(_DRAW_CORNER_PROJECTION_KN)
    if corner is None:
        corner = Corner()
        corner.randomize(PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    return corner


def save_draw_corner_projection(corner):
    save_cali_dat(_DRAW_CORNER_PROJECTION_KN, corner)


def load_draw_corner_view():
    """
    load last tuning projector corner data
    or new corner from memory
    """
    corner = load_cali_dat(_DRAW_CORNER_VIEW_KN)
    if corner is None:
        corner = Corner()
        corner.randomize(CAMERA_WIDTH, CAMERA_HEIGHT)
    return corner


def save_draw_corner_view(corner):
    save_cali_dat(_DRAW_CORNER_VIEW_KN, corner)


# matrix

def save_view2table_matrix(matrix):
    save_cali_dat(_VIEW2TABLE_MATRIX_KN, matrix)


def load_view2table_matrix():
    matrix = load_cali_dat(_VIEW2TABLE_MATRIX_KN)
    return matrix


def save_canvas2projection_matrix(matrix):
    save_cali_dat(_CANVAS2PROJECTION_MATRIX_KN, matrix)


def load_canvas2projection_matrix():
    matrix = load_cali_dat(_CANVAS2PROJECTION_MATRIX_KN)
    return matrix


def save_table2canvas_matrix(matrix):
    save_cali_dat(_TABLE2CANVAS_MATRIX_KN, matrix)


def load_table2canvas_matrix():
    matrix = load_cali_dat(_TABLE2CANVAS_MATRIX_KN)
    return matrix


def save_camera_matrix(matrix):
    save_cali_dat(_CAMERA_MATRIX, matrix)


def load_camera_matrix(path):
    matrix = load_cali_dat(path,_CAMERA_MATRIX)
    return matrix


def save_camera_matrix_tuned(matrix):
    save_cali_dat(_CAMERA_TUNED_MATRIX, matrix)


def load_camera_matrix_tuned(path):
    matrix = load_cali_dat(path,_CAMERA_TUNED_MATRIX)
    return matrix


def save_camera_distortion(matrix):
    save_cali_dat(_CAMERA_DISTORTION, matrix)


def load_camera_distortion(path):
    matrix = load_cali_dat(path,_CAMERA_DISTORTION)
    return matrix


# Base function
def load_cali_dat(path,key):
    data = load_dat(path, key)
    return data


def save_cali_dat(key, value):
    save_dat(_calibrator_path, key, value)
