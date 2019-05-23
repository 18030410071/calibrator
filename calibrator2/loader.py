from util.file import load_dat, save_dat
from util.calibrator import Corner

_calibrator_path = "./data/param"

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

_CAMERA_MATRIX_R = 'camera_matrix_r'
_CAMERA_TUNED_MATRIX_R = 'camera_tuned_matrix_r'
_CAMERA_DISTORTION_R = 'camera_distortion_r'


_CAMERA_PROJ_STEREO_MATRIX = 'camera_proj_stereo_matrix'
_CAMERA_PROJ_STEREO_MATRIX_R = 'camera_proj_stereo_matrix_r'

_CAMERA_PROJ_OWN_MATRIX = 'camera_proj_own_matrix'
_CAMERA_PROJ_OWN_MATRIX_R = 'camera_proj_own_matrix_r'

_CAMERA_ROT_MATRIX = 'camera_rot_matrix'
_CAMERA_ROT_MATRIX_R = 'camera_rot_matrix_r'

_CAMERA_Q_MATRIX = 'camera_q_matrix'



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


def load_camera_matrix():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_MATRIX)
    return matrix


def save_camera_matrix_tuned(matrix):
    save_cali_dat(_CAMERA_TUNED_MATRIX, matrix)


def load_camera_matrix_tuned():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_TUNED_MATRIX)
    return matrix


def save_camera_distortion(matrix):
    save_cali_dat(_CAMERA_DISTORTION, matrix)


def load_camera_distortion():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_DISTORTION)
    return matrix


# camera_r

def save_camera_matrix_r(matrix):
    save_cali_dat(_CAMERA_MATRIX_R, matrix)


def load_camera_matrix_r():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_MATRIX_R)
    return matrix


def save_camera_matrix_tuned_r(matrix):
    save_cali_dat(_CAMERA_TUNED_MATRIX_R, matrix)


def load_camera_matrix_tuned_r():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_TUNED_MATRIX_R)
    return matrix


def save_camera_distortion_r(matrix):
    save_cali_dat(_CAMERA_DISTORTION_R, matrix)


def load_camera_distortion_r():
    matrix = load_cali_dat(_calibrator_path,_CAMERA_DISTORTION_R)
    return matrix

# P1_stereo
def save_camera_matrix_stereo_proj(matrix):
    save_cali_dat(_CAMERA_PROJ_STEREO_MATRIX, matrix)
def load_camera_matrix_stereo_proj():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_PROJ_STEREO_MATRIX)
    return matrix
# P2_stereo
def save_camera_matrix_stereo_proj_r(matrix):
    save_cali_dat(_CAMERA_PROJ_STEREO_MATRIX_R, matrix)
def load_camera_matrix_stereo_proj_r():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_PROJ_STEREO_MATRIX_R)
    return matrix

# P1_own
def save_camera_matrix_own_proj(matrix):
    save_cali_dat(_CAMERA_PROJ_OWN_MATRIX, matrix)
def load_camera_matrix_own_proj():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_PROJ_OWN_MATRIX)
    return matrix
# P2_own
def save_camera_matrix_own_proj_r(matrix):
    save_cali_dat(_CAMERA_PROJ_OWN_MATRIX_R, matrix)
def load_camera_matrix_own_proj_r():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_PROJ_OWN_MATRIX_R)
    return matrix


# R1
def save_camera_matrix_rot(matrix):
    save_cali_dat(_CAMERA_ROT_MATRIX, matrix)
def load_camera_matrix_rot():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_ROT_MATRIX)
    return matrix
# R2
def save_camera_matrix_rot_r(matrix):
    save_cali_dat(_CAMERA_ROT_MATRIX_R, matrix)
def load_camera_matrix_rot_r():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_ROT_MATRIX_R)
    return matrix
# Q
def save_camera_matrix_q(matrix):
    save_cali_dat(_CAMERA_Q_MATRIX, matrix)
def load_camera_matrix_q():
    matrix = load_cali_dat(_calibrator_path, _CAMERA_Q_MATRIX)
    return matrix


# Base function
def load_cali_dat(path,key):
    data = load_dat(path, key)
    return data


def save_cali_dat(key, value):
    save_dat(_calibrator_path, key, value)
