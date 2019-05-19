import os
import shelve


def load_dat(dat_path, key):
    """
    load sections and contents from dat file
    :param dat_path: dat file path
    :param key: key to load
    :return: sections and contents
    """
    if key is None:
        raise KeyError('cannot load NoneType key')
    # if not os.path.exists(dat_path):
    #     raise IOError('cannot find file: %s' % dat_path)
    dat = shelve.open(dat_path)
    try:
        response = dat[key]
    except KeyError as e:
        # raise e
        return None
    finally:
        dat.close()
    return response


def save_dat(dat_path, key, value):
    """
    save (key, value) into dat file
    :param key: key
    :param value: value
    :param dat_path: dat file path
    """
    if key is None:
        raise KeyError('cannot load NoneType key')
    # if not os.path.exists(dat_path):
    #     raise IOError('cannot find file: %s' % dat_path)
    dat = shelve.open(dat_path)
    dat[key] = value
    dat.close()
