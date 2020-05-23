from src_james.ensemble.colors import object_dict, object_01area_dict, object_color_dict


def max_mode(A1):
    stats, name_dic = object_dict(A1)
    max_key = max(stats, key=lambda k: stats[k])
    return max_key, name_dic


def min_mode(A1):
    stats, name_dic = object_dict(A1)
    min_key = min(stats, key=lambda k: stats[k])
    return min_key, name_dic


# object_01area_dict
def max_01_area(A1):
    stats, name_dic = object_01area_dict(A1)
    max_key = max(stats, key=lambda k: stats[k])
    return max_key, name_dic


def min_01_area(A1):
    stats, name_dic = object_01area_dict(A1)
    min_key = min(stats, key=lambda k: stats[k])
    return min_key, name_dic


mode_select = [max_mode, min_mode, max_01_area, min_01_area]


def max_color_mode(A1, c):
    if object_color_dict(A1, c):
        stats, name_dic = object_color_dict(A1, c)
        max_key = max(stats, key=lambda k: stats[k])
        return max_key, name_dic
    else:
        return False


def min_color_mode(A1, c):
    if object_color_dict(A1, c):
        stats, name_dic = object_color_dict(A1, c)
        min_key = min(stats, key=lambda k: stats[k])
        return min_key, name_dic
    else:
        return False
   