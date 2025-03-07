def return_bounding_box_points(bbox):
    x_b_1 = int(bbox[0] - (bbox[3] / 2))
    y_b_1 = int(bbox[1] - (bbox[2] / 2))
    x_b_2 = int(bbox[0] + (bbox[3] / 2))
    y_b_2 = int(bbox[1] + (bbox[2] / 2))
    p_1 = (x_b_1, y_b_1)
    p_2 = (x_b_2, y_b_2)

    return p_1, p_2
