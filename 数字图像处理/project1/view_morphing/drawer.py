import cv2


def draw_delaunay(src, simplices, points):
    src = src.copy()
    for tri in simplices:
        p1, p2, p3 = points[tri]
        p1 = tuple(p1)
        p2 = tuple(p2)
        p3 = tuple(p3)
        src = cv2.line(src, p1, p2, (0x80, 0xff, 0), 1)
        src = cv2.line(src, p2, p3, (0x80, 0xff, 0), 1)
        src = cv2.line(src, p3, p1, (0x80, 0xff, 0), 1)

    for point in points:
        src = cv2.circle(src, tuple(point), 3, (0xe0, 0xb2, 0), cv2.FILLED)

    return src


def draw_points(src, points):
    src = src.copy()
    for x, y in points:
        src = cv2.circle(src, (x,y), 1, (255,0,0), cv2.FILLED)
    return src
