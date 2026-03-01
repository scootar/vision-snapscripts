import cv2
import numpy as np

DIST_THRESHOLD = 20.0

def runPipeline(image, llrobot):

    # ---- Example detection stage ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (11, 149, 154), (29, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.array([[]]), image, [0]

    # ---- Build Nx4 array of boxes ----
    # columns: left, top, right, bottom
    boxes = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append([x, y, x+w, y+h])
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)

    boxes = np.array(boxes, dtype=np.float32)

    left   = boxes[:,0][:,None]
    top    = boxes[:,1][:,None]
    right  = boxes[:,2][:,None]
    bottom = boxes[:,3][:,None]

    # ---- Vectorized overlap test ----
    overlap = ~(
        (right  <= left.T)  |
        (left   >= right.T) |
        (bottom <= top.T)   |
        (top    >= bottom.T)
    )

    # ---- Vectorized edge distance ----
    n = len(boxes)
    zeros = np.zeros((n, n), dtype=np.float32)

    dx = np.maximum.reduce([
        left.T - right,
        left - right.T,
        zeros
    ])

    dy = np.maximum.reduce([
        top.T - bottom,
        top - bottom.T,
        zeros
    ])

    dist_sq = dx*dx + dy*dy

    # ---- Connectivity condition ----
    adjacency = overlap | (dist_sq <= DIST_THRESHOLD*DIST_THRESHOLD)

    # Remove self-loops
    np.fill_diagonal(adjacency, False)

    # ---- Connected components (simple DFS) ----
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        component = []

        while stack:
            idx = stack.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            component.append(idx)

            neighbors = np.where(adjacency[idx])[0]
            for nb in neighbors:
                if not visited[nb]:
                    stack.append(nb)

        clusters.append(component)

    # ---- Merge clusters ----
    merged = []

    for comp in clusters:
        cluster_boxes = boxes[comp]

        minL = np.min(cluster_boxes[:,0])
        minT = np.min(cluster_boxes[:,1])
        maxR = np.max(cluster_boxes[:,2])
        maxB = np.max(cluster_boxes[:,3])

        weight = len(comp)

        merged.append((minL, minT, maxR-minL, maxB-minT, weight))

        cv2.rectangle(
            image,
            (int(minL), int(minT)),
            (int(maxR), int(maxB)),
            (255,0,0),
            2
        )

    # ---- Pack output ----
    llpython = [len(merged)]
    for x,y,w,h,wt in merged:
        llpython += [float(x), float(y), float(w), float(h), float(wt)]

    return np.array([[]]), image, llpython
