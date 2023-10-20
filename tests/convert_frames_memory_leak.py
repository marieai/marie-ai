import os
import cv2
import numpy as np
import tracemalloc

from marie.utils.docs import convert_frames

tracemalloc.start()
if __name__ == "__main__":
    frame = np.zeros((480, 640), dtype=np.uint8)   # No memory leaks, successful conversion
    # frame = np.zeros((480, 640, 3), dtype=np.float64)   # Memory leak, failed conversion
    N = 1_000_000 # N = 1_000_000 will result (~1.3 GB reserved memory)
    for i in range(N):
        # Measure memory usage
        if (i % (N//10) == 0):
            memory_usage = os.popen('free -h').readlines()[1].split()
            print(("i = %06d | Memory used %s from %s")%(i, memory_usage[2], memory_usage[1]))
        # Attempt conversion
        try:
            frames = [frame]
            gray = convert_frames([frame], img_format='pil')
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except  Exception as e:
            pass    # For the sake of simplicity I pass here, the exception did get logged in the actual application

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)


# ref : https://github.com/opencv/opencv/issues/23633