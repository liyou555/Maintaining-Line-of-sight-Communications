import sys
import argparse

from jetson_utils import videoSource, videoOutput,cudaToNumpy
import time
import cv2
import matplotlib.pyplot as plt
# parse command line
parser = argparse.ArgumentParser()
# parser.add_argument("input", type=str, help="URI of the input stream")
parser.add_argument("input", type=str, default="csi://0", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

args = parser.parse_known_args()[0]

# create video sources & outputs
output = videoOutput(args.output, argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000
input = videoSource(args.input, argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
# output = videoOutput(args.output, argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}
# import tensorflow as tf
# capture frames until end-of-stream (or the user exits)
while True:
    # format can be:   rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be:  -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    image = input.Capture(format='rgb8', timeout=1000)

    if image is None:  # if a timeout occurred
        continue

    # 保持图像显示3秒
    # print("Displaying image for 3 seconds...")
    # start_time = time.time()
    # while time.time() - start_time < 10:  # 持续 3 秒
    #     output.Render(image)  # 重复渲染同一帧，确保窗口刷新不关闭
        
    output.Render(image)
    
    image = cudaToNumpy(image)
    plt.imsave("image.jpg", image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # plt.imsave("image.jpg", image)
    print("save image")
		
    # output.Render(image)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
        