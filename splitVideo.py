import cv2

count = 0
scale = 1
vidcap = cv2.VideoCapture('juggling_Trim2.mp4')
success, image = vidcap.read()

while success:
    image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    cv2.imwrite(r"frames_Trim2\frame{}.jpg".format(count), image)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
