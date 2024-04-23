import os
import cv2
import glob
import argparse

class VideoControler:
    def video_to_image(args):

        videoPath = os.path.join(args.dir, args.name)
        print(videoPath)
        cap = cv2.VideoCapture(videoPath)

        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_fps = cap.get(cv2.CAP_PROP_FPS)

        success, frame = cap.read()
        count = 0

        middle_path = args.name.split(".")[0]
        writePath = os.path.join(args.dir, middle_path)
        if not os.path.exists(writePath):
            os.mkdir(writePath)
        
        while success:
            cv2.imwrite(writePath + "\\%06d.jpg" % count, frame)
            success, frame = cap.read()
            print('Read a new Frame : ', success)
            count += 1

        cap.release()


    def image_to_video(args):
        middle_path = args.name.split(".")[0]
        writePath = os.path.join(args.dir, middle_path)

        filenames = os.listdir(writePath)
        filenames.sort()

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(os.path.join(writePath, f"dehaze_{args.name}.mp4"), fourcc, 30, (1920, 1080))

        for filename in filenames:
            cap = cv2.imread(writePath + "\\" + filename)
            out.write(cap)

        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='C:\\Users\\seyou\\Desktop\\FromSE\\', type=str, help='dataset name')
    parser.add_argument('--name', default='1400.mp4', type=str, help='dataset name')
    args = parser.parse_args()

    vid = VideoControler()
    vid.video_to_image(args)
    vid.image_to_video(args)
