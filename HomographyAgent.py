import os

import numpy as np
import cv2
import cv2
from ffpyplayer.player import MediaPlayer


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(x, ' ', y)


def print_debug_version(flag,item_to_print):
    if not flag:
        return
    print(item_to_print)
def show_images_debug(flag,item_to_show):
    if not flag:
        return
    cv2.imshow('item_to_show',item_to_show)
    cv2.waitKey(0)
class HomographyAgent():

    def __init__(self,debug_version):
        self.flag = debug_version

    def addTwoImages(self, path_first_image, path_second_image):
        first_image = cv2.imread(path_first_image)
        show_images_debug(self.flag,first_image)
        # cv2.imshow("img",first_image)
        # cv2.waitKey(0)
        second_image = cv2.imread(path_second_image)
        show_images_debug(self.flag, second_image)

        first_image = cv2.resize(first_image, (512, 512))
        second_image = cv2.resize(second_image, (512, 512))

        final_image = cv2.addWeighted(first_image, 0.5, second_image, 0.5, 0.0)

        cv2.imshow("combine_image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def Panorama(self,folder_dir):
        images = []
        for image in os.listdir(folder_dir):
            Img = cv2.imread(f'{folder_dir}/{image}')
            show_images_debug(self.flag, Img)
            img = cv2.resize(Img, (512, 512))
            images.append(img)

        stitcher = cv2.Stitcher.create()
        (status, result) = stitcher.stitch(images)
        if status == cv2.STITCHER_OK:
            print("OK")
            cv2.imshow("result", result)
            cv2.waitKey(0)
        else:
            print("NOT OK")


    def PrespectiveCorrection(self,image_path):

        global points
        points = []
        src_img = cv2.imread(image_path)
        # src_img = cv2.resize(src_img, (512, 512))
        cv2.imshow('image', src_img)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        pts_src = points
        pts_src = np.array(pts_src, dtype=np.float32)
        print("pts_src", pts_src)
        cv2.waitKey(0)
        pts_dst = [[0, 0], [512 - 1, 0], [512 - 1, 512 - 1], [0, 512 - 1]]
        pts_dst = np.array(pts_dst, dtype=np.float32)
        print(pts_dst)

        transform_matrix, mask = cv2.findHomography(pts_src, pts_dst)

        output = cv2.warpPerspective(src_img, transform_matrix, (512, 512))
        cv2.imshow('output', output)
        cv2.waitKey(0)


    def replacePartOfImage(self,dst_image):
        want_example = input("you need to choice 4 points and than square will be replaced by the second image you will insert\n\n do you want to watch an example ? [y/n] ")
        if want_example == 'y':
            self.DisplayAvideo('asa')

        global points
        points = []
        dst_image = cv2.imread(dst_image)
        cv2.imshow('image', dst_image)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        dst_points = np.float32(points)
        print_debug_version(self.flag,dst_points)
        # print(dst_points)

        src_image = input("insert your src image path \n")
        src_image = cv2.imread(src_image)
        show_images_debug(self.flag,src_image)
        src_points = np.float32([[0, 0], [src_image.shape[1], 0], [src_image.shape[1], src_image.shape[0]], [0, src_image.shape[0]]])
        h = cv2.getPerspectiveTransform(src_points, dst_points)
        src_warped = cv2.warpPerspective(src_image, h, (dst_image.shape[1], dst_image.shape[0]))

        cv2.imwrite("ladies_in_small.jpg", src_warped)
        mask = cv2.bitwise_not(src_warped)

        white_img = cv2.imread('white_board.png')
        white_img = cv2.resize(white_img, (512, 512))
        src_green_points = np.float32(
            [[0, 0], [src_image.shape[1], 0], [src_image.shape[1], src_image.shape[0]], [0, src_image.shape[0]]])
        h = cv2.getPerspectiveTransform(src_green_points, dst_points)
        src_green_warped = cv2.warpPerspective(white_img, h, (dst_image.shape[1], dst_image.shape[0]))

        help_pic = cv2.add(src_green_warped, dst_image)
        show_images_debug(self.flag,help_pic)

        cv2.waitKey(0)
        cv2.imwrite("help_pic.png", help_pic)

        dst = cv2.imread('help_pic.png')

        dst = cv2.resize(dst, (512, 512))
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

        inv_mask = cv2.imread("ladies_in_small.jpg")
        inv_mask = cv2.resize(inv_mask, (512, 512))
        dst = cv2.resize(dst, (512, 512))
        lower_white = np.array([0, 0, 0], dtype=np.uint8)
        upper_white = np.array([0, 0, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(dst, dst, mask=mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        #
        #
        inv_mask = cv2.bitwise_not(mask)
        #
        # # Mask dst with inverted mask
        dst_masked = cv2.bitwise_and(dst, dst, mask=inv_mask)
        src_warped = cv2.imread('ladies_in_small.jpg')
        src_warped = cv2.resize(src_warped, (512, 512))

        # Put src_warped over dst
        result = cv2.add(dst_masked, src_warped)
        cv2.imshow('result', result)
        cv2.waitKey(0)


    def DisplayAvideo(self,video_path):


        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture('example.mp4')
        player = MediaPlayer('example.mp4')
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            audio_frame , val = player.get_frame()
            if ret == True:

                # Display the resulting frame

                cv2.imshow('Frame', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()


    def applyHomograpyWithCorrespondingPoints(self,first_image_path,second_image_path):
        global points
        points = []
        # Read source image.
        im_src = cv2.imread(first_image_path) #book2
        cv2.imshow('image', im_src)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        # Four corners of the book in source image
        # pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]]) #good
        # pts_src = np.array([[54,354], [235,190], [396,281], [234,498]])
        pts_src = np.array(points)
        points = []
        # Read destination image.
        im_dst = cv2.imread(second_image_path) #book1
        # Four corners of the book in destination image.
        # pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]]) #good
        cv2.imshow('image', im_dst)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)

        # pts_dst = np.array([[46,448], [109,103], [354,121], [446,595]])
        pts_dst = np.array(points)
        # Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

        # Display images
        cv2.imshow("Source Image", im_src)
        cv2.imshow("Destination Image", im_dst)
        cv2.imshow("Warped Source Image", im_out)

        cv2.waitKey(0)