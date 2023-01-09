
import HomographyAgent as HomographyAgent


def main(homography_agent):

    while(True):
        first_input = input("insert your choice:\n 1) combine two images \n 2) create a panorama \n 3) perspective correction \n 4) replace part of an image \n 5) apply Homography With Corresponding Points \n 6) Exit \n")
        match first_input:
            case '1':
                first_image_path = input("insert the full path of your first image\n")
                second_image_path = input("insert the full path of your second image\n")
                homography_agent.addTwoImages(first_image_path,second_image_path)
            case '2':
                folder_path = input("insert the full path of your folder which contain the candidates images for panorama\n")
                homography_agent.Panorama(folder_path)
            case '3':
                image_path = input("insert the full path of your image \n")
                homography_agent.PrespectiveCorrection(image_path)
            case '4':
                dst_image = input("insert your destination full path image \n")
                homography_agent.replacePartOfImage(dst_image)
            case '5':
                first_image_path = input("insert the full path of your first image\n")
                second_image_path = input("insert the full path of your second image\n")
                homography_agent.applyHomograpyWithCorrespondingPoints(first_image_path,second_image_path)
            case '6':
                print("logout")
                break
            case default:
                print("logout")
                break


if __name__ == '__main__':
    debug_version = input("do you want to run in a debug version? [y/n] \n")
    if debug_version == 'y':
        debug_version = True
    else:
        debug_version = False
    homography_agent = HomographyAgent.HomographyAgent(debug_version)
    main(homography_agent)