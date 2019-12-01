import cv2
from imutils import paths
import numpy as np
import argparse
import imutils

def stitch(image1, image2, lowe_ratio=0.75, max_t=4.0, is_match=False):
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    descriptors = cv2.xfeatures2d.SIFT_create()
    (key_points_2, features_2) = descriptors.detectAndCompute(image2, None)
    key_points_2 = np.float32([i.pt for i in key_points_2])

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    descriptors = cv2.xfeatures2d.SIFT_create()
    (key_points_1, features_1) = descriptors.detectAndCompute(image1, None)
    key_points_1 = np.float32([i.pt for i in key_points_1])

    match_instance = cv2.DescriptorMatcher_create("BruteForce")
    all_matches = match_instance.knnMatch(features_2, features_1, 2)

    matches = []

    for val in all_matches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            matches.append((val[0].trainIdx, val[0].queryIdx))

    if len(matches) < 5:
        return None

    points2 = np.float32([key_points_2[i] for (_,i) in matches])
    points1 = np.float32([key_points_1[i] for (i,_) in matches])

    (Homography, status) = cv2.findHomography(points2, points1, cv2.RANSAC, max_t)
    #print(points1)
    #print(points2)
    #print(points1.shape)
    #print(points2.shape)

    # Code for bundle adjustment from scratch begins here
    
    arr=[]
    optimaldistances=list()
    removeddistances=list()
    for i in range(points1.shape[0]):
        diff=pow((points2[i][0] - points1[i][0]),2) + pow((points2[i][1] - points1[i][1]),2)
        d=sqrt(diff)                   # Calculating the euclidean distance
        arr.append(d)
    std1=stdev(arr)                    # Finding standard deviation from the distances obtained
    m1=mean(arr)                       # Finding the mean of the array
    indexes=[]                         # Creating an index array 
    for i in range(len(arr)):           
        if(arr[i]>(m1 - 2*std1)):      # Defining a threshhold
            optimaldistances.append(arr[i])
            indexes.append(i)          # Appending the indexes of the key points that are relevant
        else:
            removeddistances.append(arr[i])
    keypoints1update=list()
    keypoints2update=list()
    
    # Appending the co-ordinates of the new points obtained after bundle adjustment
    for i in indexes:
        keypoints1update.append((points1[i][0],points1[i][1]))  
        keypoints2update.append((points2[i][0],points2[i][1]))
    
    print("The removed distances are: ")
    for i in range(len(removeddistances)):
        print(removeddistances[i])
    
    prevlength=points1.shape[0]
    newlength=len(keypoints1update)
    print("Size of original key points: ",prevlength)
    print("Size of optimal key points after bundle adjustment ",newlength)

    # Code for bundle adjustment ends here

    val = image2.shape[1] + image1.shape[1]
    result = cv2.warpPerspective(image2, Homography, (val , image2.shape[0]))
    result[0:image1.shape[0], 0:image1.shape[1]] = image1

    if is_match:
        (h2,w2) = image2.shape[:2]
        (h1, w1) = image1.shape[:2]
        vis = np.zeros((max(h2, h1), w2 + w1, 3), dtype="uint8")
        vis[0:h2, 0:w2] = image2
        vis[0:h1, w2:] = image1

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(key_points_2[queryIdx][0]), int(key_points_2[queryIdx][1]))
                ptB = (int(key_points_1[trainIdx][0]) + w2, int(key_points_1[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return (result, vis)

    return result

# Code for straightening from scratch starts here
def straightening(result):
    #count=0
    x,y,rgb=result.shape           
    newresultx=[]
    newresulty=[]
    #print(len(result))
    for i in range(x):
        for j in range(y):
            k=result[i,j]
            if(k[0]!=0 and k[1]!=0 and k[2]!=0):           # Find the portion where we have non zero pixel values
                newresultx.append(i)                       # Add the corresponding x-coordinate            
                newresulty.append(j)                       # Add the corresponding y-coordinate
    result=result[0:max(newresultx),0:max(newresulty)]     # Store the resized image in the result(removing the black portion)
    #print("-",len(result))
    return result           

    #print(count)

# Code for straightening ends here

def get_input(image_args):
    filename = []
    image_paths = sorted(list(paths.list_images(image_args)))
    images = []

    for path in image_paths:
        x = cv2.imread(path)
        x = imutils.resize(x,width=400)
        x = imutils.resize(x,height=400)
        images.append(x)
    return images


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--images", type=str, required=True)
    arg.add_argument("-o", "--output", type=str, required=True)
    args = vars(arg.parse_args())

    images = get_input(args["images"])

    if(len(images)) == 2:
        (result, Z) = stitch(images[0], images[1], is_match=True)
    else:
        (result, Z) = stitch(images[-2], images[-1], is_match=True)
        for i in range(len(images) - 2):
            (result, Z) = stitch(images[len(images)-i-3], result, is_match=True)

    cv2.imwrite("output.jpg", result)
    # Panaromic straightened image
    result1=straightening(result)
    cv2.imwrite("output1.jpg", result1)
