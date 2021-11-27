# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:24:27 2021

@author: vasil

CODE FOR PART 3 - RANSAC
"""

import cv2
import numpy as np
import getopt
import sys
import random
import scipy.io
from skimage.morphology import binary_dilation
from skimage import io
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt
#
# Read in an image file, errors out if we can't find the file
#
MIN_MATCH_COUNT = 10
def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img

def reProjErr(corrs, h):
    p1reg = cv2.warpPerspective(corrs[:,0:2],h,corrs[:,0:2].shape)
    hinv = np.linalg.inv(finalH)/np.linalg.inv(h)[2,2]
    p2reg = cv2.warpPerspective(corrs[:,2:4],hinv,corrs[:,2:4].shape)
    
    p1reg_dist = np.linalg.norm(corrs[:,2:4].T-p1reg)
    p2reg_dist = np.linalg.norm(corrs[:,0:2].T-p2reg)
    
    return p1reg_dist, p2reg_dist

# This draws matches and optionally a set of inliers in a different color
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

#
# Runs sift algorithm to find features
#
def findFeatures(img):
    print("Finding Features...")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints,img)
    cv2.imwrite('sift_keypoints.png', img)

    return keypoints, descriptors

def findFeaturesWithKp(img, KP):
    print("Finding Features...")
    sift = cv2.SIFT_create()
    # KP, descriptors = sift.detectAndCompute(img,cv2.UMat(canvas)) # change KP to "useless" if you wanna use professor's Key Points
    useless, descriptors = sift.compute(img,KP) # change KP to "useless" if you wanna use professor's Key Points

    img = cv2.drawKeypoints(img, KP, img)
    cv2.imwrite('sift_keypoints.png', img)

    return KP, descriptors

#
# Matches features given a list of keypoints, descriptors, and images
#
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    cv2.imwrite('Matches.png', matchImg)
    return matches

p1reg_dist = []
p2reg_dist = []

for im1 in ['00']:
    for im2 in ['01','02','03']:
        if im1 != im2:
            img1name = "./DataSet01/{}.png".format(im1)
            img2name = "./DataSet01/{}.png" .format(im2)

            print("Image 1 Name: " + img1name)
            print("Image 2 Name: " + img2name)
            
            #query image
            img1 = readImage(img1name)
            # img1 = img1/np.max(img1)
            #train image
            img2 = readImage(img2name)
            # img2 = img2/np.max(img2)
            rows,cols = img1.shape
            
            # Loading features
            # mat = scipy.io.loadmat('./DataSet01/Features.mat')
            # Features = mat['Features'].T
            # features = []
            
            # for i in range (0,Features.shape[0]):
            #     features.append(Features[i][0][0])
            
            # del mat, Features
            # kp1 = np.floor(features[int(img1name[-5])]) # x (kp[:,1]) and y (kp[:,0]) 
            # kp2 = np.floor(features[int(img2name[-5])])
            
            
            # Step 2
            correspondenceList = []
            for mode in ['projective']:
                correspondenceList = []
                if img1 is not None and img2 is not None:
                    
                    sift = cv2.SIFT_create()
                    
                    kp1, desc1 = sift.detectAndCompute(img1, None)
                    kp2, desc2 = sift.detectAndCompute(img2, None)
                    
                    print ("Found keypoints in " + img1name + ": " + str(len(kp1)))
                    print ("Found keypoints in " + img2name + ": " + str(len(kp2)))
                    keypoints = [kp1,kp2]
                    matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
                    for match in matches:
                        (x1, y1) = keypoints[0][match.queryIdx].pt
                        (x2, y2) = keypoints[1][match.trainIdx].pt
                        correspondenceList.append([x1, y1, x2, y2])

                    corrs = np.matrix(correspondenceList)

                    #run ransac algorithm
                    # finalH, inliers = ransac(corrs, estimation_thresh)
                    
                    
                    finalH, inliers = cv2.findHomography(corrs[:,0:2], corrs[:,2:4], cv2.RANSAC,5.0)
                    
                    print ("Final homography: ", finalH)
                    print ("Final inliers count: ", len(inliers))

                    matchImg = drawMatches(img1,kp1,img2,kp2,matches,inliers)
                    cv2.imwrite('InlierMatches.png', matchImg)

                    f = open('homography.txt', 'w')
                    f.write("Final homography: \n" + str(finalH)+"\n")
                    f.write("Final inliers count: " + str(len(inliers)))
                    f.close()

                    matchImg = drawMatches(img1,kp1,img2,kp2,matches,inliers)
                    cv2.imwrite('./Results/RANSAC/{}/InlierMatches_{}_to_{}_{}.png'.format(mode,im1,im2,mode), matchImg)
                
                    f = open('./Results/RANSAC/{}homography_{}_to_{}_{}.txt'.format(mode,im1,im2,mode), 'w')
                    f.write("Final homography: \n" + str(finalH)+"\n")
                    f.write("Final inliers count: " + str(len(inliers)))
                    f.close()
                
                
                    # Actually registering the image
                    dst = cv2.warpPerspective(img1,finalH,(cols,rows))
                    
                    # plt.figure()
                    # io.imshow(img1)
                    # plt.figure()
                    # io.imshow(img2)
                    # plt.figure()
                    # io.imshow(dst)
                    
                    io.imsave('./Results/RANSAC/{}/{}_to_{}_{}.png'.format(mode,im1,im2,mode), dst)
                    
                    f = open('./Results/RANSAC/{},homography_{}_to_{}_{}.txt'.format(mode,im1,im2,mode), 'w')
                    f.write("Final homography: \n" + str(finalH)+"\n")
                    
                    # Reprojection Analysis
                    p1reg_dist.append(reProjErr(corrs, finalH)[0]/255)
                    p2reg_dist.append(reProjErr(corrs, finalH)[1]/255)
                    
                    added_image = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
                    added_image[:,:,1] = dst
                    added_image[:,:,2] = dst
                    # added_image = cv2.addWeighted(img2,0.4,dst,0.1,0)
                    io.imsave('./Results/RANSAC/{}/overlay_{}_to_{}_{}.png'.format(mode,im1,im2,mode), added_image)
                    
                    
                    
print('Projecting matching points from im1 onto im2, errors:{}'.format(p1reg_dist))
print('Projecting matching points from im2 onto im1, errors:{}'.format(p2reg_dist))  
                    