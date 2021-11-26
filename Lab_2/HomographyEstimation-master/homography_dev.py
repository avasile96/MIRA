# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:24:27 2021

@author: vasil

CODE FOR PART 2
"""

import cv2
import numpy as np
import getopt
import sys
import random
import scipy.io
from skimage.morphology import binary_dilation
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt
from skimage import io
#
# Read in an image file, errors out if we can't find the file
#
def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img



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

def findFeaturesWithKp(img, canvas, KP):
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


#
# Computers a homography from 4-correspondences
#
def computeHomography(matches, model):
    #loop through correspondences and create assemble matrix
    aList = []
    if (model == 'Euclidean'):
        src = np.array(matches)[:,0:2]
        dst = np.array(matches)[:,2:4]
        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        A = dst_demean.T @ src_demean / num

        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        h = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        rank = np.linalg.matrix_rank(A)
        
        # Computing the Rotation Matrix
        if rank == 0:
            return np.nan * h
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                h[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                h[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            h[:dim, :dim] = U @ np.diag(d) @ V
        
        # Getting the Transaltion Terms
        h[:dim, dim] = dst_mean - (h[:dim, :dim] @ src_mean.T)
        return h
    
    elif (model == 'Similarity'):
        src = np.array(matches)[:,0:2]
        dst = np.array(matches)[:,2:4]
        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        A = dst_demean.T @ src_demean / num

        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        h = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)
        
        rank = np.linalg.matrix_rank(A)
        # Computing the Rotation Matrix
        if rank == 0:
            return np.nan * h
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                h[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                h[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            h[:dim, :dim] = U @ np.diag(d) @ V

        # Getting the Scale
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        
        # Getting the Transaltion Terms
        h[:dim, dim] = dst_mean - scale * (h[:dim, :dim] @ src_mean.T)
        h[:dim, :dim] *= scale
        return h
    
    elif (model == 'Affine'):
        for corr in matches:
            p1 = np.matrix([corr.item(0), corr.item(1), 1]) # first point of the correspondence
            p2 = np.matrix([corr.item(2), corr.item(3), 1]) # second point of the correspondence
    
            a1 = [p1.item(0), p1.item(1), 1, 0, 0, 0]
            a2 = [0, 0, 0, p1.item(0), p1.item(1), 1]
            
            aList.append(a1)
            aList.append(a2)
    
        matrixA = np.matrix(aList)
        U, S, V = np.linalg.svd(matrixA)
        h = np.zeros([3,3])
    
        #reshape the min singular value into a 3 by 3 matrix
        h[0:2] = np.reshape(V[5], (2, 3)) # getting the least important eigenvalue vector
        h[-1,-1] = 1

        return h
    elif (model == 'Projection'):
        for corr in matches:
            p1 = np.matrix([corr.item(0), corr.item(1), 1]) # first point of the correspondence
            p2 = np.matrix([corr.item(2), corr.item(3), 1]) # second point of the correspondence
            
            a1 = [p1.item(0), p1.item(1), 1, 0, 0, 0, -p2.item(0) * p1.item(0), 
                  -p1.item(0) * p2.item(0)]
            a2 = [0, 0, 0, p1.item(0), p1.item(1), 1, -p1.item(0) * p2.item(1), 
                  -p1.item(1) * p2.item(1)]

            aList.append(a1)
            aList.append(a2)
    
        matrixA = np.matrix(aList)
    
        #svd composition
        # u, s, v = np.linalg.svd(matrixA)
        B = matches
        b = np.reshape(B,[-1,1])
        x = matrixA/b
        h = np.zeros([1,9])
        
        h[-1,-1]=1
        h[0,0:8] = v[7]
        h = h.reshape([3,3])
    
        #normalize and now we have h
        h = (1/h.item(8)) * h
        return h

def keyPointMask(img,kp):
    canvas = np.zeros_like(img, dtype = np.uint8)
    KP=[] # cv2 keypoint format
    for y,x in kp:
        canvas[int(x),int(y)] = 1
        KP.append(cv2.KeyPoint(y,x,100))
    return np.array(binary_dilation(canvas, selem = np.ones([4,4])),dtype = np.uint8), KP        
        

for im1 in ['00']:
    for im2 in ['01','02','03']:
        if im1 != im2:
            img1name = "./DataSet01/{}.png".format(im1)
            img2name = "./DataSet01/{}.png" .format(im2)
            
            print("Image 1 Name: " + img1name)
            print("Image 2 Name: " + img2name)
            
            #query image
            img1 = readImage(img1name)
            #train image
            img2 = readImage(img2name)
            rows,cols = img1.shape
            
            # Loading features
            mat = scipy.io.loadmat('./DataSet01/Features.mat')
            Features = mat['Features'].T
            features = []
            
            for i in range (0,Features.shape[0]):
                features.append(Features[i][0][0])
            
            # del mat, Features
            kp1 = np.floor(features[int(img1name[-5])]) # x (kp[:,1]) and y (kp[:,0]) 
            kp2 = np.floor(features[int(img2name[-5])])
            
            
            # Step 2
            for mode in ['Euclidean', 'Similarity', 'Affine', 'Projection']:
                correspondenceList = []
                if img1 is not None and img2 is not None:
                    
                    canvas1, KP1 = keyPointMask(img1, kp1) # KP1 is kp1 in cv2 keypoint format
                    canvas2, KP2 = keyPointMask(img2, kp2)
                    
                    KP1, desc1 = findFeaturesWithKp(img1,canvas1,KP1)
                    KP2, desc2 = findFeaturesWithKp(img2,canvas2,KP2)
                    
                    print ("Found keypoints in " + img1name + ": " + str(len(KP1)))
                    print ("Found keypoints in " + img2name + ": " + str(len(KP2)))
                    keypoints = [KP1,KP2]
                    matches = matchFeatures(KP1, KP2, desc1, desc2, img1, img2)
                    
                    for match in matches:
                        (x1, y1) = keypoints[0][match.queryIdx].pt
                        (x2, y2) = keypoints[1][match.trainIdx].pt
                        correspondenceList.append([y1, x1, y2, x2])
                
                    corrs = np.matrix(correspondenceList)
                
                    # #run computeHomography algorithm
                    # mode = 'Euclidean'
                    finalH = computeHomography(corrs, mode)
                
                    print ("Final homography: ", finalH)
                    
                    # Reprojection Error
                    invH = np.linalg.inv(finalH)/np.linalg.inv(finalH)[2,2]
                    R = []
                    for corr in correspondenceList:
                        # print (corr) # y1-corr[0] x1-corr[1] y2-corr[2] x2-corr[3]
                        a = corr [0] - np.divide (finalH[0,0]*corr[2] + finalH[0,1]*corr[3] +finalH[0,2], 
                                                  finalH[2,0]*corr[2] + finalH[2,1]*corr[3] +1)
                        b = corr[1] - np.divide(finalH[1,0]*corr[2]+finalH[1,1]*corr[3]+finalH[1,2],
                                                finalH[2,0]*corr[3]+finalH[2,1]*corr[2]+1)
                        c = corr[2] - np.divide(invH[0,0]*corr[0]+invH[0,1]*corr[1]+invH[0,2],
                                                invH[2,0]*corr[0]+invH[2,1]*corr[1]+1)
                        d = corr[3] - np.divide(invH[1,0]*corr[0]+invH[1,1]*corr[1]+invH[1,2],
                                                invH[2,0]*corr[0]+invH[2,1]*corr[1]+1)
                        R.append(np.array([a,b,c,d]))
                        
                    R_arr = np.array(R)
                    R_arr = R_arr.reshape([R_arr.shape[0]*R_arr.shape[1],1])
                    Ere = np.inner(np.array(R),np.array(R))/len(R)
                    # Ere = np.dot(R_arr.T,R_arr)/len(R)
                    print ("Error: ", Ere)
                    
                    # Actually registering the image
                    dst = cv2.warpPerspective(img1,finalH,(cols,rows))
                    
                    # plt.figure()
                    # io.imshow(img1)
                    # plt.figure()
                    # io.imshow(img2)
                    # plt.figure()
                    # io.imshow(dst)
                    
                    io.imsave('./Results/{}/{}_to_{}_{}.png'.format(mode,im1,im2,mode), dst)
                    
                    f = open('./Results/{}/homography_{}_to_{}_{}.txt'.format(mode,im1,im2,mode), 'w')
                    f.write("Final homography: \n" + str(finalH)+"\n")
                    
                    added_image = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
                    added_image[:,:,1] = dst
                    added_image[:,:,2] = dst
                    # added_image = cv2.addWeighted(img2,0.4,dst,0.1,0)
                    io.imsave('./Results/{}/overlay_{}_to_{}_{}.png'.format(mode,im1,im2,mode), added_image)
