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

def _center_and_normalize_points(points):
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points

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
        
        A = dst.T @ src

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
    
            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  0, 0,p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  0, 0, p2.item(0) * p1.item(2)]
            aList.append(a1)
            aList.append(a2)
    
        matrixA = np.matrix(aList)
    
        #svd composition
        u, s, v = np.linalg.svd(matrixA)
    
        #reshape the min singular value into a 3 by 3 matrix
        # h = np.reshape(v[8], (3, 3)) # getting the least important eigenvalue vector
        h = np.reshape(v[6], (3, 3)) # getting the least important eigenvalue vector

        #normalize and now we have h
        h = (1/h.item(8)) * h
        return h

    elif (model == 'Projection'):
        for corr in matches:
            
            p1 = np.matrix([corr.item(0), corr.item(1), 1]) # first point of the correspondence
            p2 = np.matrix([corr.item(2), corr.item(3), 1]) # second point of the correspondence
    
            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            aList.append(a1)
            aList.append(a2)
    
        matrixA = np.matrix(aList)
    
        #svd composition
        u, s, v = np.linalg.svd(matrixA)
    
        #reshape the min singular value into a 3 by 3 matrix
        h = np.reshape(v[8], (3, 3)) # getting the least important eigenvalue vector
        # 8 because it's the last in the v vector, the least imporant
    
        #normalize and now we have h
        h = (1/h.item(8)) * h
        return h
    
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)
    
def computeHomographyRANSAC(corr, thresh, model):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = computeHomography(randomFour, model)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers

def keyPointMask(img,kp):
    KP=[] # cv2 keypoint format
    for y,x in kp:
        KP.append(cv2.KeyPoint(y,x,100))
    return KP    

def reProjErr(corrs, h):
    p1reg = cv2.warpPerspective(corrs[:,0:2],h,corrs[:,0:2].shape)
    hinv = np.linalg.inv(finalH)/np.linalg.inv(h)[2,2]
    p2reg = cv2.warpPerspective(corrs[:,2:4],hinv,corrs[:,2:4].shape)
    
    p1reg_dist = np.linalg.norm(corrs[:,2:4].T-p1reg)
    p2reg_dist = np.linalg.norm(corrs[:,0:2].T-p2reg)
    
    return p1reg_dist, p2reg_dist
        
estimation_thresh = 0.6

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
            #train image
            img2 = readImage(img2name)
            rows,cols = img1.shape
            
            
            # Step 2
            correspondenceList = []
            for mode in ['Euclidean', 'Similarity', 'Affine', 'Projection']:
                correspondenceList = []
                if img1 is not None and img2 is not None:
                    
                    KP1, desc1 = findFeatures(img1)
                    KP2, desc2 = findFeatures(img2)
                    
                    print ("Found keypoints in " + img1name + ": " + str(len(KP1)))
                    print ("Found keypoints in " + img2name + ": " + str(len(KP2)))
                    keypoints = [KP1,KP2]
                    matches = matchFeatures(KP1, KP2, desc1, desc2, img1, img2)
                    
                    for match in matches:
                        (x1, y1) = keypoints[0][match.queryIdx].pt
                        (x2, y2) = keypoints[1][match.trainIdx].pt
                        correspondenceList.append([x1, y1, x2, y2])
                
                    corrs = np.matrix(correspondenceList)
                
                    # #run computeHomography algorithm
                    finalH, inliers = computeHomographyRANSAC(corrs, estimation_thresh, mode)
                    print ("Final homography: ", finalH)
                    
                    # Reprojection Error
                    invH = np.linalg.inv(finalH)/np.linalg.inv(finalH)[2,2]
                    R = []
                    for corr in correspondenceList:
                        # print (corr) # y1-corr[0] x1-corr[1] y2-corr[2] x2-corr[3]
                        a = corr[1] - np.divide(finalH[0,0]*corr[3]+finalH[0,1]*corr[2]+finalH[0,2],
                                                finalH[2,0]*corr[3]+finalH[2,1]*corr[2]+1)
                        b = corr[0] - np.divide(finalH[1,0]*corr[3]+finalH[1,1]*corr[2]+finalH[1,2],
                                                finalH[2,0]*corr[3]+finalH[2,1]*corr[2]+1)
                        c = corr[3] - np.divide(invH[0,0]*corr[1]+invH[0,1]*corr[0]+invH[0,2],
                                                invH[2,0]*corr[1]+invH[2,1]*corr[0]+1)
                        d = corr[2] - np.divide(invH[1,0]*corr[1]+invH[1,1]*corr[0]+invH[1,2],
                                                invH[2,0]*corr[1]+invH[2,1]*corr[0]+1)
                        R.append(np.array([a,b,c,d]))
                        
                    R_arr = np.array(R)
                    R_arr = R_arr.reshape([R_arr.shape[0]*R_arr.shape[1],1])
                    Ere = np.inner(np.array(R),np.array(R))/len(R)
                    Ere = np.dot(R_arr.T,R_arr)/len(R)
                    print ("Final inliers count: ", len(inliers))
                
                    matchImg = drawMatches(img1,KP1,img2,KP2,matches,inliers)
                    cv2.imwrite('./Results/RANSAC/{}/InlierMatches_{}_to_{}_{}.png'.format(mode,im1,im2,mode), matchImg)
                
                    f = open('./Results/RANSAC/{}/homography_{}_to_{}_{}.txt'.format(mode,im1,im2,mode), 'w')
                    f.write("Final homography: \n" + str(finalH)+"\n")
                    f.write("Final inliers count: " + str(len(inliers)))
                    f.close()
                
                
                    # Actually registering the image
                    dst = cv2.warpPerspective(img1,finalH,(cols,rows))

                    
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