#ifndef ORB_H
#define ORB_H

#include <stdlib.h>   
#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/hal/interface.h>

using namespace std;

struct MOMENTS{
    float m10;
    float m01;
    float m00;
};

float computeMoment(cv::Mat &image, int x, int y, int p, int q);
void computeMoments(MOMENTS &moments, cv::Mat &image, cv::KeyPoint &keyPoint);
vector<float> computeCentroid(MOMENTS &moments);
float computeTheta(MOMENTS &moments);
cv::Point2i rotate(int x, int y, float theta, cv::KeyPoint &keyPoint);
vector<int> getKeyDescriptor(cv::Mat &image, cv::KeyPoint &keyPoint, float theta);
vector<vector<int>> computeDescriptor(cv::Mat &image, vector<cv::KeyPoint> &keyPoints);
bool compareDistance (cv::DMatch a, cv::DMatch b);
void sortMatches(vector<cv::DMatch> matches);
vector<cv::DMatch> filterMatches(vector<cv::DMatch> &matches);
vector<cv::DMatch> match(vector<vector<int>> &firstDescriptor, vector<vector<int>> &secondeDescriptor);

#endif