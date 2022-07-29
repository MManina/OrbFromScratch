#include "Orb.h"
#include <opencv4/opencv2/imgcodecs.hpp>

using namespace std;

#define threshold 20

int main(int argc, char **argv) {

    if(argc < 2){
        cout << "Error : This program needs 2 images in arguments" << endl;
        return 0;
    }

    // Read 2 images
    cv::Mat firstImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat secondImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    // Get feature points of the first image
    vector<cv::KeyPoint> firstKeyPoints;
    cv::FAST(firstImage, firstKeyPoints, threshold);

    // Get feature points of the second image
    vector<cv::KeyPoint> secondKeyPoints;
    cv::FAST(secondImage, secondKeyPoints, threshold);

    // Get descriptor for the 2 images
    vector<vector<int>> firstDescriptor = computeDescriptor(firstImage, firstKeyPoints);
    vector<vector<int>> secondDescriptor = computeDescriptor(secondImage, secondKeyPoints);

    // Match feature point of the 2 images
    vector<cv::DMatch> matchs = match(firstDescriptor, secondDescriptor);

    // Show Match
    cv::Mat imgMatch;
    cv::drawMatches(firstImage, firstKeyPoints, secondImage, secondKeyPoints, matchs, imgMatch);
    cv::imshow("Display image", imgMatch);
    cv::waitKey(0);

    return 0;
}