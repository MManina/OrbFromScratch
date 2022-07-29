#include "Orb.h"

using namespace std;

/*
Calculate a moment at one given point
@param image: reference of the matrix of an image in wich you want to caclulate a moment
@param x: coordinate x of the point in the image 
@param y: coordinate y of the point in the image 
@param p: 1 if you want to calculate for x else 0
@param q: 1 if you want to calculate for y else 0
@return : the moment calculated with the given parameters
*/
float computeMoment(cv::Mat &image, int x, int y, int p, int q){
    return pow(x, p) * pow(y, q) * image.at<uchar>(y, x);
}

/*
Calculate a moment at a given key point
@param moments: reference of a moment to be modified
@param image: reference of the matrix of an image in wich you want to caclulate a moment
@param keyPoint: reference of the key point
*/
void computeMoments(MOMENTS &moments, cv::Mat &image, cv::KeyPoint &keyPoint){
    moments.m10 = 0;
    moments.m01 = 0;
    moments.m00 = 0;
    int rayon = keyPoint.size/2;    // rayon of the meaningful keypoint neighborhood
    for(int x=-rayon; x<=rayon; x++){
        for(int y=-rayon; y<=rayon; y++){
            moments.m10 += computeMoment(image, keyPoint.pt.x+x, keyPoint.pt.y+y, 1, 0);
            moments.m01 += computeMoment(image, keyPoint.pt.x+x, keyPoint.pt.y+y, 0, 1);
            moments.m00 += computeMoment(image, keyPoint.pt.x+x, keyPoint.pt.y+y, 0, 0);
        }
    }
}

/*
Calculate the centroid of a image block
@param moments: reference of the moment
@return : the calculated centroid
*/
vector<float> computeCentroid(MOMENTS &moments){
    vector<float> centroid{moments.m10/moments.m00, moments.m01/moments.m00};
    return centroid;
}

/*
Calculate the orientation of a image block
@param moments: reference of the moment
@return : the calculated angle
*/
float computeTheta(MOMENTS &moments){
    return atan2(moments.m01, moments.m10);
}

/*
Calculate the rotation of a point
@param x: coordinate x of the point to rotate with the keyPoint as the origin
@param y: coordinate y of the point to rotate with the keyPoint as the origin
@param theta: angle of rotation
@param keyPoint: reference of a keyPoint
@return : a point with the new position
*/
cv::Point2i rotate(int x, int y, float theta, cv::KeyPoint &keyPoint){
    cv::Point2i p;
    p.x = x * cos(theta) + y * sin(theta) + keyPoint.pt.x;
    p.y = -x * sin(theta) + y * cos(theta) + keyPoint.pt.y;
    return p;
}

/*
Calculate the description of a given key point
@para image: reference of the matrix of an image
@param keyPoint: reference of the key point
@param theta: orientation of the image block
@return : description vector of the key point
*/
vector<int> getKeyDescriptor(cv::Mat &image, cv::KeyPoint &keyPoint, float theta){
    vector<int> keyDesc;
    int rayon = keyPoint.size/2;    // rayon of the meaningful keypoint neighborhood
    cv::Point2i p;  // Point p to compare
    cv::Point2i q;  // Point q to compare
    // Compare each symmetrical pair of points
    for(int x=0; x<=rayon; x++){
        for(int y=-rayon; y<=rayon; y++){
            // if x=0 compare symmetrical up and bottom pair of points
            if(x==0){
                // Calculate rotation for the point p
                p = rotate(x, y, theta, keyPoint);
                // Calculate rotation for the point q
                q = rotate(x, -y, theta, keyPoint);
            // else compare symmetrical right and left pair of points
            }else{
                // Calculate rotation for the point p
                p = rotate(x, y, theta, keyPoint);
                // Calculate rotation for the point q
                q = rotate(-x, y, theta, keyPoint);
            }
            // if p > q add 1 to descriptor else 0
            if(image.at<uchar>(p.y, p.x) > image.at<uchar>(q.y, q.x)){
                keyDesc.push_back(1);
            }else{
                keyDesc.push_back(0);
            }
        }
    }
    return keyDesc;
}

/*
Calculate the description of an given image
@para image: reference of the matrix of an image
@param keyPoint: reference of the key point
@return : description vector of the image
*/
vector<vector<int>> computeDescriptor(cv::Mat &image, vector<cv::KeyPoint> &keyPoints){
    vector<vector<int>> descriptor;
    vector<int> keyDesc;
    MOMENTS moments;

    for(cv::KeyPoint kp : keyPoints){
        
        fill(keyDesc.begin(), keyDesc.end(), 0);

        // Compute the moment of the image block around the key point
        computeMoments(moments, image, kp);
        computeMoments(moments, image, kp);
        computeMoments(moments, image, kp);

        // Compute the orientation of the image block around the key point
        float theta = computeTheta(moments);

        // Compute the descriptor of of the image block around the key point
        keyDesc = getKeyDescriptor(image, kp, theta);

        // Add keyPoint descriptor to the image descriptor
        descriptor.push_back(keyDesc);
    }

    return descriptor;
}

/*
Compare 2 DMatch according to their distance
@para a: first DMatch to compare
@param b: second DMatch to compare
@return : boolan
*/
bool compareDistance (cv::DMatch a, cv::DMatch b){
    return (a.distance<b.distance);
}

/*
Sort vector of DMatch
@para matches: vector of DMatch to sort
*/
void sortMatches(vector<cv::DMatch> matches){
    sort(matches.begin(), matches.end(), compareDistance);
}

/*
Filter vector of DMatch
@para matches: vector of DMatch to filter
*/
vector<cv::DMatch> filterMatches(vector<cv::DMatch> &matches){
    vector<cv::DMatch> fMatch;
    cv::DMatch m;
    sortMatches(matches);
    for(int i=0; i<int(matches.size()*0.7); i++){
        m.queryIdx = matches[i].queryIdx;
        m.trainIdx = matches[i].trainIdx;
        m.distance = matches[i].distance;
        fMatch.push_back(m);
    }
    return fMatch;
}

/*
Find the best match for each feature vector
@param firstDescriptor: reference of the first descriptor of an image
@param secondDescriptor: reference of the second descriptor of an image
@return : match vector between 2 descriptor
*/
vector<cv::DMatch> match(vector<vector<int>> &firstDescriptor, vector<vector<int>> &secondDescriptor){
    vector<cv::DMatch> matches;
    cv::DMatch m;
    int k = firstDescriptor[0].size();  // number of pair of point
    int distance = 0;   // distance between 2 vectors

    for(int i=0; i<firstDescriptor.size() ; i++){

        m.queryIdx = i; // id of the vector in the first descriptor
        m.trainIdx = 0; // id of the vector in the second descriptor
        m.distance = k; // distance max 
        
        for(int j=i-10; j<i+10; j++){
            distance = 0;

            // Count the number of difference
            for(int l=0; l<k; l++){
                // if j out of the vector's size, go inside
                if(j<0){
                    j=0;
                }
                else if(j>secondDescriptor.size()){
                    j=secondDescriptor.size();
                }
                distance += firstDescriptor[i][l] ^ secondDescriptor[j][l];
            }

            // if the number is smaller then save it
            if(distance < m.distance){
                m.trainIdx = j;
                m.distance = distance;
            }
        }

        matches.push_back(m);
    }
    return filterMatches(matches);
}