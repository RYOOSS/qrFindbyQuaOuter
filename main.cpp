#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
//#include <zbar.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/LuminanceSource.h>
#include <zxing/MatSource.h>
#include <zxing/DecodeHints.h>
#include <zxing/Binarizer.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <imageUtils.h>

using namespace cv;
using namespace ryoo;
using namespace std;


Eigen::MatrixXf cameraIntrinsics(3, 3);

vector<Point2f> findContourCenter(std::vector<std::vector<Point>> contours) {
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); ++i) {
        mu[i] = moments(contours[i], false);
    }
    vector<Point2f> mc(contours.size());
    for (int j = 0; j < contours.size(); ++j) {
        mc[j] = Point2f(mu[j].m10 / mu[j].m00, mu[j].m01 / mu[j].m00);
    }
    return mc;
}

Point2f findContourCenter(std::vector<std::vector<Point>> contours, int index) {
    Moments mu = moments(contours[index], false);
    Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
    return mc;
}

std::string deCode(const Mat &qrCode) {
    std::string content;
    try {
        Mat qrCodeGray, qrCodeSharpen;
        cvtColor(qrCode, qrCodeGray, COLOR_BGR2GRAY);
        sharpen2D(qrCodeGray, qrCodeSharpen);
        imwrite("QRRQ.jpg", qrCodeSharpen);
        zxing::MultiFormatReader formatReader;
        zxing::Ref<zxing::LuminanceSource> source = MatSource::create(const_cast<Mat &>(qrCodeSharpen));
        zxing::Ref<zxing::Reader> reader;
        reader.reset(new zxing::qrcode::QRCodeReader);
        zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));
        zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));
        zxing::Ref<zxing::Result> result(reader->decode(bitmap, zxing::DecodeHints::QR_CODE_HINT));

        content = result->getText()->getText();
        std::cout << "\033[32m" << content << "\033[0m" << std::endl;
    }
    catch (zxing::Exception e) {

    }
    return content;
}

Mat preProcessImg(const Mat &img) {
//    Mat img1;
//    medianBlur(img, img1, 3);
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY); // 转灰度图
    Mat imgBinary;
    double thresh = 200;
//    adaptiveThreshold(imgGray,imgBinary,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,21,10);
    threshold(imgGray, imgBinary, thresh, 255, THRESH_OTSU | THRESH_BINARY); // 转二值图
    return imgBinary;
}

vector<Vec4i> filterBlock(const Mat &img, vector<vector<Point>> contours, vector<Vec4i> hierarchy) {
    vector<Vec4i> blockConfid; // 信任特征块序号
//    vector<Point2f> mc = findContourCenter(contours);
    for (int i = 0; i < contours.size(); ++i) {
        const double minMidArea = 10; //中间块最小面积
        const double maxMidArea = pow(min(img.rows, img.cols), 2) * 25 / 441;//中间块最大面积
        int firstChild = hierarchy[i][2]; //子块
        int parent = hierarchy[i][3]; //父块
        if (firstChild < 0 || parent < 0) continue; //筛选掉无父子的轮廓
        int uncle0 = hierarchy[parent][0];
        int uncle1 = hierarchy[parent][1];
        int grand = hierarchy[parent][3];
        if ((uncle0 < 0 && uncle1 < 0) || grand < 0)continue; //筛选掉独立轮廓
        double area = contourArea(contours[i]);
        if (area < minMidArea || maxMidArea < area) continue; //面积筛选掉面积过大过小轮廓
        double ratio1 = contourArea(contours[parent]) / area,
                ratio2 = area / contourArea(contours[firstChild]);
        vector<float> centroidVec{findContourCenter(contours, i).x, findContourCenter(contours, i).y, 1.0,
                                  findContourCenter(contours, parent).x, findContourCenter(contours, parent).y, 1.0,
                                  findContourCenter(contours, firstChild).x, findContourCenter(contours, firstChild).y,
                                  1.0};
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> centroid(centroidVec.data(),
                                                                                                   3, 3);
        if (34.0 / 25.0 <= ratio1 && ratio1 <= 64.0 / 25.0 //面积比例 外/中 = 49/25
            && 16.0 / 9.0 <= ratio2 && ratio2 <= 34.0 / 9.0 // 面积比例 中/内 = 25/9
            && abs(centroid.determinant()) <= 1) {//三中心三角形面积 < 1
            blockConfid.emplace_back(firstChild, i, parent, grand);
        }
    }//寻找特征块
//    cout << "fliterBlock" << endl;
    return blockConfid;
}

vector<vector<int>> classifyMultiQr(vector<Vec4i> blockConfid) {
    vector<vector<int>> outLabel;
    int frameLabelTemp = -1;
    for (auto &i : blockConfid) { // 挑选共同外框
        int frameLabelNow = i[3];
        if (frameLabelNow != frameLabelTemp) {
            outLabel.push_back({frameLabelNow});
            frameLabelTemp = frameLabelNow;
        }
    }

    for (auto &i:blockConfid) {
        for (auto &j : outLabel) {
            if (i[3] == j[0]) {
                j.push_back(i[2]);
            }
        }
    }
    cout << "classifyMultiQr" << endl;
    return outLabel;
}

bool checkQRLabelEmpty(vector<vector<int>> &outLabel) {
    for (int i = 0; i < outLabel.size(); ++i) {
        if (outLabel[i].size() < 4) {
            outLabel.erase(outLabel.begin() + i);
            i--;
        }
    }
    cout << "outLabel " << outLabel.empty() << " empty" << endl;
    return outLabel.empty();
}

vector<vector<Point2f>>
get3BlockCornerSubPixel(const Mat &img, const vector<vector<Point>> &contours, vector<int> inLabel) {
    assert(inLabel.size() == 4);
    vector<vector<Point2f>> qrCorners;
    for (int i = 1; i < 4; ++i) {
        vector<Point2f> qrCorner;
        Mat codeContour = Mat::zeros(img.rows, img.cols, CV_8UC1);
        drawContours(codeContour, contours, inLabel[i], 255, 1, 8);
        TermCriteria criteria(1, 40, 0.001);
        goodFeaturesToTrack(codeContour, qrCorner, 4, 0.01, 10, Mat(), 3, false, 0.04);
        cornerSubPix(codeContour, qrCorner, Size(3, 3), Size(-1, -1), criteria);
        qrCorners.push_back(qrCorner);
        cout << "Block" << to_string(i) << " Corner Pixel is " << endl << qrCorner << endl;
    }
    return qrCorners;
}

bool checkQRCorners(const vector<vector<Point2f>> &qrCorners) {
    if (qrCorners.size() != 3) { return true; }
    for (const auto &qrCorner:qrCorners) {
        if (qrCorner.size() != 4) { return true; }
    }
    return false;
}

vector<Point2f> getOuterCornerSubPixel(const Mat &img, const vector<vector<Point>> &contours, int outerLabel) {
    vector<Point2f> outerCorner;
    Mat codeContour = Mat::zeros(img.rows, img.cols, CV_8UC1);
    drawContours(codeContour, contours, outerLabel, 255, 1, 8);
    TermCriteria criteria(1, 40, 0.001);
    goodFeaturesToTrack(codeContour, outerCorner, 4, 0.01, 10, Mat(), 3, false, 0.04);
    cornerSubPix(codeContour, outerCorner, Size(3, 3), Size(-1, -1), criteria);
    sortPointsForPerspective(outerCorner);

    return outerCorner;

}

bool checkOuterCornersEmpty(const vector<Point2f> &outerCorner) {
    if (outerCorner.size() != 4) { return true; }
    return false;
}

vector<vector<Point>> cutOuterContours4Part(vector<Point> contour) {
    vector<Point> approx;
    approxPolyDP(Mat(contour), approx, 10, true);
    vector<vector<Point>> contour4Parts(4);
    if (approx.size() == 4) {
        contour.insert(contour.end(), contour.begin(), contour.end());
//        sortPointsForPerspective(contour);
        int i = 0;
        int _index = -1, _begin = -1, _end = -1;
        while (true) {
            for (int j = 0; j < 4; ++j) {
                if (contour[i] == approx[j]) {
                    _index = j;
                    if (_begin == -1) {
                        _begin = i;
                        _end = _begin + contour.size() / 2;
                    }
                }
            }
            if (_index > -1) {
                contour4Parts[_index].push_back(contour[i]);
            }
            if (i == _end) { break; }
            i++;
        }
        sortPointsForHomograph(contour4Parts);
    }
    std::cout << "cut" << std::endl;
    return contour4Parts;
}

bool checkContourPartsEmpty(vector<vector<Point>> contour4Parts) {
    if (contour4Parts.size() != 4) { return true; }
    for (auto contour4Part:contour4Parts) {
        if (contour4Part.empty()) { return true; }
    }
    return false;
}

Mat getHomographTransform(vector<vector<Point>> contour4Parts) {
    vector<vector<Point2f>> qrFrames(4);
    float increment;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < contour4Parts[i].size(); ++j) {
            increment = 229.0 / contour4Parts[i].size() * j;
            switch (i) {
                case 0:
                    qrFrames[0].push_back(Point2f(-10.0, -10 + increment));
                    break;
                case 1:
                    qrFrames[1].push_back(Point2f(-10.0 + increment, 219.0));
                    break;
                case 2:
                    qrFrames[2].push_back(Point2f(219.0, 219.0 - increment));
                    break;
                case 3:
                    qrFrames[3].push_back(Point2f(219.0 - increment, -10));
                    break;
                default:
                    break;
            }
        }
    }
//
    vector<Point2f> qrFrame, contour4Part;
    for (int i = 0; i < 4; ++i) {
        qrFrame.insert(qrFrame.end(), qrFrames[i].begin(), qrFrames[i].end());
        contour4Part.insert(contour4Part.end(), contour4Parts[i].begin(), contour4Parts[i].end());
    }

    Mat T;
    T = findHomography(contour4Part, qrFrame, RANSAC);
    std::cout << T << std::endl;
    return T;
}


//void presentQR(Mat img, vector<Mat> qrCodes, vector<vector<int>> outLabel) {}

Mat imgFindQr(const Mat &imgBinary, Mat img) {

    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(imgBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE); // 轮廓
    vector<Vec4i> blockConfid = filterBlock(img, contours, hierarchy); // 信任特征块序号
    if (blockConfid.empty()) { return img; } //确保信任特征块非空cZ
    vector<vector<int>> outLabel = classifyMultiQr(blockConfid); //多特征块归类多二维码中
//    for (int i = 0; i < outLabel.size(); ++i) {
//
//        if (contourArea(contours[outLabel[i][0]]) / 30 < contourArea(contours[outLabel[i][2]]) <
//            contourArea(contours[outLabel[i][0]]) / 20) {
//            drawContours(img, contours, outLabel[i][0], Scalar(0, 255, 0), 1, 8);
//        }
//    }
    if (checkQRLabelEmpty(outLabel)) { return img; }; //确保特征块非空

    vector<Mat> qrCodes;
    vector<string> decoded_infos;
    vector<Point2f> qrFrame(4);

    qrFrame[0] = Point2f(-1, -1);
    qrFrame[1] = Point2f(23, -1);
    qrFrame[2] = Point2f(-1, 23);
    qrFrame[3] = Point2f(23, 23);
    for (int i = 0; i < outLabel.size(); ++i) {

        Mat qrCode;

        vector<Point2f> outerCorner = getOuterCornerSubPixel(img, contours, outLabel[i][0]);
        if (checkOuterCornersEmpty(outerCorner)) { continue; }
        std::cout << "\033[31mouter Corner Pixel is\033[0m" << std::endl << outerCorner << std::endl;

        Mat perTran = getPerspectiveTransform(outerCorner, qrFrame);


        warpPerspective(img, qrCode, perTran, Size(23, 23));

        qrCodes.push_back(qrCode);


        decoded_infos.push_back(deCode(qrCodes[i]));
//        if (decoded_infos[i] == "093100JIYINxxx") {
        if (decoded_infos[i] == "087100JIYINxxx") {
            Eigen::MatrixXf qrSquareInSpace = getBack2Space(outerCorner, cameraIntrinsics);
            std::cout << "\033[31mQRcode in the Space\033[0m" << std::endl << qrSquareInSpace << std::endl;
            Eigen::MatrixXf qrPose(4, 4), skewX(3, 3);
            Eigen::VectorXf qrPoseX(3), qrPoseY(3), qrPoseZ(3), qrPoseP(3);
            qrPoseX = (qrSquareInSpace.row(1) - qrSquareInSpace.row(0)).normalized().transpose();
            qrPoseY = (qrSquareInSpace.row(2) - qrSquareInSpace.row(1)).normalized().transpose();
            skewX << 0, -qrPoseX(2), qrPoseX(1), qrPoseX(2), 0, -qrPoseX(0), -qrPoseX(1), qrPoseX(0), 0;
            qrPoseZ = skewX * qrPoseY;
            qrPoseP << qrSquareInSpace.col(0).mean(), qrSquareInSpace.col(1).mean(), qrSquareInSpace.col(2).mean();
            qrPose << qrPoseX, qrPoseY, qrPoseZ, qrPoseP, 0, 0, 0, 1;
            std::cout << "\033[31mTransform between end and qr is \033[0m" << std::endl << qrPose << std::endl;
        }

    }

    for (int k = 0; k < decoded_infos.size(); ++k) {
        string qrframe = "qrcode";
        qrframe.append(to_string(k + 1));
        imshow(qrframe, qrCodes[k]);
        qrCodes.push_back(qrCodes[k]);
        drawContours(img, contours, outLabel[k][0], Scalar(0, 0, 255), 1, 8);
        putText(img, decoded_infos[k], contours[outLabel[k][0]][0], FONT_HERSHEY_PLAIN, 2.0,
                Scalar(0, 0, 255), 2);
    }

    imwrite("code.jpg", img);
    return img;

}

Mat read1Image() {
    Mat img, res, imgBinary;
    img = imread("/home/ryoo/CLionProjects/QRcode/qrFindbyQuaOuter/qrcode.jpg");
//    resize(img, img, Size(img.cols / 4, img.rows / 4), 0, 0, INTER_LINEAR);
    return img;
}

int main(int argc, char **argv) {

    cameraIntrinsics << 1.118687327324564e+03, 0.0, 703.25397, 0, 1.135263811351046e+03, 482.50630, 0.0, 0.0, 1.0;
    Mat image, res, imgBinary;
    image = read1Image();
//      vector<  string> decoded_info;
//      vector<Point> points;
//      vector<Mat> straight_qrcode;
//    QRCodeDetector qrdetector =QRCodeDetector();
//    qrdetector.detectAndDecodeMulti(img,decoded_info,points,straight_qrcode);
//      string name ="demo";
//
//    for (int i = 0; i < straight_qrcode.size(); ++i) {
//        name.append(  to_string(i+1));
//        imshow(name, straight_qrcode[i]);
//
//    }
//    waitKey();
    imgBinary = preProcessImg(image);
    res = imgFindQr(imgBinary, image);
    imshow(" ", imgBinary);
    imshow("demo", image);
    waitKey();
    return 0;


//
//    VideoCapture cap;
//    cap.open(0);
//
//    if (!cap.isOpened())
//        return -1;
//    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(CAP_PROP_FRAME_HEIGHT, 800);
//    cout << "The Intrinsics of camera is " << endl << cameraIntrinsics << endl;
//    while (true) {
//        cap >> image;//等价于cap.read(frame)
//        imgBinary = preProcessImg(image);
//        res = imgFindQr(imgBinary, image);
//        imshow(" ", imgBinary);
//        imshow("test", res);
//        char c = waitKey(66);
//        if (image.empty())
//            break;
//        if (c == 27)
//            break;
//
//    }
//    cap.release();
//    destroyAllWindows();
//    return 0;
}