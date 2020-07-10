#ifndef CAMERAOPEN_IMAGEUTILS_H
#define CAMERAOPEN_IMAGEUTILS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include <map>

using namespace cv;
namespace ryoo {

//    typedef struct IndexNode {
//        int index;
//        std::vector<IndexNode *> children;
//        IndexNode *parent;
//    } ;
//
//    class IndexTree {
//    private:
//        IndexNode *root;
//    public:
//        void init(IndexNode *root);
//
//        void putChild(IndexNode *node, IndexNode *parent);
//
//        void putChildren(std::vector<IndexNode *> nodes, IndexNode *parent);
//
//    };

    struct RelatedPoints {
        std::vector<Point2f> points;
        std::vector<Point2f> pointsRelate;
        std::vector<int> index;
    };

    class Line {
    public:
        Line(Point2f p1, Point2f p2);

        Line(Line &L);

        ~Line() = default;

        double getLength() const { return _length; }

        Eigen::Vector3d getGeneralEquation() { return _gen_equ; };
    private:
        Point2f _p1 = {0, 0}, _p2 = {1, 1};
        double _length;
        Eigen::RowVector3d _gen_equ = {1.0, -1.0, 0.0};
    };

    Point2f get3LinesIntersection(Line l1, Line l2, Line l3);

    RelatedPoints find2PointsFurthest(std::vector<Point2f> points);

    float get3PointsArea(const Point2f &p1, const Point2f &p2, const Point2f &p3);

    void sortPointsForPerspective(std::vector<Point2f> &corners);

    void sortPointsForHomograph(std::vector<std::vector<Point>> &corners);

    std::vector<Point2f> find4PointsForPerspective(std::vector<std::vector<Point2f>> &corners);

    void sharpen2D(const Mat &image, Mat &result);
}


#endif //CAMERAOPEN_IMAGEUTILS_H
