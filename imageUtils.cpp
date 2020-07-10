#include "imageUtils.h"
#include <utility>


using namespace cv;
namespace ryoo {
//    void IndexTree::init(IndexNode *root) { this->root = root; }
//
//    void IndexTree::putChild(IndexNode *node, IndexNode *parent) {
//        parent->children.push_back(node);
//        node->parent = parent;
//    }
//
//    void IndexTree::putChildren(std::vector<IndexNode *> nodes, IndexNode *parent) {
//        for (auto & node : nodes) {
//            putChild(node, parent);
//        }
//    }




    Line::Line(Point2f p1, Point2f p2) : _p1(std::move(p1)), _p2(std::move(p2)) {
        _length = sqrt(pow(_p1.x - _p2.x, 2) + pow(_p1.y - _p2.y, 2));
        _gen_equ = {(_p2.y - _p1.y), -(p2.x - p1.x), (_p2.x - _p1.x) * _p1.y - (_p2.y - _p1.y) * _p1.x};
    }

    Line::Line(Line &L) : _p1(L._p1), _p2(L._p2) {
        _length = L._length;
        _gen_equ = L._gen_equ;
    }

    RelatedPoints find2PointsFurthest(std::vector<Point2f> points) {
        RelatedPoints furthest2Points;

        if (points.size() == 1) {
            furthest2Points.points = points;
            furthest2Points.pointsRelate = points;
            furthest2Points.index = {0};
            return furthest2Points;
        }
        float temp = -1, dis;
        int _0, _1;
        for (int i = 0; i < points.size(); ++i) {
            for (int j = (int) points.size() - 1; j > i; --j) {
                dis = (float) (pow(points[i].x - points[j].x, 2) + pow(points[i].y - points[j].y, 2));
                if (dis > temp) {
                    temp = dis;
                    if (points[i].x - points[j].x < -0.001) { _0 = i, _1 = j; }
                    else if (abs(points[i].x - points[j].x) <= 0.001) {
                        if (points[i].y - points[j].y < 0) { _0 = i, _1 = j; }
                    } else { _0 = j, _1 = i; }
                }
            }
        }
        furthest2Points.points = points;
        furthest2Points.index.push_back(_0);
        furthest2Points.index.push_back(_1);
        furthest2Points.pointsRelate.push_back(points[_0]);
        furthest2Points.pointsRelate.push_back(points[_1]);
        return furthest2Points;
    }

    float get3PointsArea(const Point2f &p1, const Point2f &p2, const Point2f &p3) {
        float area;
        std::vector<float> matrixVec{p1.x, p1.y, 1.0,
                                     p2.x, p2.y, 1.0,
                                     p3.x, p3.y, 1.0};
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(
                matrixVec.data(), 3, 3);
        area = matrix.determinant() / 2.0;
        return area;
    }

    /*
     * x          x
     *   \        |
     *     x      x
     *       \    |
     *  x--x -----X
     */
    Point2f get3LinesIntersection(Line l1, Line l2, Line l3) {
        Eigen::MatrixXf A(3, 2);
        A
                << l1.getGeneralEquation().x(), l1.getGeneralEquation().y(), l2.getGeneralEquation().x(), l2.getGeneralEquation().y(), l3.getGeneralEquation().x(), l3.getGeneralEquation().y();
        Eigen::MatrixXf b(3, 1);
        b << -l1.getGeneralEquation().z(), -l2.getGeneralEquation().z(), -l3.getGeneralEquation().z();
        Eigen::MatrixXf x(2, 1);
        x = (A.transpose() * A).inverse() * A.transpose() * b;
        Point2f intersection(x(0), x(1));
        return intersection;
    }

    /*
    * p0---p1
    *     /
    *    /
    *   /
    * p2---p3
    * */
    void sortPointsForPerspective(std::vector<Point2f> &corners) {
        for (int i = 0; i < 4; ++i) {
            for (int j = i; j > 0; j--) {
                if (corners[j].x < corners[j - 1].x)
                    swap(corners[j], corners[j - 1]);
                else break;
            }
        }
        if (corners[0].y > corners[1].y) { swap(corners[0], corners[1]); }
        if (corners[2].y > corners[3].y) { swap(corners[2], corners[3]); }
        swap(corners[1], corners[2]);
    }

    /*
    * p0   p3
    * |     |
    * |     |
    * |     |
    * p1---p2
    * */
    void sortPointsForHomograph(std::vector<std::vector<Point>> &approx) {
        for (int i = 0; i < 4; ++i) {
            for (int j = i; j > 0; j--) {
                if (approx[j][0].x < approx[j - 1][0].x)
                    swap(approx[j], approx[j - 1]);
                else break;
            }
        }
        if (approx[0][0].y > approx[1][0].y) { swap(approx[0], approx[1]); }
        if (approx[2][0].y < approx[3][0].y) { swap(approx[2], approx[3]); }
    }

    /*
    * 1m1----1r    2m1----2r
    *   | ↖  |       |    |
    *   |   ↖|       |    |
    *  1l----1m2    2l----2m2
    *
    * 0m1----0r     ↖
    *   |    |        ↖
    *   |    |          ↖
    *  0l----0m2          . 3
    * */
    std::vector<Point2f> find4PointsForPerspective(std::vector<std::vector<Point2f>> &cornersVec) {


        float tempMin = -1, tempMax = INFINITY, area;
        int _0l, _1, _2r, _1m1, _1m2;
        std::vector<Point2f> points;
        Point2f p0, p1, p2, p3;
        for (int i = 0; i < 3; ++i) {
//            sortPointsForPerspective(cornersVec[i]);
            for (int j = 0; j < 4; ++j) {
                points.push_back(cornersVec[i][j]);
            }
        }
        RelatedPoints furthest2point = find2PointsFurthest(points);
        std::vector<std::vector<Point2f>> tempVec(cornersVec);
        tempVec[0] = cornersVec[furthest2point.index[0] / 4];
        tempVec[2] = cornersVec[furthest2point.index[1] / 4];
        tempVec[1] = cornersVec[3 - furthest2point.index[0] / 4 - furthest2point.index[1] / 4];
        cornersVec = tempVec;
        _0l = furthest2point.index[0] % 4;
        _2r = furthest2point.index[1] % 4;

        for (int i = 0; i < 4; ++i) {
            area = get3PointsArea(cornersVec[0][_0l], cornersVec[2][_2r], cornersVec[1][i]);
            if (abs(area) > tempMin) {
                p1 = cornersVec[1][i];
                _1m1 = i;
                tempMin = abs(area);
            }
            if (abs(area) < tempMax) {
                _1m2 = i;
                tempMax = abs(area);
            }
        }
        if (get3PointsArea(cornersVec[1][_1m2], cornersVec[1][_1m1], cornersVec[0][_0l]) >
            0) { //（y轴朝下）小于零代表0号块在左手，大于零0号块应在右手
            std::swap(cornersVec[0], cornersVec[2]);
        }
        p0 = cornersVec[0][_0l];
        p2 = cornersVec[2][_2r];
        float area0, area2, temp0 = -1, temp2 = -1;
        int _0m2, _2m2;
        for (int i = 0; i < 4; ++i) {
            area0 = get3PointsArea(p0, p2, cornersVec[0][i]);
            if (area0 > 0 && area0 > temp0) {
                temp0 = area0;
                _0m2 = i;
            }
            area2 = get3PointsArea(p0, p2, cornersVec[2][i]);
            if (area2 > 0 && area2 > temp2) {
                temp2 = area2;
                _2m2 = i;
            }
        }
        Line line0(p0, cornersVec[0][_0m2]), line1(p1, cornersVec[1][_1m2]), line2(p2, cornersVec[2][_2m2]);
        p3 = get3LinesIntersection(line0, line1, line2);
        std::vector<Point2f> qrSquareCorners = {p0, p1, p2, p3};
        sortPointsForPerspective(qrSquareCorners);
        std::cout << "Find 4 Points of QR Code" << std::endl;
        return qrSquareCorners;

    }


    void sharpen2D(const Mat &image,Mat &result)
    {
        // 首先构造一个内核
        Mat kernel(3,3,CV_32F,Scalar(0));
        /// 对 对应内核进行赋值
        kernel.at<float>(1,1) = 5.0;
        kernel.at<float>(0,1) = -1.0;
        kernel.at<float>(2,1) = -1.0;
        kernel.at<float>(1,0) = -1.0;
        kernel.at<float>(1,2) = -1.0;
        /// 对图像进行滤波操作
        filter2D(image,result,image.depth(),kernel);
    }


}