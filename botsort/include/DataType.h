#pragma once

#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <optional>
#include <utility>
#include <vector>

// #include <opencv2/core.hpp>


constexpr uint8_t DET_ELEMENTS = 4;
constexpr uint32_t FEATURE_DIM = 512;
constexpr uint8_t KALMAN_STATE_SPACE_DIM = 8;
constexpr uint8_t KALMAN_MEASUREMENT_SPACE_DIM = 4;

// Detection
/**
 * @brief Detection vector with DET_ELEMENTS elements.
 */
using DetVec = Eigen::Matrix<float, 1, DET_ELEMENTS>;

namespace bot_sort {

/**
 * @brief Struct representing a Rectangle. This replaces cv::Rect_<> to avoid including OpenCV in the header file.
 * 
 * T x: X coordinate of the top-left corner
 * T y: Y coordinate of the top-left corner
 * T width: Width of the rectangle
 * T height: Height of the rectangle
 */
template<typename T>
class Rect_ {    
public:    
    Rect_() = default;
    Rect_(const T &_x, const T &_y, const T &_width, const T &_height){
        x = _x;
        y = _y;
        width = _width;
        height = _height;
    };

    ~Rect_() {};

    T x, y, width, height;
};



/**
 * @brief Struct representing a detection
 * 
 * cv::Rect_<float> bbox_tlwh: Bounding box of the detection in the format (top left x, top left y, width, height)
 * int class_id: Class ID of the detection
 * float confidence: Confidence score of the detection
 */
struct Detection {
    bot_sort::Rect_<float> bbox_tlwh;
    std::string class_id;
    float   confidence; // score of detection
    double *featArray;  // Pointer to the feature array, if Re-ID is enabled
    int     featDim;    // Dimension of the feature vector, if Re-ID is enabled
    int     detId;      // Optional detection ID

    Detection(const bot_sort::Rect_<float> &_rect,const std::string &_class_id, const float &_confidence,
              double *_featArray = nullptr, int _featDim = 0, int _detId = -1): 
                bbox_tlwh(_rect), class_id(_class_id),confidence(_confidence),
                featArray(_featArray),featDim(_featDim), detId(_detId) {};
};

} // namespace bot_sort


// Re-ID Features
/**
 * @brief Re-ID feature vector with FEATURE_DIM elements.
 */
using FeatureVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;
/**
 * @brief Re-ID feature matrix with dynamic rows and FEATURE_DIM columns.
 */
using FeatureMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;


// Kalman Filter
/**
 * @brief Kalman Filter state space vector with KALMAN_STATE_SPACE_DIM elements.
 */
using KFStateSpaceVec = Eigen::Matrix<float, 1, KALMAN_STATE_SPACE_DIM>;
/**
 * @brief Kalman Filter state space matrix with KALMAN_STATE_SPACE_DIM rows and columns.
 */
using KFStateSpaceMatrix =
        Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM>;
/**
 * @brief Kalman Filter state space data containing a mean vector and a covariance matrix.
 */
using KFDataStateSpace = std::pair<KFStateSpaceVec, KFStateSpaceMatrix>;

/**
 * @brief Kalman Filter measurement space vector with KALMAN_MEASUREMENT_SPACE_DIM elements.
 */
using KFMeasSpaceVec = Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM>;
/**
 * @brief Kalman Filter measurement space matrix with KALMAN_MEASUREMENT_SPACE_DIM rows and columns.
 */
using KFMeasSpaceMatrix = Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM,
                                        KALMAN_MEASUREMENT_SPACE_DIM>;
/**
 * @brief Kalman Filter measurement space data containing a mean vector and a covariance matrix.
 */
using KFDataMeasurementSpace = std::pair<KFMeasSpaceVec, KFMeasSpaceMatrix>;


// Camera Motion Compensation
/**
 * @brief 3x3 homography matrix.
 * 
 */
using HomographyMatrix = Eigen::Matrix<float, 3, 3>;


// Tracker
/**
 * @brief Tracker data containing a track ID and a feature vector.
 */
using TrackerData = std::pair<int, FeatureVector>;
/**
 * @brief Match data containing a track ID and a detection ID.
 */
using MatchData = std::pair<int, int>;


// Association and Linear Assignment
/**
 * @brief Cost matrix for linear assignment with dynamic rows and columns.
 */
using CostMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
/**
 * @brief Association data containing matched and unmatched tracks and detections.
 */
struct AssociationData
{
    std::vector<MatchData> matches;///< Matched track and detection pairs.
    std::vector<int> unmatched_track_indices;///< Unmatched track indices.
    std::vector<int> unmatched_det_indices;  ///< Unmatched detection indices
};
