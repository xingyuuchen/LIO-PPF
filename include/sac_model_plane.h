#pragma once

#include <type_traits>
#include <omp.h>
#include <glog/logging.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/concatenate.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include "sac_model.h"
#include "sac_model_plane.h"
#include "reducible_vector.h"
//#include "timer.h"


#define Point2PlaneDist(p, plane_coef) fabsf(Point2PlaneDistSigned((p), (plane_coef)))

#define Point2PlaneDistSigned(p, plane_coef) ((plane_coef)[0] * (p).x + (plane_coef)[1] * (p).y + \
    (plane_coef)[2] * (p).z + (plane_coef)[3])


class PlaneWithCentroid {
  public:
    // ax + by + cz + d = 0
    Eigen::Vector4f coef;
    // centroid from the least square fitting.
    // note that d can be calculated from centroid, but not vice versa.
    Eigen::Vector3f centroid;
};


template<class PointT, int kSampleSize = 6, class PlaneModelT = PlaneWithCentroid>
class SampleConsensusModelPlane : public SampleConsensusModel<
        PointT, kSampleSize, PlaneModelT> {
  public:
    
    static_assert(kSampleSize >= 3, "cannot fit a plane with less than 3 points!");
    
    static constexpr bool kIsWithPca = kSampleSize > 3;
    static constexpr bool kIsWithCentroid = std::is_same<PlaneModelT, PlaneWithCentroid>();

    using PointCloud = typename SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::PointCloud;
    using PointCloudPtr = typename SampleConsensusModel<
            PointT, kSampleSize, PlaneModelT>::PointCloudPtr;
    using PointCloudConstPtr = typename SampleConsensusModel<
            PointT, kSampleSize, PlaneModelT>::PointCloudConstPtr;
    using Ptr = std::shared_ptr<SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>>;
    
    
    explicit SampleConsensusModelPlane(const PointCloudPtr &cloud,
                                       bool random = false, int num_parallel = 4);
    
    virtual ~SampleConsensusModelPlane() = default;
    
    bool ComputeModelCoef(const std::vector<int> &samples,
                          PlaneModelT &model_coef) override;
    
    bool FitPlaneFrom3points(const std::vector<int> &samples,
                             PlaneModelT &plane_coef);
    bool FitPlaneFromNPoints(const std::vector<int> &samples,
                             PlaneModelT &plane_coef);
    
    void getDistancesToModel(const PlaneModelT &plane,
                             std::vector<double> &distances) override;
    
    int CountWithinDistance(const PlaneModelT &plane,
                            float dist_thresh) override;
    
    void SelectWithinDistance(const PlaneModelT &plane,
                              float dist_thresh, std::vector<int> &inliers) override;
    
    int RemoveWithinDistance(const PlaneModelT &plane, float dist_thresh,
                             pcl::IndicesPtr inliers_idx, bool is_append_all_inliers) override;
    
    void OptimizeModel(const std::vector<int> &inliers,
                       PlaneModelT &optimized_plane) override;
    
    void projectPoints(const std::vector<int> &inliers,
                       const PlaneModelT &plane,
                       PointCloud &projected_points,
                       bool copy_data_fields) override;
    
    bool doSamplesVerifyModel(const std::set<int> &indices,
                              const PlaneModelT &plane,
                              double threshold) override;
    
  private:
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::model_name_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::input_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::all_inliers_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::indices_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::error_sqr_dists_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::samples_radius_search_;
    using SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::samples_radius_;
    int num_threads_;
};


template<class PointT, int kSampleSize, class PlaneModelT>
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::SampleConsensusModelPlane(
        const PointCloudPtr &cloud,
        bool random /* = false*/,
        int num_parallel/* = 4*/)
            : SampleConsensusModel<PointT, kSampleSize, PlaneModelT>(cloud, random)
            , num_threads_(num_parallel) {
    model_name_ = "SampleConsensusModelPlane";
}


template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::ComputeModelCoef(
        const std::vector<int> &samples, PlaneModelT &model_coef) {
    DLOG_ASSERT(samples.size() == kSampleSize);
//    if constexpr(kSampleSize > 3) {  // this is a c++17 extension
    if (kIsWithPca) {
        return FitPlaneFromNPoints(samples, model_coef);
    }
    return FitPlaneFrom3points(samples, model_coef);
}

template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::FitPlaneFrom3points(
        const std::vector<int> &samples,
        PlaneModelT &plane) {
    pcl::Array4fMap p0 = input_->points[samples[0]].getArray4fMap();
    pcl::Array4fMap p1 = input_->points[samples[1]].getArray4fMap();
    pcl::Array4fMap p2 = input_->points[samples[2]].getArray4fMap();
    
    Eigen::Array4f p1p0 = p1 - p0;
    Eigen::Array4f p2p0 = p2 - p0;
    
    // Avoid some crashes by checking for collinearity here
    // FIXME divided by zero
    Eigen::Array4f dy1dy2 = p1p0 / p2p0;
    if (dy1dy2[0] == dy1dy2[1] && dy1dy2[2] == dy1dy2[1]) {
        // check for collinearity
        return false;
    }
    
    // Compute the plane coefficients from the 3 given points in a straightforward manner
    // calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
    plane.coef[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
    plane.coef[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
    plane.coef[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
    plane.coef[3] = 0;
    plane.coef.normalize();
    plane.coef[3] = -1 * (plane.coef.dot(p0.matrix()));
    if (kIsWithCentroid) {
        // For efficiency, the centroid is calculated later
        // in OptimizedModel().
    }
    return true;
}

template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::FitPlaneFromNPoints(
        const std::vector<int> &samples,
        PlaneModelT &plane) {
    
    Eigen::Matrix<float, kSampleSize, 3> A;
    Eigen::Matrix<float, kSampleSize, 1> b;
    b.fill(-1);
    
    for (int i = 0; i < kSampleSize; ++i) {
        A(i, 0) = input_->points[samples[i]].x;
        A(i, 1) = input_->points[samples[i]].y;
        A(i, 2) = input_->points[samples[i]].z;
    }
    // solve plane coefficients: abc, d=1
    Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    
    float pa = x[0];
    float pb = x[1];
    float pc = x[2];
    float norm = sqrt(pa * pa + pb * pb + pc * pc);
    pa /= norm; pb /= norm; pc /= norm;
    float pd = (float) 1 / norm;

    // If distance between any one of the points and the plane exceeds 0.2 m,
    // the samples are considered bad.
    for (int i = 0; i < kSampleSize; ++i) {
        auto &p = input_->points[samples[i]];
        float dist = fabsf(pa * p.x + pb * p.y + pc * p.z + pd);
        if (dist > 0.2) {
            return false;
        }
    }
    plane.coef[0] = pa;
    plane.coef[1] = pb;
    plane.coef[2] = pc;
    plane.coef[3] = pd;
    if (kIsWithCentroid) {
        // For efficiency, the centroid is calculated later
        // in OptimizedModel().
    }
    return true;
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::getDistancesToModel(
        const PlaneModelT &plane, std::vector<double> &distances) {
    distances.resize(indices_->size());
    
    // Iterate through the 3d points and calculate the distances from them to the plane
    for (size_t i = 0; i < indices_->size(); ++i) {
        // Calculate the distance from the point to the plane normal as the dot product
        // D = (P-A).N/|N|
        /*distances[i] = fabs (model_coef[0] * input_->points[(*indices_)[i]].x +
                             model_coef[1] * input_->points[(*indices_)[i]].y +
                             model_coef[2] * input_->points[(*indices_)[i]].z +
                             model_coef[3]);*/
        Eigen::Vector4f pt(input_->points[(*indices_)[i]].x,
                           input_->points[(*indices_)[i]].y,
                           input_->points[(*indices_)[i]].z,
                           1);
        distances[i] = fabs(plane.coef.dot(pt));
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
int
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::CountWithinDistance(
        const PlaneModelT &plane, const float dist_thresh) {
    
    const int n_pts = (int) indices_->size();
    int nr_p = 0;
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as 'shared' in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) \
            shared(n_pts, plane, dist_thresh) reduction(+: nr_p)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) reduction(+: nr_p)
#endif
    for (int i = 0; i < n_pts; ++i) {
        float dist = Point2PlaneDist(input_->points[(*indices_)[i]], plane.coef);
        if (dist < dist_thresh) {
            nr_p++;
        }
    }
    return nr_p;
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::SelectWithinDistance(
        const PlaneModelT &plane, const float dist_thresh,
        std::vector<int> &inliers) {
    
    const int n_pts = indices_->size();
    
    inliers.clear();
    error_sqr_dists_.clear();
    inliers.reserve(n_pts);
    error_sqr_dists_.reserve(n_pts);
    
    const int n_each_thread = n_pts / num_threads_ + 1;
    // The ReducibleVector is 5.7x faster than using omp critical clause.
    OmpReducibleVector<int> reduce_inliers(n_each_thread, &inliers);
    OmpReducibleVector<double> reduce_sqr_dists(n_each_thread, &error_sqr_dists_);
    
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as shared in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) shared(n_pts, plane, dist_thresh) \
            reduction(merge_reducible_vec_i: reduce_inliers) \
            reduction(merge_reducible_vec_d: reduce_sqr_dists)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) \
            reduction(merge_reducible_vec_i: reduce_inliers) \
            reduction(merge_reducible_vec_d: reduce_sqr_dists)
#endif
    for (int i = 0; i < n_pts; ++i) {
        float dist = Point2PlaneDist(input_->points[(*indices_)[i]], plane.coef);
        if (dist < dist_thresh) {
            reduce_inliers.FastPushBack((*indices_)[i]);
            reduce_sqr_dists.FastPushBack(static_cast<double>(dist));
        }
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
int
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::RemoveWithinDistance(
        const PlaneModelT &plane, const float dist_thresh,
        pcl::IndicesPtr inliers_idxs, bool is_append_all_inliers) {
    DLOG_ASSERT(indices_->size() == input_->size());
    const int n_pts = (int) indices_->size();
    if (inliers_idxs) {
        inliers_idxs->clear();
    } else {
        inliers_idxs = pcl::IndicesPtr(new std::vector<int>);
    }
    const float midlier_dist_thresh = 4 * dist_thresh;
    pcl::IndicesPtr midliers_idxs(new std::vector<int>);
    
    inliers_idxs->reserve(n_pts);
    midliers_idxs->reserve(n_pts);
    
    const int n_each_thread = n_pts / num_threads_ + 1;
    // FIXME(xingyuuchen): Mixed usage of shared_ptr and raw ptr.
    OmpReducibleVector<int> reduce_inliers(n_each_thread, inliers_idxs.get());
    OmpReducibleVector<int> reduce_midliers(n_each_thread, midliers_idxs.get());
    
#if defined(__GNUC__) && (__GNUC__ >= 9)
/* variables with const qualifier will not be auto pre-determined
 * as shared in omp in higher gcc version */
#pragma omp parallel for num_threads(num_threads_) default(none) \
        shared(n_pts, plane, dist_thresh, midlier_dist_thresh) \
        reduction(merge_reducible_vec_i: reduce_inliers, reduce_midliers)
#else
#pragma omp parallel for num_threads(num_threads_) default(none) shared(plane) \
        reduction(merge_reducible_vec_i: reduce_inliers, reduce_midliers)
#endif
    for (int i = 0; i < n_pts; ++i) {
        int idx = (*indices_)[i];
        float dist = Point2PlaneDist(input_->points[idx], plane.coef);
        if (dist < dist_thresh) {
            reduce_inliers.FastPushBack(idx);
        } else if (dist < midlier_dist_thresh) {
            reduce_midliers.FastPushBack(idx);
        }
    }
    
    int n_remove = (int) inliers_idxs->size();
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input_);
    extract.setIndices(inliers_idxs);
    if (is_append_all_inliers) {
        PointCloud inliers_cloud;
        extract.setNegative(false);
        extract.filter(inliers_cloud);
        *all_inliers_ += inliers_cloud;
    }
    if (!midliers_idxs->empty()) {
        n_remove += midliers_idxs->size();
        midliers_idxs->insert(midliers_idxs->end(),
             inliers_idxs->begin(), inliers_idxs->end());
        extract.setIndices(midliers_idxs);
    }
    extract.setNegative(true);
    extract.filter(*input_);
    
    if (samples_radius_search_) {
        DLOG_ASSERT(samples_radius_ > 0);
        samples_radius_search_->setInputCloud(input_);
    }
    
    SampleConsensusModel<PointT, kSampleSize, PlaneModelT>::setInputCloud(input_);
    return n_remove;
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::OptimizeModel(
        const std::vector<int> &inliers, PlaneModelT &optimized_plane) {
    // Need more than the minimum sample size to make a difference
    DLOG_ASSERT(inliers.size() > kSampleSize) << " " << inliers.size();
    
    PlaneModelT plane_parameters;
    
    // Use Least-Squares to fit the plane through all the given
    // sample points and find out its coefficients
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    Eigen::Vector4f xyz_centroid;
    
    // TODO(xingyuuchen): OpenMp speed up
    computeMeanAndCovarianceMatrix(*input_, inliers,
                                   covariance_matrix, xyz_centroid);
    
    // Compute the model coefficients
    EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
    EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
    pcl::eigen33(covariance_matrix, eigen_value, eigen_vector);
    
    // Hessian form (D = nc . p_plane (centroid here) + p)
    optimized_plane.coef[0] = eigen_vector[0];
    optimized_plane.coef[1] = eigen_vector[1];
    optimized_plane.coef[2] = eigen_vector[2];
    optimized_plane.coef[3] = 0;
    // noting the eigen_vector here is already normalized
    optimized_plane.coef[3] = -1 * (optimized_plane.coef.dot(xyz_centroid));
    if (kIsWithCentroid) {
        optimized_plane.centroid = xyz_centroid.template head<3>();
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
void
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::projectPoints(
        const std::vector<int> &inliers, const PlaneModelT &plane,
        PointCloud &projected_points,
        bool copy_data_fields) {
    
    projected_points.header = input_->header;
    projected_points.is_dense = input_->is_dense;
    
    Eigen::Vector4f mc(plane.coef[0], plane.coef[1], plane.coef[2], 0);
    
    // normalize the vector perpendicular to the plane...
    mc.normalize();
    // ... and store the resulting normal as a local copy of the model coefficients
    Eigen::Vector4f tmp_mc = plane.coef;
    tmp_mc[0] = mc[0];
    tmp_mc[1] = mc[1];
    tmp_mc[2] = mc[2];
    
    // Copy all the data fields from the input cloud to the projected one?
    if (copy_data_fields) {
        // Allocate enough space and copy the basics
        projected_points.points.resize(input_->points.size());
        projected_points.width = input_->width;
        projected_points.height = input_->height;
        
        typedef typename pcl::traits::fieldList<PointT>::type FieldList;
        // Iterate over each point
        for (size_t i = 0; i < input_->points.size(); ++i)
            // Iterate over each dimension
            pcl::for_each_type<FieldList>(pcl::NdConcatenateFunctor<PointT, PointT>(
                    input_->points[i], projected_points.points[i]));
        
        // Iterate through the 3d points and calculate the distances from them to the plane
        for (size_t i = 0; i < inliers.size(); ++i) {
            // Calculate the distance from the point to the plane
            Eigen::Vector4f p(input_->points[inliers[i]].x,
                              input_->points[inliers[i]].y,
                              input_->points[inliers[i]].z,
                              1);
            // use normalized coefficients to calculate the scalar projection
            float distance_to_plane = tmp_mc.dot(p);
            
            pcl::Vector4fMap pp = projected_points.points[inliers[i]].getVector4fMap();
            pp.matrix() = p - mc * distance_to_plane;        // mc[3] = 0, therefore the 3rd coordinate is safe
        }
    } else {
        // Allocate enough space and copy the basics
        projected_points.points.resize(inliers.size());
        projected_points.width = static_cast<uint32_t> (inliers.size());
        projected_points.height = 1;
        
        typedef typename pcl::traits::fieldList<PointT>::type FieldList;
        // Iterate over each point
        for (size_t i = 0; i < inliers.size(); ++i)
            // Iterate over each dimension
            pcl::for_each_type<FieldList>(pcl::NdConcatenateFunctor<PointT, PointT>(
                    input_->points[inliers[i]], projected_points.points[i]));
        
        // Iterate through the 3d points and calculate the distances from them to the plane
        for (size_t i = 0; i < inliers.size(); ++i) {
            // Calculate the distance from the point to the plane
            Eigen::Vector4f p(input_->points[inliers[i]].x,
                              input_->points[inliers[i]].y,
                              input_->points[inliers[i]].z,
                              1);
            // use normalized coefficients to calculate the scalar projection
            float distance_to_plane = tmp_mc.dot(p);
            
            pcl::Vector4fMap pp = projected_points.points[i].getVector4fMap();
            // mc[3] = 0, therefore the 3rd coordinate is safe
            pp.matrix() = p - mc * distance_to_plane;
        }
    }
}


template<class PointT, int kSampleSize, class PlaneModelT>
bool
SampleConsensusModelPlane<PointT, kSampleSize, PlaneModelT>::doSamplesVerifyModel(
        const std::set<int> &indices, const PlaneModelT &plane,
        const double threshold) {
    
    for (const int &idx : indices) {
        Eigen::Vector4f pt(input_->points[idx].x,
                           input_->points[idx].y,
                           input_->points[idx].z,
                           1);
        if (fabs(plane.coef.dot(pt)) > threshold) {
            return false;
        }
    }
    return true;
}


