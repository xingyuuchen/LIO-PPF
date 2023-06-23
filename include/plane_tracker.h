#pragma once

#include <vector>
#include <list>
#include <glog/logging.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <ros/node_handle.h>
#include "sac_model_plane.h"
#include "ransac.h"
//#include "timer.h"



template<class PointType>
class PlaneTracker {
  public:
    
    using PointCloudPtr = typename pcl::PointCloud<PointType>::Ptr;
    
    explicit PlaneTracker(int n_max_planes = 5);
    
    void ExtractLargePlanes(PointCloudPtr, float *rpyxyz_wl_init_guess = nullptr);

    void ExtractLargePlanes(PointCloudPtr, Eigen::Vector3f *t_wl_init_guess = nullptr,
                            Eigen::Quaternionf *q_wl_init_guess = nullptr);

    void ExtractLargePlanes(PointCloudPtr, Eigen::Matrix4f *T_wl_init_guess = nullptr);

    void SetCurrTransWl(float *rpyxyz_wl);
    
    void SetCurrTransWl(Eigen::Vector3f &t_wl, Eigen::Quaternionf &q_wl);
    
    void TransCurrPlanes(Eigen::Matrix4f &trans, Eigen::Matrix4f &trans_inv);

    void SetPlaneLeastInliers(int n_least_inliers);
    
    inline void SetIncrementalFitting(bool is_incremental_fitting) {
        is_incremental_fitting_ = is_incremental_fitting;
    }

    inline int NumPlanes() const { return all_planes_world_.size(); }
    
    inline std::vector<PlaneWithCentroid> &LastPlanes() { return all_planes_world_; }
    
    inline PointCloudPtr CurrCloudPlane() { return curr_cloud_plane_; }
    
    inline PointCloudPtr CurrCloudOther() { return curr_cloud_other_; }
  
  private:
    
    int n_least_points_on_a_plane_;
    int n_max_planes_;
    bool is_incremental_fitting_;
    
    PointCloudPtr curr_cloud_;
    PointCloudPtr curr_cloud_plane_;
    PointCloudPtr curr_cloud_other_;
    
    std::vector<PlaneWithCentroid> curr_planes_local_;
    std::vector<PlaneWithCentroid> all_planes_world_;
    
    static const double kAngleEpsilon;
};


template<class PointType>
const double PlaneTracker<PointType>::kAngleEpsilon = std::sin(3 * M_PI / 180);


template<class PointType>
PlaneTracker<PointType>::PlaneTracker(int n_max_planes /* = 5*/)
        : n_least_points_on_a_plane_(-1)
        , n_max_planes_(n_max_planes)
        , is_incremental_fitting_(false)
        , curr_cloud_plane_(nullptr)
        , curr_cloud_other_(nullptr) {
 
    curr_cloud_.reset(new pcl::PointCloud<PointType>());
}

template<class PointType>
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        float *rpyxyz_wl_init_guess/* = nullptr*/) {
    if (!is_incremental_fitting_) {
        ExtractLargePlanes(pc, (Eigen::Matrix4f *) nullptr);
        return;
    }
    LOG_ASSERT(rpyxyz_wl_init_guess != nullptr) <<
             "Please provide initial guess if using incremental fitting.";
    Eigen::Affine3f T_w_lnp1;
    pcl::getTransformation(rpyxyz_wl_init_guess[3], rpyxyz_wl_init_guess[4], rpyxyz_wl_init_guess[5], rpyxyz_wl_init_guess[0],
                       rpyxyz_wl_init_guess[1], rpyxyz_wl_init_guess[2], T_w_lnp1);
    ExtractLargePlanes(pc, &(T_w_lnp1.matrix()));
}


template<class PointType>
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        Eigen::Vector3f *t_wl_init_guess/* = nullptr*/,
                        Eigen::Quaternionf *q_wl_init_guess/* = nullptr*/) {
    if (!is_incremental_fitting_) {
        ExtractLargePlanes(pc, (Eigen::Matrix4f *) nullptr);
        return;
    }
    LOG_ASSERT(t_wl_init_guess != nullptr && q_wl_init_guess != nullptr) <<
             "Please provide initial guess if using incremental fitting.";
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.template block<3, 3>(0, 0) = q_wl_init_guess->toRotationMatrix();
    trans.template block<3, 1>(0, 3) = *t_wl_init_guess;
    ExtractLargePlanes(pc, &trans);
}

template<class PointType>
void PlaneTracker<PointType>::ExtractLargePlanes(PointCloudPtr pc,
                        Eigen::Matrix4f *T_wl_init_guess/* = nullptr*/) {
    LOG_ASSERT(n_least_points_on_a_plane_ > 0) <<
         "illegal n_least_points_on_a_plane_: " << n_least_points_on_a_plane_;
    pcl::copyPointCloud(*pc, *curr_cloud_);
    size_t original = curr_cloud_->size();

    std::vector<PlaneWithCentroid> planes_init_guess;
    if (is_incremental_fitting_ && !LastPlanes().empty()) {
        LOG_ASSERT(T_wl_init_guess != nullptr) <<
             "Please provide initial guess if using incremental fitting.";
        // compute plane initial guess here
        for (auto &plane_world : all_planes_world_) {
            PlaneWithCentroid plane_lnp1;
            for (int i = 0; i < 4; ++i) {
                plane_lnp1.coef[i] = plane_world.coef[0] * (*T_wl_init_guess)(0, i) + plane_world.coef[1] * (*T_wl_init_guess)(1, i) +
                                     plane_world.coef[2] * (*T_wl_init_guess)(2, i) + plane_world.coef[3] * (*T_wl_init_guess)(3, i);
            }
            planes_init_guess.template emplace_back(plane_lnp1);
        }
    }

    constexpr int kSampleSize = 3;
    using RansacModel = SampleConsensusModelPlane<PointType, kSampleSize>;
    typename RansacModel::Ptr model_plane(new RansacModel(curr_cloud_));
    RansacWithPca<RansacModel> ransac(model_plane, n_max_planes_, n_least_points_on_a_plane_);
    
    ransac.SetDistanceThreshold(0.1);
    
    if (kSampleSize > 3) {
        typename pcl::search::Search<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(curr_cloud_);
        ransac.GetSampleConsensusModel()->SetSamplesMaxDist(10, tree);
        ransac.setMaxIterations(20);
    } else {
        ransac.setMaxIterations(300);
    }
    ransac.ComputeModel(planes_init_guess);
    
    curr_planes_local_ = ransac.AllModelsCoef();
    curr_cloud_plane_ = ransac.GetSampleConsensusModel()->GetInliersCloud();
    // curr_cloud_ is now the remaining points
    curr_cloud_other_ = curr_cloud_;
}


template <class PointType>
void PlaneTracker<PointType>::SetCurrTransWl(float *rpyxyz_wl) {
    Eigen::Affine3f trans;
    pcl::getTransformation(rpyxyz_wl[3], rpyxyz_wl[4], rpyxyz_wl[5], rpyxyz_wl[0],
                           rpyxyz_wl[1], rpyxyz_wl[2], trans);

    TransCurrPlanes(trans.matrix(), trans.inverse().matrix());
}

template <class PointType>
void PlaneTracker<PointType>::SetCurrTransWl(Eigen::Vector3f &t_wl,
                                             Eigen::Quaternionf &q_wl) {
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.template block<3, 3>(0, 0) = q_wl.toRotationMatrix();
    trans.template block<3, 1>(0, 3) = t_wl;

    Eigen::Matrix3f R_inv = q_wl.conjugate().toRotationMatrix();
    Eigen::Vector3f t_inv = -(R_inv * t_wl);
    Eigen::Matrix4f trans_inv = Eigen::Matrix4f::Identity();
    trans_inv.template block<3, 3>(0, 0) = R_inv;
    trans_inv.template block<3, 1>(0, 3) = t_inv;

    TransCurrPlanes(trans, trans_inv);
}

template <class PointType>
void PlaneTracker<PointType>::TransCurrPlanes(Eigen::Matrix4f &trans, Eigen::Matrix4f &trans_inv) {
    if (is_incremental_fitting_) {
        // If using incremental fitting, some new planes are obtained by checking and
        // refining from old planes, therefore, clear old planes to avoid duplication.
        all_planes_world_.clear();
    }
    std::vector<PlaneWithCentroid> new_planes;

    for (auto &local_plane : curr_planes_local_) {
        PlaneWithCentroid world_plane;
        for (int i = 0; i < 4; ++i) {
            world_plane.coef[i] = local_plane.coef[0] * trans_inv(0, i) + local_plane.coef[1] * trans_inv(1, i) +
                                  local_plane.coef[2] * trans_inv(2, i) + local_plane.coef[3] * trans_inv(3, i);
        }
        // TODO: use homogenous coordinate
        for (int i = 0; i < 3; ++i) {
            world_plane.centroid[i] = trans(i, 0) * local_plane.centroid[0] + trans(i, 1) * local_plane.centroid[1] +
                                      trans(i, 2) * local_plane.centroid[2] + trans(i, 3);
        }
        new_planes.template emplace_back(world_plane);
    }
    
    // This does not affect performance too much as there wouldn't be too many planes.
    all_planes_world_.insert(all_planes_world_.begin(),
            new_planes.begin(), new_planes.end());

    while (all_planes_world_.size() > n_max_planes_) {
        all_planes_world_.pop_back();
    }
}

template <class PointType>
void PlaneTracker<PointType>::SetPlaneLeastInliers(int n_least_inliers) {
    LOG_ASSERT(n_least_inliers > 0);
    n_least_points_on_a_plane_ = n_least_inliers;
}

