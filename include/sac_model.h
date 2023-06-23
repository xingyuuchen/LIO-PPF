#pragma once

#include <cfloat>
#include <ctime>
#include <climits>
#include <set>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/filters/extract_indices.h>


template<typename PointT, int kSampleSize, typename ModelT>
class SampleConsensusModel {
  public:
    using ModelType = ModelT;
    using PointCloud = typename pcl::PointCloud<PointT>;
    using PointCloudConstPtr = typename pcl::PointCloud<PointT>::ConstPtr;
    using PointCloudPtr = typename pcl::PointCloud<PointT>::Ptr;
    using SearchPtr = typename pcl::search::Search<PointT>::Ptr;
    
    using Ptr = boost::shared_ptr<SampleConsensusModel<PointT, kSampleSize, ModelT>>;
    using ConstPtr = boost::shared_ptr<const SampleConsensusModel<
            PointT, kSampleSize, ModelT>>;
    
    
  protected:
    explicit SampleConsensusModel(bool random = false)
            : input_(), indices_(), samples_radius_(0.), samples_radius_search_(),
              shuffled_indices_(), rng_alg_(), rng_dist_(new boost::uniform_int<>(0, std::numeric_limits<int>::max())),
              rng_gen_(), error_sqr_dists_() {
        rng_alg_.seed(random ? static_cast<unsigned>(std::time(nullptr)) : 12345u);
    
        all_inliers_.template reset(new PointCloud());
        
        rng_gen_.reset(new boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>(rng_alg_, *rng_dist_));
    }
  
  public:
    explicit SampleConsensusModel(const PointCloudPtr &cloud, bool random = false)
            : input_(), indices_(), samples_radius_(0.), samples_radius_search_(),
              shuffled_indices_(), rng_alg_(), rng_dist_(new boost::uniform_int<>(0, std::numeric_limits<int>::max())),
              rng_gen_(), error_sqr_dists_() {
        rng_alg_.seed(random ? static_cast<unsigned>(std::time(nullptr)) : 12345u);
        
        // Sets the input cloud and creates a vector of "fake" indices
        setInputCloud(cloud);
    
        all_inliers_.reset(new PointCloud());
    
        // Create a random number generator object
        rng_gen_.reset(new boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>(rng_alg_, *rng_dist_));
    }
    
    virtual ~SampleConsensusModel() = default;
    
    virtual void GetSamples(int &iterations, std::vector<int> &samples) {
        DLOG_ASSERT(indices_->size() >= kSampleSize);
        // get a second point which is different from the first
        samples.resize(kSampleSize);
        if (samples_radius_ < std::numeric_limits<double>::epsilon()) {
            drawIndexSample(samples);
        } else {
            drawIndexSampleRadius(samples);
        }
    }
    
    virtual bool ComputeModelCoef(const std::vector<int> &samples,
                                  ModelT &model_coef) = 0;
    
    virtual void OptimizeModel(const std::vector<int> &inliers,
                               ModelT &optimized_coef) = 0;
    
    virtual void getDistancesToModel(const ModelT &model_coef,
                                     std::vector<double> &distances) = 0;
    
    virtual int CountWithinDistance(const ModelT &model_coef,
                                    float dist_thresh) = 0;
    
    virtual void SelectWithinDistance(const ModelT &model_coef,
                                      float dist_thresh, std::vector<int> &inliers) = 0;
    
    virtual int RemoveWithinDistance(const ModelT &model_coef,
                                     float dist_thresh, pcl::IndicesPtr inliers_idx,
                                     bool is_append_all_inliers) = 0;
    
    virtual void projectPoints(const std::vector<int> &inliers,
                               const ModelT &model_coef,
                               PointCloud &projected_points,
                               bool copy_data_fields) = 0;
    
    /** \brief Verify whether a subset of indices verifies a given set of
      * model coefficients. Pure virtual.
      *
      * \param[in] indices the data indices that need to be tested against the model
      * \param[in] model_coef the set of model coefficients
      * \param[in] threshold a maximum admissible distance threshold for
      * determining the inliers from the outliers
      */
    virtual bool doSamplesVerifyModel(const std::set<int> &indices,
                                      const ModelT &model_coef,
                                      double threshold) = 0;
    
    inline void setInputCloud(const PointCloudPtr &cloud) {
        input_ = cloud;
//        if (!indices_)
            indices_.reset(new std::vector<int>());
        if (indices_->empty()) {
            // Prepare a set of indices to be used (entire cloud)
            indices_->resize(cloud->points.size());
            for (size_t i = 0; i < cloud->points.size(); ++i)
                (*indices_)[i] = static_cast<int> (i);
        }
        shuffled_indices_ = *indices_;
    }
    
    inline PointCloudPtr GetInputCloud() const { return input_; }
    
    inline PointCloudPtr GetInliersCloud() const { return all_inliers_; }
    
    inline size_t NumPoints() const { return input_->size(); }
    
    inline void setIndices(const boost::shared_ptr<std::vector<int>> &indices) {
        indices_ = indices;
        shuffled_indices_ = *indices_;
    }
    
    inline void setIndices(const std::vector<int> &indices) {
        indices_.reset(new std::vector<int>(indices));
        shuffled_indices_ = indices;
    }
    
    inline boost::shared_ptr<std::vector<int>> getIndices() const { return indices_; }
    
    inline const std::string &getClassName() const { return model_name_; }
    
    inline void SetSamplesMaxDist(const double &radius, SearchPtr search) {
        samples_radius_ = radius;
        samples_radius_search_ = search;
    }
    
    inline void GetSamplesMaxDist(double &radius) { radius = samples_radius_; }
    
//    friend class ProgressiveSampleConsensus<PointT>;
    
    /** \brief Compute the variance of the errors to the model.
      * \param[in] error_sqr_dists a vector holding the distances
      */
    inline double ComputeVariance(const std::vector<double> &error_sqr_dists) {
        std::vector<double> dists(error_sqr_dists);
        const size_t medIdx = dists.size() >> 1;
        std::nth_element(dists.begin(), dists.begin() + medIdx, dists.end());
        double median_error_sqr = dists[medIdx];
        return 2.1981 * median_error_sqr;
    }
    
    /** \brief Compute the variance of the errors to the model from the internally
      * estimated vector of distances. The model must be computed first (or at least
      * SelectWithinDistance must be called).
      */
    inline double ComputeVariance() {
        LOG_ASSERT(!error_sqr_dists_.empty()) << "The variance of the Sample Consensus model distances cannot be estimated, as the model has not been computed yet. Please compute the model first or at least run SelectWithinDistance before continuing.";
        return ComputeVariance(error_sqr_dists_);
    }
  
  protected:
    
    inline void drawIndexSample(std::vector<int> &sample) {
        size_t index_size = shuffled_indices_.size();
        for (unsigned int i = 0; i < kSampleSize; ++i) {
            // The 1/(RAND_MAX+1.0) trick is when the random numbers are not uniformly distributed and for small modulo
            // elements, that does not matter (and nowadays, random number generators are good)
            //std::swap (shuffled_indices_[i], shuffled_indices_[i + (rand () % (index_size - i))]);
            std::swap(shuffled_indices_[i], shuffled_indices_[i + (rnd() % (index_size - i))]);
        }
        std::copy(shuffled_indices_.begin(), shuffled_indices_.begin() + kSampleSize, sample.begin());
    }
    
    inline void drawIndexSampleRadiusWtf(std::vector<int> &sample) {
        size_t index_size = shuffled_indices_.size();
        
        std::swap(shuffled_indices_[0], shuffled_indices_[rnd() % index_size]);
        
        std::vector<int> indices;
        std::vector<float> sqr_dists;
        
        // If indices have been set when the search object was constructed,
        // radiusSearch() expects an index into the indices vector as its
        // first parameter. This can't be determined efficiently, so we use
        // the point instead of the index.
        // Returned indices are converted automatically.
        samples_radius_search_->radiusSearch(input_->at(shuffled_indices_[0]),
                                             samples_radius_, indices, sqr_dists);
    
        if (indices.size() < kSampleSize - 1) {
            for (unsigned int i = 1; i < kSampleSize; ++i)
                shuffled_indices_[i] = shuffled_indices_[0];
        } else {
            for (unsigned int i = 0; i < kSampleSize - 1; ++i)
                std::swap(indices[i], indices[i + (rnd() % (indices.size() - i))]);
            for (unsigned int i = 1; i < kSampleSize; ++i)
                shuffled_indices_[i] = indices[i - 1];
        }
        
        std::copy(shuffled_indices_.begin(), shuffled_indices_.begin() + kSampleSize, sample.begin());
    }
    
    inline void drawIndexSampleRadius(std::vector<int> &sample) {
        size_t index_size = shuffled_indices_.size();
        
        std::swap(shuffled_indices_[0], shuffled_indices_[rnd() % index_size]);
        
        std::vector<int> indices;
        std::vector<float> sqr_dists;
        
        samples_radius_search_->radiusSearch(input_->at(shuffled_indices_[0]),
                                             samples_radius_, indices, sqr_dists);
        /* Attention: indices[0] == shuffled_indices_[0] */
        DLOG_ASSERT(indices[0] == shuffled_indices_[0]);

        if (indices.size() < kSampleSize) {
            sample.clear();
        } else {
            for (unsigned int i = 1; i < kSampleSize; ++i)
                std::swap(indices[i], indices[i + (rnd() % (indices.size() - i))]);
            for (unsigned int i = 0; i < kSampleSize; ++i) {
                sample[i] = indices[i];
            }
        }
    }
    
    std::string model_name_;
    
    PointCloudPtr input_;
    
    PointCloudPtr all_inliers_;
    
    boost::shared_ptr<std::vector<int>> indices_;
    
    double samples_radius_;
    
    SearchPtr samples_radius_search_;
    
    std::vector<int> shuffled_indices_;
    
    boost::mt19937 rng_alg_;
    
    boost::shared_ptr<boost::uniform_int<>> rng_dist_;
    
    /** \brief Boost-based random number generator. */
    boost::shared_ptr<boost::variate_generator<boost::mt19937 &, boost::uniform_int<>>> rng_gen_;
    
    /** \brief A vector holding the distances to the computed model. Used internally. */
    std::vector<double> error_sqr_dists_;
    
    /** \brief Boost-based random number generator. */
    inline int rnd() { return rng_gen_->operator()(); }
  
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

